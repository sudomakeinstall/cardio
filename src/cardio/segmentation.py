import functools
import logging

import itk
import vtk
import numpy as np
import pydantic as pc
from vtkmodules.vtkCommonDataModel import vtkPlanes

# from vtkmodules.vtkFiltersModeling import vtkSurfaceNets3D
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer

from . import Object
from .property_config import PropertyConfig
from .utils import reset_direction, InterpolatorType


class Segmentation(Object):
    """Segmentation object with multi-label mesh extraction using SurfaceNets."""

    actors: list[vtkActor] = pc.Field(default_factory=list, exclude=True)
    property_config: PropertyConfig = pc.Field(default=None, exclude=True)
    include_labels: list[int] | None = pc.Field(default=None)
    label_properties: dict[int, dict] = pc.Field(default_factory=dict)

    def __init__(self, cfg: dict, renderer: vtkRenderer):
        include_labels = cfg.get("include_labels", None)
        label_properties = cfg.get("label_properties", {})

        super().__init__(
            label=cfg["label"],
            directory=cfg["directory"],
            pattern=cfg.get("pattern", "${frame}.nii.gz"),
            visible=cfg["visible"],
            renderer=renderer,
            clipping_enabled=cfg.get("clipping_enabled", True),
            include_labels=include_labels,
            label_properties=label_properties,
            property_config=PropertyConfig.model_validate(cfg.get("property", {})),
        )

        # Generate actors for all frames
        self._generate_actors()

    def _generate_actors(self):
        """Generate VTK actors for all frames using SurfaceNets3D."""
        for frame, path in enumerate(self.path_list):
            logging.info(f"{self.label}: Loading segmentation frame {frame}.")

            # Read and process segmentation image
            image = itk.imread(path)
            image = reset_direction(image, InterpolatorType.NEAREST)
            vtk_image = itk.vtk_image_from_image(image)

            # Create SurfaceNets3D filter
            surface_nets = vtk.vtkSurfaceNets3D()
            surface_nets.SetInputData(vtk_image)

            # Configure label selection if specified
            if self.include_labels is not None:
                surface_nets.SetOutputStyle(surface_nets.OUTPUT_STYLE_SELECTED)
                surface_nets.InitializeSelectedLabelsList()
                for label in self.include_labels:
                    surface_nets.AddSelectedLabel(label)

            # Execute filter
            surface_nets.Update()
            mesh = surface_nets.GetOutput()

            # Create scalar array from boundary labels (use higher value)
            boundary_labels = mesh.GetCellData().GetArray("BoundaryLabels")

            if boundary_labels:
                from vtkmodules.vtkCommonCore import vtkIntArray

                # Create scalar array using the maximum of the two boundary labels
                scalar_array = vtkIntArray()
                scalar_array.SetName("Labels")
                scalar_array.SetNumberOfTuples(boundary_labels.GetNumberOfTuples())

                for i in range(boundary_labels.GetNumberOfTuples()):
                    label1 = int(boundary_labels.GetComponent(i, 0))
                    label2 = int(boundary_labels.GetComponent(i, 1))
                    # Use the higher label value (excluding background=0)
                    max_label = (
                        max(label1, label2)
                        if max(label1, label2) > 0
                        else min(label1, label2)
                    )
                    scalar_array.SetValue(i, max_label)

                mesh.GetCellData().SetScalars(scalar_array)

            # Create single actor with scalar coloring
            actor = self._create_segmentation_actor(mesh)
            self.actors.append(actor)

    def _create_segmentation_actor(self, mesh):
        """Create a VTK actor with scalar-based coloring for the segmentation mesh."""
        from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

        # Create a mapper with scalar coloring
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.SetScalarModeToUseCellData()
        mapper.ScalarVisibilityOn()

        # Create color transfer function for label-based coloring
        color_func = vtkColorTransferFunction()

        # Get the label range from the scalar array
        scalar_array = mesh.GetCellData().GetArray("Labels")
        if scalar_array:
            scalar_range = scalar_array.GetRange()
            min_label = int(scalar_range[0])
            max_label = int(scalar_range[1])

            # Set colors for each label
            for label in range(min_label, max_label + 1):
                if label == 0:  # Skip background
                    continue

                if label in self.label_properties:
                    props = self.label_properties[label]
                    color_func.AddRGBPoint(
                        label,
                        props.get("r", 1.0),
                        props.get("g", 0.0),
                        props.get("b", 0.0),
                    )
                else:
                    # Default coloring based on label value
                    color = self._get_default_color(label)
                    color_func.AddRGBPoint(label, *color)

            mapper.SetLookupTable(color_func)
            mapper.SetScalarRange(min_label, max_label)

        # Create actor
        actor = vtkActor()
        actor.SetMapper(mapper)

        return actor

    def _get_default_color(self, label):
        """Generate a default color for a label."""
        # Simple color generation based on label value
        np.random.seed(label)
        return np.random.rand(3)

    def setup_pipeline(self, frame: int):
        """Setup the rendering pipeline for a specific frame."""
        for actor in self.actors:
            self.renderer.AddActor(actor)
            actor.SetVisibility(False)
            # Apply base property configuration if available
            base_prop = self.property_config.vtk_property
            if base_prop:
                # Note: For scalar-colored actors, we preserve the color transfer function
                # by not overriding the mapper's lookup table
                pass

        if self.visible and frame < len(self.actors):
            self.actors[frame].SetVisibility(True)

        self.renderer.ResetCamera()

    def toggle_clipping(self, enabled: bool):
        """Enable or disable clipping for all segmentation actors."""
        if not self.actors:
            return

        if enabled and self.clipping_planes:
            # Apply clipping to all actors
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.SetClippingPlanes(self.clipping_planes)
        else:
            # Remove clipping from all actors
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.RemoveAllClippingPlanes()

    def update_clipping_bounds(self, bounds):
        """Update clipping bounds from UI controls."""
        if not self.clipping_planes:
            return

        # Update clipping planes with new bounds
        super()._create_clipping_planes_from_bounds(self.clipping_planes, bounds)

        # Apply to all actors if clipping is enabled
        if self.clipping_enabled:
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.SetClippingPlanes(self.clipping_planes)
