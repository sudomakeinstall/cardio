import logging
import typing as ty

import itk
import numpy as np
import pydantic as pc
import vtk

from .object import Object
from .orientation import (
    read_frames,
)
from .property_config import vtkPropertyConfig
from .utils import label_color

_MARKER_ARRAY = "_snap_marker"


def masked_centroid(
    mesh: vtk.vtkPolyData, mask: ty.Sequence[bool]
) -> list[float] | None:
    """Center of mass of the mesh cells selected by ``mask``."""
    if not any(mask):
        return None

    marker = vtk.vtkIntArray()
    marker.SetName(_MARKER_ARRAY)
    marker.SetNumberOfTuples(len(mask))
    for i, selected in enumerate(mask):
        marker.SetValue(i, 1 if selected else 0)

    masked = vtk.vtkPolyData()
    masked.ShallowCopy(mesh)
    masked.GetCellData().AddArray(marker)
    masked.GetCellData().SetActiveScalars(_MARKER_ARRAY)

    thresh = vtk.vtkThreshold()
    thresh.SetInputData(masked)
    thresh.SetLowerThreshold(1)
    thresh.SetUpperThreshold(1)
    thresh.SetThresholdFunction(thresh.THRESHOLD_BETWEEN)
    thresh.Update()
    if thresh.GetOutput().GetNumberOfCells() == 0:
        return None

    geom = vtk.vtkGeometryFilter()
    geom.SetInputConnection(thresh.GetOutputPort())
    geom.Update()

    com = vtk.vtkCenterOfMass()
    com.SetInputConnection(geom.GetOutputPort())
    com.SetUseScalarsAsWeights(False)
    com.Update()
    return list(com.GetCenter())


class Segmentation(Object):
    """Segmentation object with multi-label mesh extraction using SurfaceNets."""

    pattern: str = pc.Field(
        default="{frame}.nii.gz",
        description="Filename pattern with $frame placeholder",
    )
    _actors: list[vtk.vtkActor] = pc.PrivateAttr(default_factory=list)
    _meshes: list[vtk.vtkPolyData] = pc.PrivateAttr(default_factory=list)
    _label_images: list[vtk.vtkImageData] = pc.PrivateAttr(default_factory=list)
    _mpr_actors: dict[int, dict[str, dict]] = pc.PrivateAttr(default_factory=dict)
    properties: vtkPropertyConfig = pc.Field(
        default_factory=vtkPropertyConfig, description="Property configuration"
    )
    include_labels: list[int] | None = pc.Field(default=None)
    label_properties: dict[int, dict] = pc.Field(default_factory=dict)

    @pc.model_validator(mode="after")
    def initialize_segmentation(self):
        """Generate VTK actors for all frames using SurfaceNets3D."""
        for path in self.path_list:
            for image in read_frames(path):
                logging.info(
                    f"{self.label}: Loading segmentation frame {len(self._actors)}."
                )

                vtk_image = itk.vtk_image_from_image(image)
                self._label_images.append(vtk_image)

                mesh = self._extract_mesh(vtk_image)
                self._meshes.append(mesh)
                self._actors.append(self._create_segmentation_actor(mesh))

        return self

    def _extract_mesh(self, vtk_image) -> vtk.vtkPolyData:
        """Extract a multi-label surface mesh with per-cell label scalars."""
        surface_nets = vtk.vtkSurfaceNets3D()
        surface_nets.SetInputData(vtk_image)
        max_label = int(vtk_image.GetPointData().GetScalars().GetRange()[1])
        surface_nets.GenerateLabels(max_label, 1, max_label)

        if self.include_labels is not None:
            surface_nets.SetOutputStyle(surface_nets.OUTPUT_STYLE_SELECTED)
            surface_nets.InitializeSelectedLabelsList()
            for label in self.include_labels:
                surface_nets.AddSelectedLabel(label)

        surface_nets.Update()
        mesh = surface_nets.GetOutput()

        boundary_labels = mesh.GetCellData().GetArray("BoundaryLabels")

        if boundary_labels:
            scalar_array = vtk.vtkIntArray()
            scalar_array.SetName("Labels")
            scalar_array.SetNumberOfTuples(boundary_labels.GetNumberOfTuples())

            for i in range(boundary_labels.GetNumberOfTuples()):
                label1 = int(boundary_labels.GetComponent(i, 0))
                label2 = int(boundary_labels.GetComponent(i, 1))
                # Prefer the foreground label of the pair over background=0
                cell_label = (
                    max(label1, label2)
                    if max(label1, label2) > 0
                    else min(label1, label2)
                )
                scalar_array.SetValue(i, cell_label)

            mesh.GetCellData().AddArray(scalar_array)
            mesh.GetCellData().SetActiveScalars("Labels")

        return mesh

    @property
    def actors(self) -> list[vtk.vtkActor]:
        return self._actors

    def _create_segmentation_actor(self, mesh):
        """Create a VTK actor with scalar-based coloring for the segmentation mesh."""
        # Create a mapper with scalar coloring
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.SetScalarModeToUseCellData()
        mapper.ScalarVisibilityOn()

        # Create color transfer function for label-based coloring
        color_func = vtk.vtkColorTransferFunction()

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
                    color_func.AddRGBPoint(label, *label_color(label))

            mapper.SetLookupTable(color_func)
            mapper.SetScalarRange(min_label, max_label)

        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def configure_actors(self):
        """Configure actor properties without adding to renderer."""
        for actor in self._actors:
            actor.SetVisibility(False)
            # Apply base property configuration if available
            base_prop = self.properties.vtk_property
            if base_prop:
                # Note: For scalar-colored actors, we preserve the color transfer function
                # by not overriding the mapper's lookup table
                pass

    def toggle_clipping(self, enabled: bool):
        """Enable or disable clipping for all segmentation actors."""
        if not self._actors:
            return

        if enabled and self.clipping_planes:
            # Apply clipping to all actors
            for actor in self._actors:
                mapper = actor.GetMapper()
                mapper.SetClippingPlanes(self.clipping_planes)
        else:
            # Remove clipping from all actors
            for actor in self._actors:
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
            for actor in self._actors:
                mapper = actor.GetMapper()
                mapper.SetClippingPlanes(self.clipping_planes)

    def create_mpr_actors(self, frame: int = 0):
        """Create MPR actors for axial, sagittal, and coronal views."""
        if frame >= len(self._label_images):
            frame = 0

        image_data = self._label_images[frame]
        mpr_actors = {}

        for orientation in ["axial", "sagittal", "coronal"]:
            reslice = vtk.vtkImageReslice()
            reslice.SetInputData(image_data)
            reslice.SetOutputDimensionality(2)
            reslice.SetInterpolationModeToNearestNeighbor()
            reslice.SetBackgroundLevel(0)
            reslice.AutoCropOutputOn()

            lut = self._create_label_lookup_table(image_data, opacity=1.0)
            image_to_colors = vtk.vtkImageMapToColors()
            image_to_colors.SetInputConnection(reslice.GetOutputPort())
            image_to_colors.SetLookupTable(lut)
            image_to_colors.SetOutputFormatToRGBA()

            actor = vtk.vtkImageActor()
            actor.GetMapper().SetInputConnection(image_to_colors.GetOutputPort())
            actor.SetVisibility(False)

            mpr_actors[orientation] = {
                "reslice": reslice,
                "actor": actor,
                "image_to_colors": image_to_colors,
                "lut": lut,
            }

        self._mpr_actors[frame] = mpr_actors
        self._setup_center_slices(image_data, frame)
        return mpr_actors

    def _create_label_lookup_table(self, image_data, opacity: float = 1.0):
        """Create lookup table for label-to-color mapping."""
        scalar_range = image_data.GetPointData().GetScalars().GetRange()
        min_label = int(scalar_range[0])
        max_label = int(scalar_range[1])

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(max_label + 1)
        lut.SetRange(min_label, max_label)

        lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)

        for label in range(min_label + 1, max_label + 1):
            if label in self.label_properties:
                r = self.label_properties[label].get("r", 1.0)
                g = self.label_properties[label].get("g", 1.0)
                b = self.label_properties[label].get("b", 1.0)
            else:
                r, g, b = label_color(label)
            lut.SetTableValue(label, r, g, b, opacity)

        lut.Build()
        return lut

    def _setup_center_slices(self, image_data, frame: int):
        """Set up reslice matrices to show center slices."""
        from .orientation import create_vtk_reslice_matrix

        center = image_data.GetCenter()
        actors = self._mpr_actors[frame]
        transforms = self._get_mpr_coordinate_systems()
        origin = [center[0], center[1], center[2]]

        for orientation in ["axial", "sagittal", "coronal"]:
            mat = create_vtk_reslice_matrix(transforms[orientation], origin)
            actors[orientation]["reslice"].SetResliceAxes(mat)
            actors[orientation]["reslice"].Update()  # Force VTK pipeline update

    def _get_mpr_coordinate_systems(self):
        """Get coordinate system transformation matrices for MPR views."""
        from .orientation import axcode_transform_matrix

        view_axcodes = {
            "axial": "LAS",
            "sagittal": "ASL",
            "coronal": "LSA",
        }

        transforms = {}
        for view, target_axcode in view_axcodes.items():
            transforms[view] = axcode_transform_matrix("LPS", target_axcode)

        return transforms

    def get_mpr_actors_for_frame(self, frame: int) -> dict:
        """Get MPR actors for a specific frame."""
        if frame not in self._mpr_actors:
            return self.create_mpr_actors(frame)
        return self._mpr_actors[frame]

    def update_slice_positions(
        self,
        frame: int,
        origin: list,
        rotation_sequence=None,
        rotation_angles=None,
        angle_units=None,
    ):
        """Update slice positions for MPR views with optional rotation."""
        from .orientation import (
            AngleUnits,
            EulerAxis,
            create_vtk_reslice_matrix,
            euler_angle_to_rotation_matrix,
        )

        if angle_units is None:
            angle_units = AngleUnits.DEGREES
        if frame not in self._mpr_actors:
            return

        actors = self._mpr_actors[frame]
        transforms = self._get_mpr_coordinate_systems()

        cumulative_rotation = np.eye(3)
        if rotation_sequence:
            from .orientation import quaternion_to_rotation_matrix

            for i, rotation in enumerate(rotation_sequence):
                if rotation.get("quaternion") is not None:
                    rotation_matrix = quaternion_to_rotation_matrix(
                        rotation["quaternion"]
                    )
                else:
                    angle = rotation_angles.get(i, 0) if rotation_angles else 0
                    rotation_matrix = euler_angle_to_rotation_matrix(
                        EulerAxis(rotation["axis"]), angle, angle_units
                    )
                cumulative_rotation = cumulative_rotation @ rotation_matrix

        axial_transform = cumulative_rotation @ transforms["axial"]
        sagittal_transform = cumulative_rotation @ transforms["sagittal"]
        coronal_transform = cumulative_rotation @ transforms["coronal"]

        axial_matrix = create_vtk_reslice_matrix(axial_transform, origin)
        actors["axial"]["reslice"].SetResliceAxes(axial_matrix)
        actors["axial"]["reslice"].Update()  # Force VTK pipeline update

        sagittal_matrix = create_vtk_reslice_matrix(sagittal_transform, origin)
        actors["sagittal"]["reslice"].SetResliceAxes(sagittal_matrix)
        actors["sagittal"]["reslice"].Update()  # Force VTK pipeline update

        coronal_matrix = create_vtk_reslice_matrix(coronal_transform, origin)
        actors["coronal"]["reslice"].SetResliceAxes(coronal_matrix)
        actors["coronal"]["reslice"].Update()  # Force VTK pipeline update

    def get_labels(self, frame: int = 0) -> list[int]:
        if frame >= len(self._label_images):
            frame = 0
        image_data = self._label_images[frame]
        max_label = int(image_data.GetPointData().GetScalars().GetRange()[1])
        if max_label < 1:
            return []
        acc = vtk.vtkImageAccumulate()
        acc.SetInputData(image_data)
        acc.SetComponentExtent(0, max_label, 0, 0, 0, 0)
        acc.SetComponentOrigin(0, 0, 0)
        acc.SetComponentSpacing(1, 0, 0)
        acc.Update()
        hist = acc.GetOutput().GetPointData().GetScalars()
        return [i for i in range(1, max_label + 1) if hist.GetTuple1(i) > 0]

    def _frame_mesh(self, frame: int) -> vtk.vtkPolyData | None:
        """Mesh shown at ``frame``, wrapping as the renderer does for short series."""
        if not self._meshes:
            return None
        return self._meshes[frame % len(self._meshes)]

    def label_centroid(self, labels: list[int], frame: int = 0) -> list[float] | None:
        mesh = self._frame_mesh(frame)
        if mesh is None or not labels:
            return None
        scalars = mesh.GetCellData().GetArray("Labels")
        if not scalars:
            return None
        label_set = set(labels)
        mask = [
            int(scalars.GetTuple1(i)) in label_set
            for i in range(scalars.GetNumberOfTuples())
        ]
        return masked_centroid(mesh, mask)

    def interface_centroid(
        self, labels_a: list[int], labels_b: list[int], frame: int = 0
    ) -> list[float] | None:
        mesh = self._frame_mesh(frame)
        if mesh is None or not labels_a or not labels_b:
            return None
        boundary = mesh.GetCellData().GetArray("BoundaryLabels")
        if not boundary:
            return None
        set_a, set_b = set(labels_a), set(labels_b)
        mask = []
        for i in range(boundary.GetNumberOfTuples()):
            l0 = int(boundary.GetComponent(i, 0))
            l1 = int(boundary.GetComponent(i, 1))
            mask.append((l0 in set_a and l1 in set_b) or (l0 in set_b and l1 in set_a))
        return masked_centroid(mesh, mask)

    def update_mpr_opacity(self, frame: int, opacity: float):
        """Update opacity for all MPR overlay labels."""
        if frame not in self._mpr_actors:
            return

        actors = self._mpr_actors[frame]

        for orientation in ["axial", "sagittal", "coronal"]:
            lut = actors[orientation]["lut"]
            for i in range(1, lut.GetNumberOfTableValues()):
                rgba = list(lut.GetTableValue(i))
                rgba[3] = opacity
                lut.SetTableValue(i, *rgba)
            actors[orientation]["image_to_colors"].Modified()
