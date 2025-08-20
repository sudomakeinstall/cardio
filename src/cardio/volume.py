import functools
import logging

import itk
import numpy as np
import pydantic as pc
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction, vtkPlanes
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper

from . import Object
from .transfer_functions import load_preset


def reset_direction(image):
    origin = np.asarray(itk.origin(image))
    spacing = np.asarray(itk.spacing(image))
    size = np.asarray(itk.size(image))
    direction = np.asarray(image.GetDirection())

    direction[direction == 1] = 0
    origin += np.dot(size, np.dot(np.diag(spacing), direction))
    direction = np.identity(3)

    origin = itk.Point[itk.F, 3](origin)
    spacing = itk.spacing(image)
    size = itk.size(image)
    direction = itk.matrix_from_array(direction)

    interpolator = itk.LinearInterpolateImageFunction.New(image)
    output = itk.resample_image_filter(
        image,
        size=size,
        interpolator=interpolator,
        output_spacing=spacing,
        output_origin=origin,
        output_direction=direction,
    )

    return output


class Volume(Object):
    """Volume object with transfer functions and clipping support."""

    preset_key: str = pc.Field(default=None, exclude=True)
    clipping_enabled: bool = pc.Field(default=False)

    def __init__(self, cfg: str, renderer: vtkRenderer):
        preset_key = cfg["transfer_function_preset"]

        super().__init__(
            label=cfg["label"],
            directory=cfg["directory"],
            pattern=cfg.get("pattern", "${frame}.nii.gz"),
            visible=cfg["visible"],
            renderer=renderer,
            clipping_enabled=cfg["clipping_enabled"],
            preset_key=preset_key,
        )

    @functools.cached_property
    def actors(self) -> list[vtkVolume]:
        """Generate VTK volume actors for all frames."""
        volume_actors = []

        for frame, path in enumerate(self.path_list):
            logging.info(f"{self.label}: Loading frame {frame}.")

            image = itk.imread(path)
            image = reset_direction(image)
            image = itk.vtk_image_from_image(image)

            mapper = vtkGPUVolumeRayCastMapper()
            mapper.SetInputData(image)

            actor = vtkVolume()
            actor.SetMapper(mapper)

            volume_actors.append(actor)

        return volume_actors

    @functools.cached_property
    def clipping_planes(self) -> vtkPlanes:
        """Generate clipping planes based on first actor bounds."""
        if not self.actors:
            return None

        bounds = self.actors[0].GetBounds()
        planes = vtkPlanes()
        self._create_clipping_planes_from_bounds(planes, bounds)
        return planes

    @property
    def preset(self):
        """Load preset based on current preset_key."""
        return load_preset(self.preset_key)

    def setup_pipeline(self, frame: int):
        for a in self.actors:
            self.renderer.AddVolume(a)
            a.SetVisibility(False)
            a.SetProperty(self.preset.vtk_property)
        if self.visible:
            self.actors[frame].SetVisibility(True)

        self.renderer.ResetCamera()

    def _create_clipping_planes_from_bounds(self, planes: vtkPlanes, bounds):
        """Create 6 clipping planes from box bounds."""
        import vtkmodules.vtkCommonCore as vtk

        # Create 6 planes for the box faces
        plane_list = []
        normals = [
            [1, 0, 0],
            [-1, 0, 0],  # x-min, x-max
            [0, 1, 0],
            [0, -1, 0],  # y-min, y-max
            [0, 0, 1],
            [0, 0, -1],  # z-min, z-max
        ]
        origins = [
            [bounds[0], 0, 0],
            [bounds[1], 0, 0],  # x-min, x-max
            [0, bounds[2], 0],
            [0, bounds[3], 0],  # y-min, y-max
            [0, 0, bounds[4]],
            [0, 0, bounds[5]],  # z-min, z-max
        ]

        points = vtk.vtkPoints()
        norms = vtk.vtkDoubleArray()
        norms.SetNumberOfComponents(3)
        norms.SetName("Normals")

        for i, (normal, origin) in enumerate(zip(normals, origins)):
            points.InsertNextPoint(origin)
            norms.InsertNextTuple(normal)

        planes.SetPoints(points)
        planes.SetNormals(norms)

    def toggle_clipping(self, enabled: bool):
        """Enable or disable volume clipping."""

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

    def reset_clipping_bounds(self):
        """Reset the clipping bounds to the volume bounds."""
        if not self.actors:
            return

        bounds = self.actors[0].GetBounds()
        self.update_clipping_bounds(bounds)

    def update_clipping_bounds(self, bounds):
        """Update clipping bounds from UI controls."""
        if not self.clipping_planes:
            return

        # Update clipping planes with new bounds
        self._create_clipping_planes_from_bounds(self.clipping_planes, bounds)

        # Apply to all volume actors
        for actor in self.actors:
            mapper = actor.GetMapper()
            mapper.SetClippingPlanes(self.clipping_planes)

    def apply_preset_to_actors(self):
        """Apply the current preset to all actors."""
        for actor in self.actors:
            actor.SetProperty(self.preset.vtk_property)
