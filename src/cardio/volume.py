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

    def __init__(self, cfg: str, renderer: vtkRenderer):
        preset_key = cfg["transfer_function_preset"]

        super().__init__(
            label=cfg["label"],
            directory=cfg["directory"],
            pattern=cfg.get("pattern", "${frame}.nii.gz"),
            visible=cfg["visible"],
            renderer=renderer,
            clipping_enabled=cfg.get("clipping_enabled", False),
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

    def apply_preset_to_actors(self):
        """Apply the current preset to all actors."""
        for actor in self.actors:
            actor.SetProperty(self.preset.vtk_property)
