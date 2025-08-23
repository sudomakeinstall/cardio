import logging

import itk
import numpy as np
import pydantic as pc
import vtk

from .object import Object
from .utils import InterpolatorType, reset_direction
from .volume_property_presets import load_volume_property_preset


class Volume(Object):
    """Volume object with transfer functions and clipping support."""

    pattern: str = pc.Field(
        default="${frame}.nii.gz",
        description="Filename pattern with $frame placeholder",
    )
    transfer_function_preset: str = pc.Field(
        default="bone", description="Transfer function preset key"
    )
    _actors: list[vtk.vtkVolume] = pc.PrivateAttr(default_factory=list)

    @pc.model_validator(mode="after")
    def initialize_volume(self):
        """Generate VTK volume actors for all frames."""
        for frame, path in enumerate(self.path_list):
            logging.info(f"{self.label}: Loading frame {frame}.")

            image = itk.imread(path)
            image = reset_direction(image, InterpolatorType.LINEAR)
            image = itk.vtk_image_from_image(image)

            mapper = vtk.vtkGPUVolumeRayCastMapper()
            mapper.SetInputData(image)

            actor = vtk.vtkVolume()
            actor.SetMapper(mapper)

            self._actors.append(actor)

        return self

    @property
    def actors(self) -> list[vtk.vtkVolume]:
        return self._actors

    @property
    def preset(self):
        """Load preset based on transfer_function_preset."""
        return load_volume_property_preset(self.transfer_function_preset)

    def configure_actors(self):
        """Configure volume properties without adding to renderer."""
        for volume in self._actors:
            volume.SetVisibility(False)
            volume.SetProperty(self.preset.vtk_property)

    def apply_preset_to_actors(self):
        """Apply the current preset to all actors."""
        for actor in self._actors:
            actor.SetProperty(self.preset.vtk_property)
