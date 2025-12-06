import pydantic as pc
import vtk

from .blend_transfer_functions import blend_transfer_functions
from .transfer_function_pair import TransferFunctionPairConfig
from .types import ScalarComponent


class VolumePropertyConfig(pc.BaseModel):
    """Configuration for volume rendering properties and transfer functions."""

    name: str = pc.Field(description="Display name of the preset")
    description: str = pc.Field(description="Description of the preset")

    # Lighting parameters
    ambient: ScalarComponent
    diffuse: ScalarComponent
    specular: ScalarComponent

    # Transfer functions
    transfer_functions: list[TransferFunctionPairConfig] = pc.Field(
        min_length=1, description="List of transfer function pairs to blend"
    )

    @property
    def vtk_property(self) -> vtk.vtkVolumeProperty:
        """Create a fully configured VTK volume property from this configuration."""
        # Get VTK transfer functions from each pair config
        tfs = [pair.vtk_functions for pair in self.transfer_functions]

        # Blend all transfer functions into a single composite
        blended_otf, blended_ctf = blend_transfer_functions(tfs)

        # Create and configure the volume property
        _vtk_property = vtk.vtkVolumeProperty()
        _vtk_property.SetScalarOpacity(blended_otf)
        _vtk_property.SetColor(blended_ctf)
        _vtk_property.ShadeOn()
        _vtk_property.SetInterpolationTypeToLinear()
        _vtk_property.SetAmbient(self.ambient)
        _vtk_property.SetDiffuse(self.diffuse)
        _vtk_property.SetSpecular(self.specular)

        return _vtk_property
