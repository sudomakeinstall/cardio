"""Transfer function preset management for volume rendering."""

import pathlib as pl
import typing as ty
import tomlkit as tk
import pydantic as pc
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction, vtkVolumeProperty


def blend_transfer_functions(tfs, scalar_range=(-2000, 2000), num_samples=512):
    """
    Blend multiple transfer functions using volume rendering emission-absorption model.

    Based on the volume rendering equation from:
    - Levoy, M. "Display of Surfaces from Volume Data" IEEE Computer Graphics and Applications, 1988
    - Kajiya, J.T. & Von Herzen, B.P. "Ray tracing volume densities" ACM SIGGRAPH Computer Graphics, 1984
    - Engel, K. et al. "Real-time Volume Graphics" A K Peters, 2006, Chapter 2

    The volume rendering integral: I = ∫ C(s) * μ(s) * T(s) ds
    where C(s) = emission color, μ(s) = opacity, T(s) = transmission

    For discrete transfer functions, this becomes:
    - Total emission = Σ(color_i * opacity_i)
    - Total absorption = Σ(opacity_i)
    - Final color = total_emission / total_absorption
    """
    if len(tfs) == 1:
        return tfs[0]

    sample_points = np.linspace(
        start=scalar_range[0],
        stop=scalar_range[1],
        num=num_samples,
    )

    # Initialize arrays to store blended values
    blended_opacity = []
    blended_color = []

    for scalar_val in sample_points:
        # Accumulate emission and absorption for volume rendering
        total_emission = [0.0, 0.0, 0.0]
        total_absorption = 0.0

        for otf, ctf in tfs:
            # Get opacity and color for this scalar value
            layer_opacity = otf.GetValue(scalar_val)
            layer_color = [0.0, 0.0, 0.0]
            ctf.GetColor(scalar_val, layer_color)

            # Volume rendering accumulation:
            # Emission = color * opacity (additive)
            # Absorption = opacity (multiplicative through transmission)
            for i in range(3):
                total_emission[i] += layer_color[i] * layer_opacity

            total_absorption += layer_opacity

        # Clamp values to reasonable ranges
        total_absorption = min(total_absorption, 1.0)
        for i in range(3):
            total_emission[i] = min(total_emission[i], 1.0)

        # For the final color, normalize emission by absorption if absorption > 0
        if total_absorption > 0.001:  # Avoid division by zero
            final_color = [total_emission[i] / total_absorption for i in range(3)]
        else:
            final_color = [0.0, 0.0, 0.0]

        # Clamp final colors
        final_color = [min(c, 1.0) for c in final_color]

        blended_opacity.append(total_absorption)
        blended_color.append(final_color)

    # Create new VTK transfer functions with blended values
    blended_otf = vtkPiecewiseFunction()
    blended_ctf = vtkColorTransferFunction()

    for i, scalar_val in enumerate(sample_points):
        blended_otf.AddPoint(scalar_val, blended_opacity[i])
        blended_ctf.AddRGBPoint(
            scalar_val, blended_color[i][0], blended_color[i][1], blended_color[i][2]
        )

    return blended_otf, blended_ctf


class TransferFunctionConfig(pc.BaseModel):
    """Configuration for a single transfer function."""

    window: float = pc.Field(gt=0, description="Window width for transfer function")
    level: float = pc.Field(description="Window level for transfer function")
    locolor: list[float] = pc.Field(
        min_length=3, max_length=3, description="RGB color for low values"
    )
    hicolor: list[float] = pc.Field(
        min_length=3, max_length=3, description="RGB color for high values"
    )
    opacity: float = pc.Field(ge=0, le=1, description="Maximum opacity value")
    shape: str = pc.Field(description="Transfer function shape")

    @pc.field_validator("locolor", "hicolor")
    @classmethod
    def validate_rgb_values(cls, v: list[float]) -> list[float]:
        """Validate RGB values are in [0, 1] range."""
        for val in v:
            if not (0 <= val <= 1):
                raise ValueError(f"RGB values must be in range [0, 1], got {val}")
        return v

    @pc.field_validator("shape")
    @classmethod
    def validate_shape(cls, v: str) -> str:
        """Validate transfer function shape."""
        valid_shapes = {"rightskew", "leftskew", "linear", "sigmoid"}
        if v not in valid_shapes:
            raise ValueError(f"Shape must be one of {valid_shapes}, got {v}")
        return v

    @property
    def vtk_functions(self) -> tuple[vtkPiecewiseFunction, vtkColorTransferFunction]:
        """Create VTK transfer functions from this configuration."""
        # Create opacity transfer function
        otf = vtkPiecewiseFunction()
        otf.AddPoint(self.level - self.window * 0.50, 0.0)
        otf.AddPoint(self.level + self.window * 0.14, self.opacity)
        otf.AddPoint(self.level + self.window * 0.50, 0.0)

        # Create color transfer function
        ctf = vtkColorTransferFunction()
        ctf.AddRGBPoint(
            self.level - self.window / 2,
            self.locolor[0],
            self.locolor[1],
            self.locolor[2],
        )
        ctf.AddRGBPoint(
            self.level + self.window / 2,
            self.hicolor[0],
            self.hicolor[1],
            self.hicolor[2],
        )

        return otf, ctf


class VolumePropertyConfig(pc.BaseModel):
    """Configuration for volume rendering properties and transfer functions."""

    name: str = pc.Field(description="Display name of the preset")
    description: str = pc.Field(description="Description of the preset")

    # Lighting parameters
    ambient: float = pc.Field(ge=0, le=1, description="Ambient lighting coefficient")
    diffuse: float = pc.Field(ge=0, le=1, description="Diffuse lighting coefficient")
    specular: float = pc.Field(ge=0, le=1, description="Specular lighting coefficient")

    # Transfer functions
    transfer_functions: list[TransferFunctionConfig] = pc.Field(
        min_length=1, description="List of transfer functions to blend"
    )

    @property
    def vtk_property(self) -> vtkVolumeProperty:
        """Create a fully configured VTK volume property from this configuration."""
        # Get VTK transfer functions from each config
        tfs = [tf.vtk_functions for tf in self.transfer_functions]

        # Blend all transfer functions into a single composite
        blended_otf, blended_ctf = blend_transfer_functions(tfs)

        # Create and configure the volume property
        property = vtkVolumeProperty()
        property.SetScalarOpacity(blended_otf)
        property.SetColor(blended_ctf)
        property.ShadeOn()
        property.SetInterpolationTypeToLinear()
        property.SetAmbient(self.ambient)
        property.SetDiffuse(self.diffuse)
        property.SetSpecular(self.specular)

        return property


def load_preset(preset_name: str) -> VolumePropertyConfig:
    """Load a specific preset from its individual file."""
    assets_dir = pl.Path(__file__).parent / "assets"
    preset_file = assets_dir / f"{preset_name}.toml"

    if not preset_file.exists():
        available = list(list_available_presets().keys())
        raise KeyError(
            f"Transfer function preset '{preset_name}' not found. "
            f"Available presets: {available}"
        )

    with preset_file.open("rt", encoding="utf-8") as fp:
        raw_data = tk.load(fp)

    try:
        return VolumePropertyConfig.model_validate(raw_data)
    except pc.ValidationError as e:
        raise ValueError(f"Invalid preset file '{preset_name}.toml': {e}") from e




def list_available_presets() -> dict[str, str]:
    """
    List all available transfer function presets.

    Returns:
        Dictionary mapping preset names to descriptions
    """
    assets_dir = pl.Path(__file__).parent / "assets"
    preset_files = assets_dir.glob("*.toml")

    presets = {}
    for preset_file in preset_files:
        preset_name = preset_file.stem
        try:
            with preset_file.open("rt", encoding="utf-8") as fp:
                preset_data = tk.load(fp)
                presets[preset_name] = preset_data["description"]
        except (KeyError, OSError):
            # Skip files that don't have the expected structure
            continue

    return presets
