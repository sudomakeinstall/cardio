"""Transfer function preset management for volume rendering."""

import pathlib as pl
import typing as ty
import tomlkit as tk
import pydantic as pc


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


def get_preset_transfer_functions(preset_name: str) -> list[dict[str, ty.Any]]:
    """
    Get transfer functions for a named preset.

    Args:
        preset_name: Name of the preset (e.g., 'cardiac', 'bone')

    Returns:
        List of transfer function dictionaries

    Raises:
        KeyError: If preset_name is not found
        ValueError: If preset file is invalid
    """
    preset = load_preset(preset_name)
    return [tf.model_dump() for tf in preset.transfer_functions]


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
