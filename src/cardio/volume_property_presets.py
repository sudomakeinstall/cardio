# System
import pathlib as pl

# Third Party
import pydantic as pc
import tomlkit as tk

# Internal
from .volume_property import VolumePropertyConfig


def load_volume_property_preset(preset_name: str) -> VolumePropertyConfig:
    """Load a specific volume property preset from its individual file."""
    assets_dir = pl.Path(__file__).parent / "assets"
    preset_file = assets_dir / f"{preset_name}.toml"

    if not preset_file.exists():
        available = list(list_volume_property_presets().keys())
        raise KeyError(
            f"Volume property preset '{preset_name}' not found. "
            f"Available presets: {available}"
        )

    try:
        with preset_file.open("rt", encoding="utf-8") as fp:
            raw_data = tk.load(fp)
        return VolumePropertyConfig.model_validate(raw_data)
    except (pc.ValidationError, Exception) as e:
        raise ValueError(f"Invalid preset file '{preset_name}.toml': {e}") from e


def list_volume_property_presets() -> dict[str, str]:
    """
    List all available volume property presets.

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
