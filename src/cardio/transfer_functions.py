"""Transfer function preset management for volume rendering."""

import pathlib as pl
import typing as ty
import tomlkit as tk


def load_preset(preset_name: str) -> dict[str, ty.Any]:
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
        return tk.load(fp)


def get_preset_data(
    preset_name: str,
) -> tuple[list[dict[str, ty.Any]], dict[str, float]]:
    """
    Get transfer functions and lighting parameters for a named preset.

    Args:
        preset_name: Name of the preset (e.g., 'cardiac', 'bone')

    Returns:
        Tuple of (transfer_functions, lighting_params)
        - transfer_functions: List of transfer function dictionaries
        - lighting_params: Dict with 'ambient', 'diffuse', 'specular' keys

    Raises:
        KeyError: If preset_name is not found
    """
    preset = load_preset(preset_name)

    transfer_functions = preset["transfer_functions"]
    lighting_params = {
        "ambient": preset["ambient"],
        "diffuse": preset["diffuse"],
        "specular": preset["specular"],
    }

    return transfer_functions, lighting_params


def get_preset_transfer_functions(preset_name: str) -> list[dict[str, ty.Any]]:
    """
    Get transfer functions for a named preset.

    Args:
        preset_name: Name of the preset (e.g., 'cardiac', 'bone')

    Returns:
        List of transfer function dictionaries

    Raises:
        KeyError: If preset_name is not found
    """
    transfer_functions, _ = get_preset_data(preset_name)
    return transfer_functions


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
