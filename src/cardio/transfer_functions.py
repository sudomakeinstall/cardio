"""Transfer function preset management for volume rendering."""

import os
import tomlkit as tk
from typing import Dict, List, Any


def load_transfer_function_presets() -> Dict[str, Any]:
    """Load transfer function presets from the assets folder."""
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    preset_file = os.path.join(assets_dir, "transfer_function_presets.toml")

    with open(preset_file, mode="rt", encoding="utf-8") as fp:
        return tk.load(fp)


def get_preset_transfer_functions(preset_name: str) -> List[Dict[str, Any]]:
    """
    Get transfer functions for a named preset.

    Args:
        preset_name: Name of the preset (e.g., 'cardiac', 'bone')

    Returns:
        List of transfer function dictionaries

    Raises:
        KeyError: If preset_name is not found
    """
    presets = load_transfer_function_presets()

    if preset_name not in presets:
        available = list(presets.keys())
        raise KeyError(
            f"Transfer function preset '{preset_name}' not found. "
            f"Available presets: {available}"
        )

    return presets[preset_name]["transfer_functions"]


def list_available_presets() -> Dict[str, str]:
    """
    List all available transfer function presets.

    Returns:
        Dictionary mapping preset names to descriptions
    """
    presets = load_transfer_function_presets()
    return {name: preset_data["description"] for name, preset_data in presets.items()}
