"""Test ColorTransferFunctionPoint."""

import pathlib as pl

import pytest as pt
import tomlkit as tk

from cardio.transfer_functions import ColorTransferFunctionPoint


def test_color_point_from_toml():
    """Test loading ColorTransferFunctionPoint from TOML."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "color_point.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)

    # Create point from TOML data
    point = ColorTransferFunctionPoint.model_validate(data)

    # Verify values
    assert point.x == 200.0
    assert point.color == (0.8, 0.4, 0.2)


def test_color_point_validation():
    """Test ColorTransferFunctionPoint validation."""
    # Valid point
    point = ColorTransferFunctionPoint(x=100.0, color=(1.0, 0.5, 0.0))
    assert point.x == 100.0
    assert point.color == (1.0, 0.5, 0.0)

    # Invalid color value (too high)
    with pt.raises(ValueError):
        ColorTransferFunctionPoint(x=0.0, color=(1.5, 0.0, 0.0))

    # Invalid color value (negative)
    with pt.raises(ValueError):
        ColorTransferFunctionPoint(x=0.0, color=(0.0, -0.1, 0.0))

    # Invalid color value (too high)
    with pt.raises(ValueError):
        ColorTransferFunctionPoint(x=0.0, color=(0.0, 0.0, 2.0))
