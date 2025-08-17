"""Test ColorTransferFunctionPoint."""

import pathlib as pl
import tomlkit as tk
import pytest as pt
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
    assert point.r == 0.8
    assert point.g == 0.4
    assert point.b == 0.2


def test_color_point_validation():
    """Test ColorTransferFunctionPoint validation."""
    # Valid point
    point = ColorTransferFunctionPoint(x=100.0, r=1.0, g=0.5, b=0.0)
    assert point.x == 100.0
    assert point.r == 1.0
    assert point.g == 0.5
    assert point.b == 0.0
    
    # Invalid r value (too high)
    with pt.raises(ValueError):
        ColorTransferFunctionPoint(x=0.0, r=1.5, g=0.0, b=0.0)
    
    # Invalid g value (negative)
    with pt.raises(ValueError):
        ColorTransferFunctionPoint(x=0.0, r=0.0, g=-0.1, b=0.0)
    
    # Invalid b value (too high)
    with pt.raises(ValueError):
        ColorTransferFunctionPoint(x=0.0, r=0.0, g=0.0, b=2.0)