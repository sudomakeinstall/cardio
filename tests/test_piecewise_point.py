"""Test PiecewiseFunctionPoint."""

import pathlib as pl
import tomlkit as tk
import pytest as pt
from cardio.transfer_functions import PiecewiseFunctionPoint


def test_piecewise_point_from_toml():
    """Test loading PiecewiseFunctionPoint from TOML."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "piecewise_point.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)
    
    # Create point from TOML data
    point = PiecewiseFunctionPoint.model_validate(data)
    
    # Verify values
    assert point.x == 100.0
    assert point.y == 0.75


def test_piecewise_point_validation():
    """Test PiecewiseFunctionPoint validation."""
    # Valid point
    point = PiecewiseFunctionPoint(x=50.0, y=0.5)
    assert point.x == 50.0
    assert point.y == 0.5
    
    # Invalid y value (too high)
    with pt.raises(ValueError):
        PiecewiseFunctionPoint(x=0.0, y=1.5)
    
    # Invalid y value (negative)
    with pt.raises(ValueError):
        PiecewiseFunctionPoint(x=0.0, y=-0.1)