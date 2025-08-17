"""Test PiecewiseFunctionConfig."""

import pathlib as pl
import tomlkit as tk
import pytest as pt
from cardio.transfer_functions import PiecewiseFunctionConfig
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction


def test_piecewise_config_from_toml():
    """Test loading PiecewiseFunctionConfig from TOML."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "piecewise_config.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)
    
    # Create config from TOML data
    config = PiecewiseFunctionConfig.model_validate(data)
    
    # Verify points
    assert len(config.points) == 3
    assert config.points[0].x == -1000.0
    assert config.points[0].y == 0.0
    assert config.points[1].x == 0.0
    assert config.points[1].y == 0.8
    assert config.points[2].x == 1000.0
    assert config.points[2].y == 0.0


def test_piecewise_config_vtk_function():
    """Test VTK function creation from PiecewiseFunctionConfig."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "piecewise_config.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)
    
    config = PiecewiseFunctionConfig.model_validate(data)
    vtk_func = config.vtk_function
    
    # Verify it's the right type
    assert isinstance(vtk_func, vtkPiecewiseFunction)
    
    # Verify points were added
    assert vtk_func.GetSize() == 3
    
    # Test some values
    assert vtk_func.GetValue(-1000.0) == 0.0
    assert vtk_func.GetValue(0.0) == 0.8
    assert vtk_func.GetValue(1000.0) == 0.0


def test_piecewise_config_validation():
    """Test PiecewiseFunctionConfig validation."""
    # Valid config
    data = {
        "points": [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 1.0}
        ]
    }
    config = PiecewiseFunctionConfig.model_validate(data)
    assert len(config.points) == 2
    
    # Invalid config - no points
    with pt.raises(ValueError):
        PiecewiseFunctionConfig.model_validate({"points": []})