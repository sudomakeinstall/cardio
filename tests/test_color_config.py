"""Test ColorTransferFunctionConfig."""

import pathlib as pl
import tomlkit as tk
import pytest as pt
from cardio.transfer_functions import ColorTransferFunctionConfig
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction


def test_color_config_from_toml():
    """Test loading ColorTransferFunctionConfig from TOML."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "color_config.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)

    # Create config from TOML data
    config = ColorTransferFunctionConfig.model_validate(data)

    # Verify points
    assert len(config.points) == 3
    assert config.points[0].x == -1000.0
    assert config.points[0].r == 0.0
    assert config.points[0].g == 0.0
    assert config.points[0].b == 1.0
    assert config.points[1].x == 0.0
    assert config.points[1].r == 1.0
    assert config.points[1].g == 0.5
    assert config.points[1].b == 0.0
    assert config.points[2].x == 1000.0
    assert config.points[2].r == 1.0
    assert config.points[2].g == 1.0
    assert config.points[2].b == 1.0


def test_color_config_vtk_function():
    """Test VTK function creation from ColorTransferFunctionConfig."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "color_config.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)

    config = ColorTransferFunctionConfig.model_validate(data)
    vtk_func = config.vtk_function

    # Verify it's the right type
    assert isinstance(vtk_func, vtkColorTransferFunction)

    # Verify points were added
    assert vtk_func.GetSize() == 3

    # Test some color values
    color = [0.0, 0.0, 0.0]
    vtk_func.GetColor(-1000.0, color)
    assert color == [0.0, 0.0, 1.0]

    vtk_func.GetColor(0.0, color)
    assert color == [1.0, 0.5, 0.0]

    vtk_func.GetColor(1000.0, color)
    assert color == [1.0, 1.0, 1.0]


def test_color_config_validation():
    """Test ColorTransferFunctionConfig validation."""
    # Valid config
    data = {
        "points": [
            {"x": 0.0, "r": 1.0, "g": 0.0, "b": 0.0},
            {"x": 100.0, "r": 0.0, "g": 1.0, "b": 0.0},
        ]
    }
    config = ColorTransferFunctionConfig.model_validate(data)
    assert len(config.points) == 2

    # Invalid config - no points
    with pt.raises(ValueError):
        ColorTransferFunctionConfig.model_validate({"points": []})
