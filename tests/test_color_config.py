"""Test ColorTransferFunctionConfig."""

import pathlib as pl

import pytest as pt
import tomlkit as tk
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

from cardio.transfer_functions import ColorTransferFunctionConfig


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
    assert config.points[0].color == (0.0, 0.0, 1.0)
    assert config.points[1].x == 0.0
    assert config.points[1].color == (1.0, 0.5, 0.0)
    assert config.points[2].x == 1000.0
    assert config.points[2].color == (1.0, 1.0, 1.0)


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
            {"x": 0.0, "color": [1.0, 0.0, 0.0]},
            {"x": 100.0, "color": [0.0, 1.0, 0.0]},
        ]
    }
    config = ColorTransferFunctionConfig.model_validate(data)
    assert len(config.points) == 2

    # Invalid config - no points
    with pt.raises(ValueError):
        ColorTransferFunctionConfig.model_validate({"points": []})
