"""Test ColorTransferFunctionPoint and ColorTransferFunctionConfig."""

import pytest as pt
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

from cardio.color_transfer_function import (
    ColorTransferFunctionConfig,
    ColorTransferFunctionPoint,
)


def test_color_point_from_toml(asset):
    """Test loading ColorTransferFunctionPoint from TOML."""
    # Create point from TOML data
    point = ColorTransferFunctionPoint.model_validate(asset("color_point.toml"))

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


def test_color_config_from_toml(asset):
    """Test loading ColorTransferFunctionConfig from TOML."""
    # Create config from TOML data
    config = ColorTransferFunctionConfig.model_validate(asset("color_config.toml"))

    # Verify points
    assert len(config.points) == 3
    assert config.points[0].x == -1000.0
    assert config.points[0].color == (0.0, 0.0, 1.0)
    assert config.points[1].x == 0.0
    assert config.points[1].color == (1.0, 0.5, 0.0)
    assert config.points[2].x == 1000.0
    assert config.points[2].color == (1.0, 1.0, 1.0)


def test_color_config_vtk_function(asset):
    """Test VTK function creation from ColorTransferFunctionConfig."""
    config = ColorTransferFunctionConfig.model_validate(asset("color_config.toml"))
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
