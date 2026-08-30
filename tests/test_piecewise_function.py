"""Test PiecewiseFunctionPoint and PiecewiseFunctionConfig."""

import pytest as pt
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction

from cardio.piecewise_function import PiecewiseFunctionConfig, PiecewiseFunctionPoint


def test_piecewise_point_from_toml(asset):
    """Test loading PiecewiseFunctionPoint from TOML."""
    # Create point from TOML data
    point = PiecewiseFunctionPoint.model_validate(asset("piecewise_point.toml"))

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


def test_piecewise_config_from_toml(asset):
    """Test loading PiecewiseFunctionConfig from TOML."""
    # Create config from TOML data
    config = PiecewiseFunctionConfig.model_validate(asset("piecewise_config.toml"))

    # Verify points
    assert len(config.points) == 3
    assert config.points[0].x == -1000.0
    assert config.points[0].y == 0.0
    assert config.points[1].x == 0.0
    assert config.points[1].y == 0.8
    assert config.points[2].x == 1000.0
    assert config.points[2].y == 0.0


def test_piecewise_config_vtk_function(asset):
    """Test VTK function creation from PiecewiseFunctionConfig."""
    config = PiecewiseFunctionConfig.model_validate(asset("piecewise_config.toml"))
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
    data = {"points": [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 1.0}]}
    config = PiecewiseFunctionConfig.model_validate(data)
    assert len(config.points) == 2

    # Invalid config - no points
    with pt.raises(ValueError):
        PiecewiseFunctionConfig.model_validate({"points": []})
