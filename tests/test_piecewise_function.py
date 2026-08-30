"""Test PiecewiseFunctionPoint and PiecewiseFunctionConfig."""

# Third Party
import pytest
import vtk

# Internal
from cardio.piecewise_function import PiecewiseFunctionConfig, PiecewiseFunctionPoint


def test_piecewise_point_from_toml(asset):
    point = PiecewiseFunctionPoint.model_validate(asset("piecewise_point.toml"))

    assert point.x == 100.0
    assert point.y == 0.75


def test_a_piecewise_point_takes_a_unit_opacity():
    point = PiecewiseFunctionPoint(x=50.0, y=0.5)

    assert point.x == 50.0
    assert point.y == 0.5


@pytest.mark.parametrize("y", [1.5, -0.1])
def test_an_opacity_outside_the_unit_range_is_rejected(y):
    with pytest.raises(ValueError):
        PiecewiseFunctionPoint(x=0.0, y=y)


def test_piecewise_config_from_toml(asset):
    config = PiecewiseFunctionConfig.model_validate(asset("piecewise_config.toml"))

    assert [(point.x, point.y) for point in config.points] == [
        (-1000.0, 0.0),
        (0.0, 0.8),
        (1000.0, 0.0),
    ]


def test_piecewise_config_vtk_function(asset):
    config = PiecewiseFunctionConfig.model_validate(asset("piecewise_config.toml"))

    vtk_func = config.vtk_function

    assert isinstance(vtk_func, vtk.vtkPiecewiseFunction)
    assert vtk_func.GetSize() == 3
    assert vtk_func.GetValue(-1000.0) == 0.0
    assert vtk_func.GetValue(0.0) == 0.8
    assert vtk_func.GetValue(1000.0) == 0.0


def test_piecewise_config_validation():
    config = PiecewiseFunctionConfig.model_validate(
        {"points": [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 1.0}]}
    )

    assert len(config.points) == 2


def test_a_piecewise_function_needs_at_least_one_point():
    with pytest.raises(ValueError):
        PiecewiseFunctionConfig.model_validate({"points": []})
