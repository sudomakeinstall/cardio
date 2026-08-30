"""Test ColorTransferFunctionPoint and ColorTransferFunctionConfig."""

# Third Party
import pytest
import vtk

# Internal
from cardio.color_transfer_function import (
    ColorTransferFunctionConfig,
    ColorTransferFunctionPoint,
)

CONFIG_POINTS = [
    (-1000.0, (0.0, 0.0, 1.0)),
    (0.0, (1.0, 0.5, 0.0)),
    (1000.0, (1.0, 1.0, 1.0)),
]


def test_color_point_from_toml(asset):
    point = ColorTransferFunctionPoint.model_validate(asset("color_point.toml"))

    assert point.x == 200.0
    assert point.color == (0.8, 0.4, 0.2)


def test_a_color_point_takes_a_unit_rgb_triple():
    point = ColorTransferFunctionPoint(x=100.0, color=(1.0, 0.5, 0.0))

    assert point.x == 100.0
    assert point.color == (1.0, 0.5, 0.0)


@pytest.mark.parametrize("color", [(1.5, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, 2.0)])
def test_a_channel_outside_the_unit_range_is_rejected(color):
    with pytest.raises(ValueError):
        ColorTransferFunctionPoint(x=0.0, color=color)


def test_color_config_from_toml(asset):
    config = ColorTransferFunctionConfig.model_validate(asset("color_config.toml"))

    assert [(point.x, point.color) for point in config.points] == CONFIG_POINTS


def test_color_config_vtk_function(asset):
    config = ColorTransferFunctionConfig.model_validate(asset("color_config.toml"))

    vtk_func = config.vtk_function

    assert isinstance(vtk_func, vtk.vtkColorTransferFunction)
    assert vtk_func.GetSize() == 3
    for x, expected in CONFIG_POINTS:
        color = [0.0, 0.0, 0.0]
        vtk_func.GetColor(x, color)
        assert color == list(expected)


def test_color_config_validation():
    config = ColorTransferFunctionConfig.model_validate(
        {
            "points": [
                {"x": 0.0, "color": [1.0, 0.0, 0.0]},
                {"x": 100.0, "color": [0.0, 1.0, 0.0]},
            ]
        }
    )

    assert len(config.points) == 2


def test_a_color_map_needs_at_least_one_point():
    with pytest.raises(ValueError):
        ColorTransferFunctionConfig.model_validate({"points": []})
