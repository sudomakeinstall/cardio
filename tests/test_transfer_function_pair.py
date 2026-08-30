"""Test TransferFunctionPairConfig."""

# Third Party
import vtk

# Internal
from cardio.transfer_function_pair import TransferFunctionPairConfig


def test_pair_config_from_toml(asset):
    pair = TransferFunctionPairConfig.model_validate(
        asset("transfer_function_pair.toml")
    )

    assert len(pair.opacity.points) == 2
    assert len(pair.color.points) == 2
    assert pair.opacity.points[0].x == 0.0
    assert pair.opacity.points[0].y == 0.0
    assert pair.color.points[0].color == (1.0, 0.0, 0.0)


def test_pair_config_vtk_functions(asset):
    pair = TransferFunctionPairConfig.model_validate(
        asset("transfer_function_pair.toml")
    )

    otf, ctf = pair.vtk_functions

    assert isinstance(otf, vtk.vtkPiecewiseFunction)
    assert isinstance(ctf, vtk.vtkColorTransferFunction)
    assert otf.GetSize() == 2
    assert ctf.GetSize() == 2


def test_pair_config_validation():
    pair = TransferFunctionPairConfig.model_validate(
        {
            "opacity": {"points": [{"x": 0.0, "y": 0.0}]},
            "color": {"points": [{"x": 0.0, "color": [1.0, 0.0, 0.0]}]},
        }
    )

    assert len(pair.opacity.points) == 1
    assert len(pair.color.points) == 1
