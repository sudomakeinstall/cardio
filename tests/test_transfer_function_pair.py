"""Test TransferFunctionPairConfig."""

import pathlib as pl

import tomlkit as tk
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

from cardio.transfer_function_pair import TransferFunctionPairConfig


def test_pair_config_from_toml():
    """Test loading TransferFunctionPairConfig from TOML."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "transfer_function_pair.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)

    # Create pair config from TOML data
    pair = TransferFunctionPairConfig.model_validate(data)

    # Verify structure
    assert len(pair.opacity.points) == 2
    assert len(pair.color.points) == 2
    assert pair.opacity.points[0].x == 0.0
    assert pair.opacity.points[0].y == 0.0
    assert pair.color.points[0].color == (1.0, 0.0, 0.0)


def test_pair_config_vtk_functions():
    """Test VTK function creation from TransferFunctionPairConfig."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "transfer_function_pair.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)

    pair = TransferFunctionPairConfig.model_validate(data)
    otf, ctf = pair.vtk_functions

    # Verify types
    assert isinstance(otf, vtkPiecewiseFunction)
    assert isinstance(ctf, vtkColorTransferFunction)

    # Verify they have the expected number of points
    assert otf.GetSize() == 2
    assert ctf.GetSize() == 2


def test_pair_config_validation():
    """Test TransferFunctionPairConfig validation."""
    # Valid config
    data = {
        "opacity": {"points": [{"x": 0.0, "y": 0.0}]},
        "color": {"points": [{"x": 0.0, "color": [1.0, 0.0, 0.0]}]},
    }
    pair = TransferFunctionPairConfig.model_validate(data)
    assert len(pair.opacity.points) == 1
    assert len(pair.color.points) == 1
