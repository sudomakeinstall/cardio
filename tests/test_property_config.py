"""Test PropertyConfig."""

import pathlib as pl
import tomlkit as tk
import pytest as pt
from cardio.property_config import PropertyConfig, Representation, Interpolation
from vtkmodules.vtkRenderingCore import vtkProperty


def test_property_config_from_toml():
    """Test loading PropertyConfig from TOML with integer representation."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "property_config.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)

    # Create config from TOML data
    config = PropertyConfig.model_validate(data)

    # Verify representation was parsed correctly
    assert config.representation == Representation.Wireframe
    assert config.representation == 1

    # Verify color values were parsed correctly
    assert config.r == 0.8
    assert config.g == 0.4
    assert config.b == 0.2

    # Verify visibility values were parsed correctly
    assert config.edge_visibility == True
    assert config.vertex_visibility == False

    # Verify shading value was parsed correctly
    assert config.shading == False

    # Verify interpolation value was parsed correctly
    assert config.interpolation == Interpolation.Phong
    assert config.interpolation == 2


def test_property_config_vtk_property():
    """Test VTK property creation from PropertyConfig."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "property_config.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)

    config = PropertyConfig.model_validate(data)
    vtk_prop = config.vtk_property

    # Verify it's the right type
    assert isinstance(vtk_prop, vtkProperty)

    # Verify representation was set correctly
    assert vtk_prop.GetRepresentation() == 1  # Wireframe

    # Verify color was set correctly
    color = vtk_prop.GetColor()
    assert color[0] == pt.approx(0.8)
    assert color[1] == pt.approx(0.4)
    assert color[2] == pt.approx(0.2)

    # Verify visibility settings were set correctly
    assert vtk_prop.GetEdgeVisibility() == True
    assert vtk_prop.GetVertexVisibility() == False

    # Verify shading setting was set correctly
    assert vtk_prop.GetShading() == False

    # Verify interpolation setting was set correctly
    assert vtk_prop.GetInterpolation() == 2  # Phong
