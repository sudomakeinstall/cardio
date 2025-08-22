"""Test vtkPropertyConfig."""

# System
import pathlib as pl

# Third Party
import tomlkit as tk
import pytest as pt
import vtk

# Internal
from cardio.property_config import vtkPropertyConfig, Representation, Interpolation


def test_property_config_from_toml():
    """Test loading vtkPropertyConfig from TOML."""
    # Load test data
    toml_path = pl.Path(__file__).parent / "assets" / "property_config.toml"
    with toml_path.open("rt", encoding="utf-8") as fp:
        data = tk.load(fp)

    # Create config from TOML data
    config = vtkPropertyConfig.model_validate(data)

    # Verify representation was parsed correctly
    assert config.representation == Representation.Wireframe
    assert config.representation == 1

    # Verify color values were parsed correctly
    assert config.color == (0.8, 0.4, 0.2)

    # Verify visibility values were parsed correctly
    assert config.edge_visibility == True
    assert config.vertex_visibility == False

    # Verify shading value was parsed correctly
    assert config.shading == False

    # Verify interpolation value was parsed correctly
    assert config.interpolation == Interpolation.Phong
    assert config.interpolation == 2

    # Verify opacity value was parsed correctly
    assert config.opacity == 0.7


def test_vtk_property_creation():
    """Test VTK property creation from vtkPropertyConfig."""
    # Create config directly to test VTK property generation
    config = vtkPropertyConfig(
        representation=Representation.Wireframe,
        color=(0.8, 0.4, 0.2),
        edge_visibility=True,
        vertex_visibility=False,
        shading=False,
        interpolation=Interpolation.Phong,
        opacity=0.7,
    )

    vtk_prop = config.vtk_property

    # Verify it's the right type
    assert isinstance(vtk_prop, vtk.vtkProperty)

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

    # Verify opacity setting was set correctly
    assert vtk_prop.GetOpacity() == pt.approx(0.7)
