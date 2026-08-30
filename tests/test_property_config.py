"""Test vtkPropertyConfig."""

# Third Party
import pytest
import vtk

# Internal
from cardio.property_config import Interpolation, Representation, vtkPropertyConfig


def test_property_config_from_toml(asset):
    config = vtkPropertyConfig.model_validate(asset("property_config.toml"))

    assert config.representation == Representation.Wireframe
    assert config.color == (0.8, 0.4, 0.2)
    assert config.edge_visibility
    assert not config.vertex_visibility
    assert not config.shading
    assert config.interpolation == Interpolation.Phong
    assert config.opacity == 0.7


def test_the_enums_number_their_members_the_way_vtk_does():
    """The members reach VTK as plain ints, so the numbering is the contract."""
    assert Representation.Wireframe == 1
    assert Interpolation.Phong == 2


def test_vtk_property_creation():
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

    assert isinstance(vtk_prop, vtk.vtkProperty)
    assert vtk_prop.GetRepresentation() == Representation.Wireframe
    assert vtk_prop.GetColor() == pytest.approx((0.8, 0.4, 0.2))
    assert vtk_prop.GetEdgeVisibility()
    assert not vtk_prop.GetVertexVisibility()
    assert not vtk_prop.GetShading()
    assert vtk_prop.GetInterpolation() == Interpolation.Phong
    assert vtk_prop.GetOpacity() == pytest.approx(0.7)
