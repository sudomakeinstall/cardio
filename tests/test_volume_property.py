"""Test VolumePropertyConfig."""

# Third Party
import pytest
import vtk

# Internal
from cardio.volume_property import VolumePropertyConfig

MINIMAL = {
    "name": "Test",
    "description": "Test config",
    "ambient": 0.5,
    "diffuse": 0.5,
    "specular": 0.0,
    "transfer_functions": [
        {
            "opacity": {"points": [{"x": 0.0, "y": 0.0}]},
            "color": {"points": [{"x": 0.0, "color": [1.0, 0.0, 0.0]}]},
        }
    ],
}


def test_volume_config_from_toml(asset):
    config = VolumePropertyConfig.model_validate(asset("volume_property.toml"))

    assert config.name == "Test Volume"
    assert config.description == "Test volume property configuration"
    assert config.ambient == 0.3
    assert config.diffuse == 0.7
    assert config.specular == 0.2

    assert len(config.transfer_functions) == 2
    for pair in config.transfer_functions:
        assert len(pair.opacity.points) == 3
        assert len(pair.color.points) == 3


def test_volume_config_vtk_property(asset):
    config = VolumePropertyConfig.model_validate(asset("volume_property.toml"))

    vtk_property = config.vtk_property

    assert isinstance(vtk_property, vtk.vtkVolumeProperty)
    assert vtk_property.GetAmbient() == 0.3
    assert vtk_property.GetDiffuse() == 0.7
    assert vtk_property.GetSpecular() == 0.2


def test_volume_config_validation():
    config = VolumePropertyConfig.model_validate(MINIMAL)

    assert config.name == "Test"
    assert len(config.transfer_functions) == 1


def test_a_volume_property_needs_at_least_one_transfer_function():
    with pytest.raises(ValueError):
        VolumePropertyConfig.model_validate(MINIMAL | {"transfer_functions": []})
