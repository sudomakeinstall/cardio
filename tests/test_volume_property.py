"""Test VolumePropertyConfig."""

import pytest as pt
from vtkmodules.vtkRenderingCore import vtkVolumeProperty

from cardio.volume_property import VolumePropertyConfig


def test_volume_config_from_toml(asset):
    """Test loading VolumePropertyConfig from TOML."""
    # Create config from TOML data
    config = VolumePropertyConfig.model_validate(asset("volume_property.toml"))

    # Verify basic properties
    assert config.name == "Test Volume"
    assert config.description == "Test volume property configuration"
    assert config.ambient == 0.3
    assert config.diffuse == 0.7
    assert config.specular == 0.2

    # Verify transfer functions
    assert len(config.transfer_functions) == 2

    # First pair
    pair1 = config.transfer_functions[0]
    assert len(pair1.opacity.points) == 3
    assert len(pair1.color.points) == 3

    # Second pair
    pair2 = config.transfer_functions[1]
    assert len(pair2.opacity.points) == 3
    assert len(pair2.color.points) == 3


def test_volume_config_vtk_property(asset):
    """Test VTK property creation from VolumePropertyConfig."""
    config = VolumePropertyConfig.model_validate(asset("volume_property.toml"))
    vtk_property = config.vtk_property

    # Verify it's the right type
    assert isinstance(vtk_property, vtkVolumeProperty)

    # Verify lighting parameters were set
    assert vtk_property.GetAmbient() == 0.3
    assert vtk_property.GetDiffuse() == 0.7
    assert vtk_property.GetSpecular() == 0.2


def test_volume_config_validation():
    """Test VolumePropertyConfig validation."""
    # Valid config
    data = {
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
    config = VolumePropertyConfig.model_validate(data)
    assert config.name == "Test"
    assert len(config.transfer_functions) == 1

    # Invalid config - no transfer functions
    invalid_data = {
        "name": "Test",
        "description": "Test config",
        "ambient": 0.5,
        "diffuse": 0.5,
        "specular": 0.0,
        "transfer_functions": [],
    }
    with pt.raises(ValueError):
        VolumePropertyConfig.model_validate(invalid_data)
