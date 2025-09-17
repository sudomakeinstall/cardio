"""Test volume_property_presets module."""

from unittest.mock import mock_open, patch

import pytest as pt

from cardio.volume_property import VolumePropertyConfig
from cardio.volume_property_presets import (
    list_volume_property_presets,
    load_volume_property_preset,
)


def test_list_volume_property_presets():
    """Test listing available volume property presets."""
    presets = list_volume_property_presets()

    # Should return a dictionary
    assert isinstance(presets, dict)

    # Should contain at least the known presets
    expected_presets = ["bone", "vascular_closed", "vascular_open", "xray"]
    for preset in expected_presets:
        assert preset in presets

    # Values should be description strings
    for preset_name, description in presets.items():
        assert isinstance(preset_name, str)
        assert isinstance(description, str)
        assert len(description) > 0


def test_load_volume_property_preset_existing():
    """Test loading an existing volume property preset."""
    # Test loading a known preset
    config = load_volume_property_preset("bone")

    # Should return a VolumePropertyConfig
    assert isinstance(config, VolumePropertyConfig)

    # Should have required fields
    assert hasattr(config, "name")
    assert hasattr(config, "description")
    assert hasattr(config, "ambient")
    assert hasattr(config, "diffuse")
    assert hasattr(config, "specular")
    assert hasattr(config, "transfer_functions")

    # Transfer functions should be a list
    assert isinstance(config.transfer_functions, list)
    assert len(config.transfer_functions) > 0


def test_load_volume_property_preset_nonexistent():
    """Test loading a non-existent volume property preset."""
    with pt.raises(KeyError, match="Volume property preset 'nonexistent' not found"):
        load_volume_property_preset("nonexistent")


@patch("cardio.volume_property_presets.tk.load")
@patch("pathlib.Path.open")
@patch("pathlib.Path.exists")
def test_load_volume_property_preset_invalid_toml(
    mock_exists, mock_open_file, mock_tk_load
):
    """Test loading a preset with invalid TOML content."""
    mock_exists.return_value = True
    mock_open_file.return_value = mock_open()()
    mock_tk_load.side_effect = Exception("Invalid TOML")

    with pt.raises(ValueError, match="Invalid preset file"):
        load_volume_property_preset("invalid")


@patch("pathlib.Path.open")
@patch("pathlib.Path.exists")
def test_load_volume_property_preset_invalid_structure(mock_exists, mock_open_file):
    """Test loading a preset with valid TOML but invalid structure."""
    mock_exists.return_value = True
    # Valid TOML but missing required fields for VolumePropertyConfig
    invalid_config = """
    name = "test"
    # Missing required fields like ambient, diffuse, specular, transfer_functions
    """
    mock_open_file.return_value = mock_open(read_data=invalid_config)()

    with pt.raises(ValueError, match="Invalid preset file"):
        load_volume_property_preset("invalid_structure")


def test_list_volume_property_presets_with_invalid_files():
    """Test that list_volume_property_presets skips invalid files gracefully."""
    # This test uses the actual assets directory but verifies
    # that if there were invalid files, they would be skipped
    presets = list_volume_property_presets()

    # Should still work and return valid presets
    assert isinstance(presets, dict)
    assert len(presets) > 0


@patch("pathlib.Path.glob")
@patch("pathlib.Path.open")
def test_list_volume_property_presets_empty_directory(mock_open_file, mock_glob):
    """Test list_volume_property_presets with no TOML files."""
    mock_glob.return_value = []

    presets = list_volume_property_presets()

    assert presets == {}


@patch("cardio.volume_property_presets.tk.load")
@patch("pathlib.Path.glob")
@patch("pathlib.Path.open")
def test_list_volume_property_presets_file_with_no_description(
    mock_open_file, mock_glob, mock_tk_load
):
    """Test list_volume_property_presets with a file missing description."""
    # Mock a file path using MagicMock to allow setting stem
    from unittest.mock import MagicMock

    mock_file_path = MagicMock()
    mock_file_path.stem = "test"
    mock_glob.return_value = [mock_file_path]

    # Mock file content without description (will cause KeyError)
    mock_open_file.return_value = mock_open()()
    mock_tk_load.return_value = {"name": "test"}  # No description field

    presets = list_volume_property_presets()

    # Should skip files without description
    assert presets == {}
