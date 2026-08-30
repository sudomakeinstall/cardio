"""Test the volume property presets, and the loaders that find them.

Both loaders read a directory of TOML files, so the error paths are driven by
writing files into a directory the test owns and pointing the module at it.
Mocking pathlib out from under them tested the mocks instead, and left the
patches loose on Path itself for the length of the test.
"""

# System
import pathlib as pl

# Third Party
import pytest

# Internal
import cardio.volume_property_presets as vpp
from cardio.volume_property import VolumePropertyConfig

SHIPPED = ("bone", "vascular_closed", "vascular_open", "xray")

COMPLETE = """
name = "Test"
description = "A test preset"
ambient = 0.5
diffuse = 0.5
specular = 0.0

[[transfer_functions]]

[transfer_functions.opacity]
[[transfer_functions.opacity.points]]
x = 0.0
y = 0.0

[transfer_functions.color]
[[transfer_functions.color.points]]
x = 0.0
color = [1.0, 0.0, 0.0]
"""


@pytest.fixture
def assets(tmp_path, monkeypatch) -> pl.Path:
    """An assets directory the test owns, in place of the shipped one."""
    monkeypatch.setattr(vpp, "ASSETS_DIR", tmp_path)
    return tmp_path


# --- the presets that ship ---------------------------------------------------


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_preset_loads(name):
    config = vpp.load_volume_property_preset(name)

    assert isinstance(config, VolumePropertyConfig)
    assert config.transfer_functions


def test_the_shipped_presets_are_listed_with_their_descriptions():
    presets = vpp.list_volume_property_presets()

    assert set(presets) == set(SHIPPED)
    assert all(description for description in presets.values())


def test_an_absent_preset_names_the_ones_that_exist():
    with pytest.raises(
        KeyError, match="Volume property preset 'nonexistent' not found"
    ):
        vpp.load_volume_property_preset("nonexistent")


# --- what the loaders do with a directory of their own -----------------------


def test_a_written_preset_is_found_and_listed(assets):
    (assets / "written.toml").write_text(COMPLETE)

    assert vpp.list_volume_property_presets() == {"written": "A test preset"}
    assert vpp.load_volume_property_preset("written").name == "Test"


def test_a_file_that_is_not_toml_is_rejected(assets):
    (assets / "broken.toml").write_text("this is not = = toml")

    with pytest.raises(ValueError, match="Invalid preset file"):
        vpp.load_volume_property_preset("broken")


def test_a_file_missing_the_required_fields_is_rejected(assets):
    """Parses as TOML, but carries none of what the model needs."""
    (assets / "sparse.toml").write_text('name = "test"\n')

    with pytest.raises(ValueError, match="Invalid preset file"):
        vpp.load_volume_property_preset("sparse")


def test_an_empty_directory_lists_nothing(assets):
    assert vpp.list_volume_property_presets() == {}


def test_a_file_without_a_description_is_skipped(assets):
    """Listing is a menu, so an entry with no label is left off it."""
    (assets / "labelled.toml").write_text(COMPLETE)
    (assets / "unlabelled.toml").write_text('name = "test"\n')

    assert set(vpp.list_volume_property_presets()) == {"labelled"}


def test_a_file_that_is_not_toml_is_skipped_from_the_listing(assets):
    """One unreadable file must not take the whole menu down with it."""
    (assets / "good.toml").write_text(COMPLETE)
    (assets / "broken.toml").write_text("this is not = = toml")

    assert set(vpp.list_volume_property_presets()) == {"good"}
