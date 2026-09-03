"""Test the nested configuration models, and their route in through Scene."""

# System
import logging

# Third Party
import pydantic as pc
import pytest

# Internal
from cardio.playback import Playback
from cardio.scene import Scene
from cardio.snap import Snap, SnapMode
from cardio.view import CameraLock, DrawerSection, Layout, Theme, View


def scene_from_toml(tmp_path, body: str, **overrides) -> Scene:
    """A Scene loaded from a TOML file, as the ``--config`` argument loads one."""
    path = tmp_path / "cfg.toml"
    path.write_text(body)
    Scene._config_file = path
    try:
        return Scene(**overrides)
    finally:
        del Scene._config_file


def test_defaults_are_an_empty_label_mode_selection():
    snap = Snap()
    assert snap.mode is SnapMode.LABEL
    assert snap.groups == ([], [], [])
    assert snap.traverse == 0
    assert not snap.locked
    assert not snap.orientation_locked


@pytest.mark.parametrize("traverse", [-1, 101])
def test_traverse_is_a_percentage(traverse):
    with pytest.raises(pc.ValidationError):
        Snap(traverse=traverse)


def test_unknown_mode_is_rejected():
    with pytest.raises(pc.ValidationError):
        Snap(mode="centroid")


@pytest.mark.parametrize(
    "mode,groups,chosen",
    [
        ("label", ([1], [], []), True),
        ("label", ([], [], []), False),
        ("interface", ([1], [2], []), True),
        ("interface", ([1], [], []), False),
        ("traverse", ([1], [2], [3]), True),
        ("traverse", ([1], [2], []), False),
    ],
)
def test_required_groups_chosen_follows_the_mode(mode, groups, chosen):
    labels_a, labels_b, labels_c = groups
    snap = Snap(mode=mode, labels_a=labels_a, labels_b=labels_b, labels_c=labels_c)
    assert snap.required_groups_chosen is chosen


def test_lock_without_the_groups_warns_rather_than_failing(caplog):
    with caplog.at_level(logging.WARNING):
        snap = Snap(mode="interface", labels_a=[1], locked=True)
    assert snap.locked
    assert "label groups are incomplete" in caplog.text


def test_orientation_lock_in_label_mode_warns(caplog):
    with caplog.at_level(logging.WARNING):
        Snap(mode="label", labels_a=[1], orientation_locked=True)
    assert "does not fit a plane" in caplog.text


def test_toml_snap_table_reaches_the_scene(tmp_path):
    scene = scene_from_toml(
        tmp_path,
        """
[snap]
mode = "traverse"
labels_a = [1]
labels_b = [2]
labels_c = [3]
traverse = 50
locked = true
orientation_locked = true
""",
    )
    assert scene.snap.mode is SnapMode.TRAVERSE
    assert scene.snap.groups == ([1], [2], [3])
    assert scene.snap.traverse == 50
    assert scene.snap.locked
    assert scene.snap.orientation_locked


def test_scene_defaults_to_an_empty_selection(tmp_path):
    assert scene_from_toml(tmp_path, "current_frame = 0").snap == Snap()


def test_unknown_snap_key_is_rejected(tmp_path):
    with pytest.raises(pc.ValidationError):
        scene_from_toml(tmp_path, "[snap]\nlabels_d = [4]\n")


def test_snap_segmentation_label_must_name_a_segmentation(tmp_path):
    with pytest.raises(
        pc.ValidationError, match="not found in available segmentations"
    ):
        scene_from_toml(tmp_path, '[snap]\nsegmentation_label = "absent"\n')


# --- the view -----------------------------------------------------------------


def test_view_defaults_to_the_quad_layout_in_dark():
    view = View()
    assert view.layout is Layout.QUAD
    assert view.theme is Theme.DARK


def test_the_quad_layout_is_the_empty_state_value():
    """``maximized_view`` says "nothing is maximized" with an empty string."""
    assert Layout.QUAD.state_value == ""
    assert Layout.TILE.state_value == "tile"


@pytest.mark.parametrize("layout", ["volume", "axial", "coronal", "sagittal", "tile"])
def test_every_maximizable_layout_keeps_its_name(layout):
    assert Layout(layout).state_value == layout


def test_unknown_layout_is_rejected():
    with pytest.raises(pc.ValidationError):
        View(layout="quadrant")


def test_view_defaults_open_the_playback_section_alone():
    assert View().open_sections == [DrawerSection.PLAYBACK.value]
    assert View().camera_lock is CameraLock.FREE
    assert View().help_visible is False


def test_open_sections_are_the_strings_the_accordion_tracks():
    view = View(drawer_sections=["tiles", "export"])
    assert view.open_sections == ["tiles", "export"]


@pytest.mark.parametrize(
    "field,value", [("camera_lock", "UR"), ("drawer_sections", ["shortcuts"])]
)
def test_unknown_view_values_are_rejected(field, value):
    with pytest.raises(pc.ValidationError):
        View(**{field: value})


def test_toml_view_table_reaches_the_scene(tmp_path):
    scene = scene_from_toml(
        tmp_path,
        """
[view]
layout = "tile"
theme = "light"
camera_lock = "LL"
drawer_sections = ["orientation", "tiles"]
help_visible = true
""",
    )
    assert scene.view.layout is Layout.TILE
    assert scene.view.theme is Theme.LIGHT
    assert scene.view.camera_lock is CameraLock.LL
    assert scene.view.open_sections == ["orientation", "tiles"]
    assert scene.view.help_visible is True


def test_unknown_view_key_is_rejected(tmp_path):
    with pytest.raises(pc.ValidationError):
        scene_from_toml(tmp_path, '[view]\nlayout_mode = "tile"\n')


# --- playback -----------------------------------------------------------------


def test_playback_defaults_match_the_sliders():
    playback = Playback()
    assert (playback.bpm, playback.bpr) == (60, 3)
    assert playback.incrementing and not playback.rotating


@pytest.mark.parametrize("field,value", [("bpm", 10), ("bpm", 200), ("bpr", 0)])
def test_playback_bounds_match_the_sliders(field, value):
    with pytest.raises(pc.ValidationError):
        Playback(**{field: value})


def test_toml_playback_table_reaches_the_scene(tmp_path):
    scene = scene_from_toml(
        tmp_path, "[playback]\nbpm = 75\nbpr = 7\nrotating = true\n"
    )
    assert scene.playback.bpm == 75
    assert scene.playback.bpr == 7
    assert scene.playback.rotating is True


def test_unknown_playback_key_is_rejected(tmp_path):
    with pytest.raises(pc.ValidationError):
        scene_from_toml(tmp_path, "[playback]\nbeats_per_minute = 75\n")
