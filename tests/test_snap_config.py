"""Test the snap selection as a configuration model, and its route in through Scene."""

# System
import logging

# Third Party
import pydantic as pc
import pytest

# Internal
from cardio.scene import Scene
from cardio.snap import Snap, SnapMode


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
