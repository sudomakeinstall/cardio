"""Test the nested configuration models, and their route in through Scene."""

# System
import logging
import pathlib as pl

# Third Party
import pydantic as pc
import pytest

# Internal
from cardio.playback import Playback
from cardio.rotation import RotationMetadata, RotationSequence, RotationStep
from cardio.scene import Scene
from cardio.snap import Snap, SnapMode
from cardio.view import CameraLock, DrawerSection, Layout, Theme, View
from cardio.window_level import presets


def scene_from_toml(tmp_path, body: str, **overrides) -> Scene:
    """A Scene loaded from a TOML file, as the ``--config`` argument loads one."""
    path = tmp_path / "cfg.toml"
    path.write_text(body)
    return Scene.load(config_file=path, **overrides)


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


@pytest.mark.parametrize("layout", ["volume", "ul", "ll", "lr", "tile"])
def test_every_maximizable_layout_keeps_its_name(layout):
    assert Layout(layout).state_value == layout


@pytest.mark.parametrize(
    "value,expected",
    [("", Layout.QUAD), (None, Layout.QUAD), ("tile", Layout.TILE)],
)
def test_the_state_value_reads_back_as_its_layout(value, expected):
    """Unset state reads as None, which is the quad view it will open in."""
    assert Layout.from_state(value) is expected


@pytest.mark.parametrize(
    "layout,shows",
    [
        (Layout.QUAD, True),
        (Layout.UL, True),
        (Layout.LL, True),
        (Layout.LR, True),
        (Layout.VOLUME, False),
        (Layout.TILE, False),
        (Layout.VOLUMETRY, False),
    ],
)
def test_only_the_layouts_with_slices_in_them_want_the_reslice(layout, shows):
    assert layout.shows_slices is shows


@pytest.mark.parametrize(
    "layout,drawn",
    [
        (Layout.QUAD, {Layout.UL, Layout.LL, Layout.LR, Layout.VOLUME}),
        (Layout.VOLUME, {Layout.VOLUME}),
        (Layout.UL, {Layout.UL}),
        (Layout.LL, {Layout.LL}),
        (Layout.LR, {Layout.LR}),
        (Layout.TILE, {Layout.TILE}),
        (Layout.VOLUMETRY, {Layout.VOLUMETRY}),
    ],
)
def test_a_layout_draws_only_what_it_shows(layout, drawn):
    assert layout.on_screen == drawn


def test_the_charts_resample_no_cut_and_so_draw_none():
    """What the volumetry layout shows is a measurement, not an image.

    Both halves matter and fail differently. Left in ``shows_slices``, every
    frame of a cine resamples three oblique planes nobody can see; left in
    ``shows_reslice``, the drawer offers the window, level and overlay controls
    over a chart none of them acts on.
    """
    assert not Layout.VOLUMETRY.shows_slices
    assert not Layout.VOLUMETRY.shows_reslice


def test_a_maximized_cut_is_alone_on_screen_though_all_three_are_resliced():
    """The distinction ``shows_slices`` cannot make, and a capture needs.

    A maximized upper-left view still resamples the other two cuts, so
    their windows are current -- but nobody can see them, and capturing them
    would be writing a view the user never chose.
    """
    assert Layout.UL.shows_slices
    assert Layout.UL.on_screen == {Layout.UL}


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
camera_lock = "ll"
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
    assert (playback.bpm, playback.bpr) == (20, 3)
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


# --- objects ------------------------------------------------------------------


def test_a_misspelled_background_key_is_rejected(tmp_path):
    with pytest.raises(pc.ValidationError):
        scene_from_toml(tmp_path, "[background]\nlite = [1.0, 1.0, 1.0]\n")


def test_a_misspelled_property_key_is_rejected(tmp_path):
    """One table deeper than the object itself, where the same typo hid."""
    with pytest.raises(pc.ValidationError):
        scene_from_toml(
            tmp_path,
            '[[meshes]]\nlabel = "obj"\ndirectory = "."\n'
            "\n[meshes.properties]\nedge_visibilty = true\n",
        )


@pytest.mark.parametrize("kind", ["meshes", "volumes", "segmentations"])
def test_a_misspelled_object_key_is_rejected(tmp_path, kind):
    """The trap this closes: ``[meshes.property]`` was ignored, not reported."""
    with pytest.raises(pc.ValidationError):
        scene_from_toml(
            tmp_path,
            f'[[{kind}]]\nlabel = "obj"\ndirectory = "."\nmpr_overlays = true\n',
        )


# --- the window and level -----------------------------------------------------
#
# A preset is a window and a level, so a config can name it or them, and used
# to be able to name both -- with the preset winning silently, whichever the
# author had actually meant.


def test_the_default_preset_supplies_the_default_window_and_level():
    scene = Scene()

    assert scene.mpr_window_level_preset == 7
    assert (scene.mpr_window, scene.mpr_level) == (
        presets[7].window,
        presets[7].level,
    )


def test_a_configured_preset_supplies_the_window_and_level(tmp_path):
    scene = scene_from_toml(tmp_path, "mpr_window_level_preset = 4\n")

    assert (scene.mpr_window, scene.mpr_level) == (
        presets[4].window,
        presets[4].level,
    )


def test_a_configured_window_and_level_are_kept(tmp_path):
    """They used to be overwritten by the preset nobody had asked for."""
    scene = scene_from_toml(tmp_path, "mpr_window = 1234.0\nmpr_level = 56.0\n")

    assert (scene.mpr_window, scene.mpr_level) == (1234.0, 56.0)


def test_a_configured_window_and_level_drop_the_default_preset(tmp_path):
    """The values no longer answer to a preset, so none is selected."""
    scene = scene_from_toml(tmp_path, "mpr_window = 1234.0\nmpr_level = 56.0\n")

    assert scene.mpr_window_level_preset is None


def test_a_window_and_level_that_match_the_preset_keep_it(tmp_path):
    scene = scene_from_toml(
        tmp_path,
        f"mpr_window = {presets[7].window}\nmpr_level = {presets[7].level}\n",
    )

    assert scene.mpr_window_level_preset == 7


def test_naming_both_and_disagreeing_is_refused(tmp_path):
    with pytest.raises(pc.ValidationError, match="one or the other"):
        scene_from_toml(tmp_path, "mpr_window = 1234.0\nmpr_window_level_preset = 4\n")


def test_only_one_of_the_pair_is_enough_to_drop_the_preset(tmp_path):
    """Half a pair still contradicts the preset that names both."""
    scene = scene_from_toml(tmp_path, "mpr_window = 1234.0\n")

    assert scene.mpr_window_level_preset is None
    assert scene.mpr_window == 1234.0


# ------------------------------------------------- where rotations come from ----


def rotation_file(tmp_path) -> pl.Path:
    """A rotation file holding one named step."""
    path = tmp_path / "rotations.toml"
    RotationSequence(
        metadata=RotationMetadata(),
        angles_list=[RotationStep(axis="Z", angle=0.5, name="from the file")],
    ).to_file(path)
    return path


def test_a_named_rotation_file_is_read_over_what_the_config_spelled(tmp_path):
    """A file is the answer once it is named, which is why saving one writes no
    sequence beside it."""
    scene = scene_from_toml(
        tmp_path,
        "\n".join(
            [
                f'mpr_rotation_file = "{rotation_file(tmp_path)}"',
                "[[mpr_rotation_sequence.angles_list]]",
                'axis = "X"',
                "angle = 1.25",
                'name = "from the config"',
            ]
        ),
    )

    assert [step.name for step in scene.mpr_rotation_sequence.angles_list] == [
        "from the file"
    ]


def test_a_rotation_file_that_is_not_there_is_refused(tmp_path):
    """It used to open on no rotations, which is a strange way to say "mistyped"."""
    missing = tmp_path / "nowhere.toml"

    with pytest.raises(pc.ValidationError, match="does not exist"):
        scene_from_toml(tmp_path, f'mpr_rotation_file = "{missing}"')


def test_the_refusal_names_the_path_that_was_asked_for(tmp_path):
    with pytest.raises(pc.ValidationError, match="tempalte.toml"):
        scene_from_toml(tmp_path, f'mpr_rotation_file = "{tmp_path}/tempalte.toml"')


def test_naming_no_rotation_file_is_not_an_error(tmp_path):
    assert scene_from_toml(tmp_path, "[tile]\nrows = 2").mpr_rotation_file is None
