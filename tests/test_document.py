"""Saving a session, and getting the same one back.

The round trip is the strongest thing the whole design says about itself. A
config is seeded into state key by key on the way in; saving walks the same map
backwards. If the two ever describe different apps, this is where it shows.
"""

# System
import itertools
import pathlib as pl

# Third Party
import pytest
import tomlkit as tk

import tests.test_app_smoke as smoke

# Internal
from cardio.document import scene_from_state, to_toml
from cardio.rotation import RotationMetadata, RotationSequence, RotationStep
from cardio.scene import Scene
from cardio.session import Session
from cardio.state import ObjectState
from tests.test_session import session_on

_names = itertools.count()

# Enough to move something in every corner of the document: the rotation
# sequence, a toggle, the window and level, the cameras, the layout, the crop.
DRIVEN = [
    ("add_rotation", {"axis": "Z"}),
    ("toggle_crosshairs", None),
    ("adjust_window_level", {"window_delta": 40.0, "level_delta": -10.0}),
    ("zoom_views", {"factor": 1.5}),
    ("toggle_maximized", {"view": "ul"}),
    ("place_camera", None),
]


def document(session: Session) -> dict:
    """Every key that says what the session is showing.

    The cameras are read out of VTK first: they are the one part of the
    document that a gesture can move without going through state.
    """
    session.logic.camera.publish()
    return smoke.document(session.server, session.logic)


def reopened(path: pl.Path) -> Session:
    session = Session.from_config(path, server=f"reopened-{next(_names)}")
    session.ready()
    return session


@pytest.fixture
def driven(tmp_path) -> Session:
    """A session taken somewhere, and away from every default it opened at.

    The actions are asked for; the per-object keys are written the way their
    checkboxes and sliders write them, there being no action behind those.
    """
    session = session_on(tmp_path)
    session.run(DRIVEN)

    volume = ObjectState.of(session.scene.volumes[0])
    mesh = ObjectState.of(session.scene.meshes[0])
    segmentation = ObjectState.of(session.scene.segmentations[0])

    with session.server.state:
        session.server.state[mesh.visibility] = False
        session.server.state[volume.clipping] = False
        session.server.state[volume.clip_y] = [-1.0, 2.5]
        session.server.state[volume.preset] = "xray"
        session.server.state[segmentation.mpr_overlay] = True

    return session


def test_a_saved_session_reopens_as_the_same_one(driven, tmp_path):
    """The whole of it, key by key, with nothing named here to go stale."""
    before = document(driven)
    after = document(reopened(driven.save(tmp_path / "saved.toml")))

    assert after == before


def test_the_round_trip_covers_the_whole_document(driven, tmp_path):
    """Guards the guard: comparing nothing would pass too."""
    assert len(document(driven)) > 40


def test_what_was_loaded_is_carried_over_rather_than_worked_out_again(driven, tmp_path):
    """Which files to read was never in state, so it cannot come from there."""
    reopen = reopened(driven.save(tmp_path / "saved.toml"))

    was, now = driven.scene.volumes[0], reopen.scene.volumes[0]
    assert (now.directory, now.file_paths) == (was.directory, was.file_paths)


def test_a_maximized_view_survives_the_layout_being_spelled_differently(
    driven, tmp_path
):
    """State says "" for the quad view; a config calls every layout by name."""
    reopen = reopened(driven.save(tmp_path / "saved.toml"))

    assert reopen.scene.view.layout.value == "ul"
    assert reopen.server.state.maximized_view == "ul"


def test_the_quad_view_survives_being_the_empty_one(tmp_path):
    session = session_on(tmp_path)
    session.do("toggle_maximized", view="ul")
    session.do("toggle_maximized", view="ul")
    assert session.server.state.maximized_view == ""

    reopen = reopened(session.save(tmp_path / "saved.toml"))

    assert reopen.scene.view.layout.value == "quad"
    assert reopen.server.state.maximized_view == ""


def test_a_crop_survives(tmp_path):
    """Which it could not before, there having been no field to write it to."""
    session = session_on(tmp_path)
    session.ready()
    keys = ObjectState.of(session.scene.volumes[0])
    with session.server.state:
        session.server.state[keys.clip_x] = [-1.0, 1.5]

    reopen = reopened(session.save(tmp_path / "saved.toml"))

    assert reopen.server.state[keys.clip_x] == [-1.0, 1.5]
    assert reopen.scene.volumes[0].crop[:2] == [-1.0, 1.5]


def test_the_ticked_viewports_are_what_is_saved(tmp_path):
    session = session_on(tmp_path)
    session.ready()
    with session.server.state:
        session.server.state.screenshot_viewport_tile = False
        session.server.state.screenshot_viewport_vr = False
        session.server.state.screenshot_viewport_volumetry = False

    scene = scene_from_state(session.server.state, session.scene)

    assert scene.screenshot_viewports == ["ul", "ll", "lr"]


def test_a_selected_preset_survives_beside_the_values_it_implies(tmp_path):
    """The two agree while a preset is selected, so writing both is no argument."""
    session = session_on(tmp_path)
    session.do("set_window_level_preset", preset=4)

    reopen = reopened(session.save(tmp_path / "saved.toml"))

    assert reopen.server.state.mpr_window_level_preset == 4
    assert reopen.server.state.mpr_window == session.server.state.mpr_window


def test_the_written_file_reads_back_as_the_document_it_was_given(driven):
    """Every field comes back saying what it said, and under the same table.

    TOML reads a bare key as belonging to the table above it, so a scalar
    written after one would come back meaning something else. Reading the file
    back is what says it did not.
    """
    scene = driven.scene_now()

    body = to_toml(scene)

    assert tk.loads(body) == scene.model_dump(mode="json", exclude_none=True)


def test_a_scene_with_nothing_to_snap_to_still_saves(tmp_path):
    """The snap selection is document state whatever the scene holds.

    A mesh has no labels to snap between, so the panel is never built and the
    controller has nothing to listen for. The keys behind it are still part of
    the document, and a save that left them out wrote a config that would not
    reload.
    """
    smoke.write_mesh(tmp_path / "mesh0.obj")
    scene = Scene(
        meshes=[{"label": "mesh", "directory": tmp_path, "file_paths": ["mesh0.obj"]}]
    )
    session = Session(scene, server=f"snapless-{next(_names)}")
    session.ready()

    reopen = reopened(session.save(tmp_path / "saved.toml"))

    assert reopen.server.state.snap_seg_label == ""
    assert not reopen.server.state.snap_locked


def test_a_configured_lock_is_kept_even_with_nothing_to_snap_to(tmp_path):
    """A lock is what the config asked for, whether or not it can take effect.

    Nothing listens to it on a scene with no segmentation, so it does not
    snap anything -- but it is still what the session says, and it used to be
    quietly turned off on the way in and lost on the way out. Snap itself
    takes the same view: asking for a lock that cannot act is a warning.
    """
    smoke.write_mesh(tmp_path / "mesh0.obj")
    scene = Scene(
        meshes=[{"label": "mesh", "directory": tmp_path, "file_paths": ["mesh0.obj"]}],
        snap={"locked": True},
    )
    session = Session(scene, server=f"locked-{next(_names)}")
    session.ready()

    assert session.server.state.snap_locked

    reopen = reopened(session.save(tmp_path / "saved.toml"))

    assert reopen.scene.snap.locked
    assert reopen.server.state.snap_locked


# ------------------------------------------- where the rotations come from ----
#
# A config says where the rotations come from either by naming a file or by
# spelling the sequence, and never by doing both: the file is read over
# whatever the config spelled, so writing both would show a sequence that
# opening it would throw away.


def rotation_file(directory: pl.Path, volume_label: str = "") -> pl.Path:
    """A rotation file holding one named step, for ``volume_label``."""
    path = directory / "rotations.toml"
    RotationSequence(
        metadata=RotationMetadata(volume_label=volume_label),
        angles_list=[RotationStep(axis="Z", angle=0.5, name="from the file")],
    ).to_file(path)
    return path


@pytest.fixture
def from_file(tmp_path) -> Session:
    """A session opened on a config that names a rotation file."""
    return session_on(tmp_path, mpr_rotation_file=rotation_file(tmp_path))


def test_a_config_naming_a_rotation_file_does_not_also_spell_the_sequence(
    from_file, tmp_path
):
    """Both would be one of them describing an app that opening it would not give."""
    body = tk.parse(from_file.save(tmp_path / "saved.toml").read_text())

    assert "mpr_rotation_file" in body
    assert "mpr_rotation_sequence" not in body


def test_a_saved_session_reopens_on_the_rotations_the_file_holds(from_file, tmp_path):
    reopen = reopened(from_file.save(tmp_path / "saved.toml"))

    assert [step.name for step in reopen.scene.mpr_rotation_sequence.angles_list] == [
        "from the file"
    ]


def test_a_named_file_goes_on_being_where_the_rotations_come_from(from_file, tmp_path):
    """What was changed in the app is saved by Save Rotations, not by this.

    Deliberate, and worth saying out loud: the alternative is a config that
    carries a sequence the file would overwrite the moment it was opened.
    """
    from_file.do("set_state", key="mpr_rotation_data.angles_list.0.angle", value=1.25)

    reopen = reopened(from_file.save(tmp_path / "saved.toml"))

    assert reopen.scene.mpr_rotation_sequence.angles_list[0].angle == 0.5


def test_a_session_with_no_rotation_file_still_saves_its_sequence(driven, tmp_path):
    """The other half of the rule: with no file named, the sequence is the answer."""
    body = tk.parse(driven.save(tmp_path / "saved.toml").read_text())

    assert "mpr_rotation_file" not in body
    assert body["mpr_rotation_sequence"]["angles_list"], "the rotation that was added"


def test_a_rotation_file_says_which_volume_it_is_for(tmp_path):
    """One file and one volume label are a matched pair, which is what makes a
    config point at a new study by changing the directory alone."""
    session = session_on(
        tmp_path, mpr_rotation_file=rotation_file(tmp_path, volume_label="vol")
    )

    assert session.scene.active_volume_label == "vol"
