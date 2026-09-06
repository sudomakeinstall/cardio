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
    ("toggle_maximized", {"view": "axial"}),
    ("place_camera", None),
]


def document(session: Session) -> dict:
    """Every key that says what the session is showing, as text.

    Text because the values include arrays, which do not answer ``==`` with a
    bool.
    """
    session.logic.camera.publish()
    return {
        key: repr(session.server.state[key]) for key in session.logic.document_keys()
    }


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

    assert reopen.scene.view.layout.value == "axial"
    assert reopen.server.state.maximized_view == "axial"


def test_the_quad_view_survives_being_the_empty_one(tmp_path):
    session = session_on(tmp_path)
    session.do("toggle_maximized", view="axial")
    session.do("toggle_maximized", view="axial")
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

    scene = scene_from_state(session.server.state, session.scene)

    assert scene.screenshot_viewports == ["axial", "coronal", "sagittal"]


def test_a_selected_preset_survives_beside_the_values_it_implies(tmp_path):
    """The two agree while a preset is selected, so writing both is no argument."""
    session = session_on(tmp_path)
    session.do("set_window_level_preset", preset=4)

    reopen = reopened(session.save(tmp_path / "saved.toml"))

    assert reopen.server.state.mpr_window_level_preset == 4
    assert reopen.server.state.mpr_window == session.server.state.mpr_window


def test_every_scalar_is_written_before_the_first_table(driven):
    """TOML reads a bare key as belonging to the table above it.

    A scalar written after one would come back meaning something else, so the
    dump orders them whatever order the model happens to be in.
    """
    body = to_toml(driven.scene_now())
    lines = [line for line in body.splitlines() if line and not line.startswith("#")]
    tables = [i for i, line in enumerate(lines) if line.startswith("[")]
    scalars = [
        i for i, line in enumerate(lines) if " = " in line and not line.startswith("[")
    ]

    assert not tables or max(scalars[: tables[0]], default=-1) < tables[0]
    assert tk.loads(body)


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
