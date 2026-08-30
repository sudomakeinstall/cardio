"""Test that a gesture stands down when snap is holding the state it writes.

Centring and aligning write the origin and the alignment step once; a lock
makes snap their standing owner and re-applies them on every frame. A gesture
that wrote the same state under a lock would look like it worked and then snap
back the next time the frame changed, so it is refused instead.

Pan and slice scroll both move ``mpr_origin``, so a position lock stands both
down. Rotation is stood down by an orientation lock, which is a lock on the
alignment step it composes on top of.
"""

# System
import math

# Third Party
import numpy as np
import pytest

# Internal
from cardio.orientation import AngleUnits, IndexOrder
from cardio.rotation import RotationStep
from tests.fakes import FakeApp, FakeScene

CENTRE = (200.0, 150.0)
ORIGIN = [1.0, 2.0, 3.0]


class Views:
    def origin_on_screen(self, view):
        return CENTRE

    def world_per_pixel(self, view):
        return 1.0


def make_app(**state):
    return FakeApp(
        FakeScene(
            [],
            index_order=IndexOrder.ITK,
            angle_units=AngleUnits.DEGREES,
            mpr_views=Views(),
        ),
        mpr_rotation_data={"angles_list": []},
        mpr_origin=list(ORIGIN),
        **state,
    )


def spoke(bearing, radius=100.0):
    angle = math.radians(bearing)
    return (
        CENTRE[0] + radius * math.cos(angle),
        CENTRE[1] + radius * math.sin(angle),
    )


def roll(app, view="axial", degrees=25.0):
    app.mpr.rotate_view(view, spoke(0.0), spoke(-degrees))


def steps(app):
    return app.rotations.rotation_sequence().angles_list


# --- what the locks mean -----------------------------------------------------


def test_no_lock_owns_nothing_by_default():
    app = make_app()

    assert app.snap.position_locked is False
    assert app.snap.orientation_locked is False


def test_a_position_lock_reads_as_owned():
    assert make_app(snap_locked=True).snap.position_locked is True


@pytest.mark.parametrize("mode", ["interface", "traverse"])
def test_an_orientation_lock_holds_only_in_the_planar_modes(mode):
    app = make_app(snap_orientation_locked=True, snap_mode=mode)

    assert app.snap.orientation_locked is True


def test_label_mode_cannot_hold_an_orientation():
    """There is no plane fitted to hold, so the flag means nothing there."""
    app = make_app(snap_orientation_locked=True, snap_mode="label")

    assert app.snap.orientation_locked is False


# --- a locked position stands down pan and scroll ----------------------------


def test_panning_is_refused_while_the_position_is_locked():
    app = make_app(snap_locked=True)

    app.mpr.pan_view("axial", 30.0, 20.0)

    assert app.server.state.mpr_origin == ORIGIN


@pytest.mark.parametrize("view", ["axial", "sagittal", "coronal"])
def test_scrolling_is_refused_while_the_position_is_locked(view):
    """The wheel goes through scroll_slice too, so it stands down with it."""
    app = make_app(snap_locked=True)

    app.mpr.scroll_slice(view, 5.0)

    assert app.server.state.mpr_origin == ORIGIN


def test_panning_and_scrolling_work_once_the_lock_is_off():
    """Centring without locking must leave every gesture available."""
    app = make_app(snap_locked=False)

    app.mpr.pan_view("axial", 30.0, 20.0)
    app.mpr.scroll_slice("axial", 5.0)

    assert app.server.state.mpr_origin != ORIGIN


def test_a_position_lock_does_not_stand_down_rotation():
    """It holds the origin, not the orientation."""
    app = make_app(snap_locked=True)

    roll(app)

    assert steps(app) != []


# --- a locked orientation stands down rotation -------------------------------


def test_rotating_is_refused_while_the_orientation_is_locked():
    app = make_app(snap_orientation_locked=True, snap_mode="interface")

    roll(app)

    assert steps(app) == []


def test_rotating_works_in_label_mode_despite_the_flag():
    """Label mode fits no plane, so nothing owns the orientation there."""
    app = make_app(snap_orientation_locked=True, snap_mode="label")

    roll(app)

    assert steps(app) != []


def test_rotating_works_once_the_orientation_lock_is_off():
    """Align then rotate is the layering workflow, and must stay open."""
    app = make_app(snap_orientation_locked=False, snap_mode="interface")

    roll(app)

    assert steps(app) != []


def test_an_orientation_lock_does_not_stand_down_pan_or_scroll():
    """It holds the alignment step, not the origin."""
    app = make_app(snap_orientation_locked=True, snap_mode="interface")

    app.mpr.pan_view("axial", 30.0, 20.0)

    assert app.server.state.mpr_origin != ORIGIN


# --- both at once ------------------------------------------------------------


def test_both_locks_stand_down_everything_that_writes_either():
    app = make_app(snap_locked=True, snap_orientation_locked=True, snap_mode="traverse")

    app.mpr.pan_view("axial", 30.0, 20.0)
    app.mpr.scroll_slice("axial", 5.0)
    roll(app)

    assert app.server.state.mpr_origin == ORIGIN
    assert steps(app) == []


def test_zoom_is_never_stood_down():
    """Zoom is camera framing and writes no shared state, so no lock owns it."""
    zoomed = []
    app = make_app(snap_locked=True, snap_orientation_locked=True, snap_mode="traverse")
    app.scene.mpr_views.zoom = zoomed.append

    app.mpr.zoom_views(2.0)

    assert zoomed == [2.0]


def test_window_level_is_never_stood_down():
    app = make_app(
        snap_locked=True,
        snap_orientation_locked=True,
        snap_mode="traverse",
        mpr_window=100.0,
        mpr_level=50.0,
    )

    app.mpr.adjust_window_level(10.0, 5.0)

    assert app.server.state.mpr_window == 110.0
    assert app.server.state.mpr_level == 55.0


def test_the_lock_that_re_applies_is_the_lock_that_stands_gestures_down():
    """apply_frame_lock and the gesture guards must not drift apart."""
    app = make_app(snap_locked=True, snap_orientation_locked=True, snap_mode="label")

    # label mode holds a position but no orientation, both here and in the guard
    assert app.snap.position_locked is True
    assert app.snap.orientation_locked is False

    app.mpr.pan_view("axial", 30.0, 20.0)
    roll(app)

    assert app.server.state.mpr_origin == ORIGIN
    assert steps(app) != []


def test_rotation_still_layers_on_an_alignment_when_unlocked():
    """Align, then rotate: the mouse step composes after the alignment step."""
    app = make_app(snap_mode="interface")
    half = float(np.sqrt(0.5))
    alignment = RotationStep(
        quaternion=[half, 0.0, 0.0, half],
        name="Interface plane",
        name_editable=False,
    )
    app.rotations.edit_steps(lambda existing: [alignment, *existing])

    roll(app)

    assert [s.name for s in steps(app)] == ["Interface plane", "Mouse S-I"]
