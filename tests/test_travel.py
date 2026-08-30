"""Test that scrolling travels the traverse path instead of leaving the plane.

Traverse mode names a line through the volume, and the slider is that line, so
scrolling there scrubs the slider rather than moving the origin along a slice
normal. That drives the mode's own parameter: the origin and the orientation
are both derived from the fraction, and a lock derives its answer from the same
fraction, so travelling cooperates with a lock instead of being overwritten by
one. Every other mode scrolls out of the plane as before.

``FakeState`` does not fire trame's change listeners, so where the app would
realign as the fraction moved, these call ``align_to_interface`` themselves --
which is also what makes it visible that the gesture writes only the fraction.
"""

# Third Party
import numpy as np
import pytest

# Internal
from tests.test_traverse import make_logic, stacked_segmentation


@pytest.fixture
def logic(tmp_path):
    return make_logic(stacked_segmentation(tmp_path))


def traverse(logic):
    return logic.server.state.snap_traverse


# --- travelling the path -----------------------------------------------------


def test_scrolling_travels_the_path_in_traverse_mode(logic):
    logic.server.state.snap_traverse = 40

    logic.mpr.scroll_slice("axial", 5.0)

    assert traverse(logic) == 45


def test_scrolling_back_travels_the_other_way(logic):
    logic.server.state.snap_traverse = 40

    logic.mpr.scroll_slice("axial", -5.0)

    assert traverse(logic) == 35


def test_travelling_writes_only_the_fraction(logic):
    """The origin follows the fraction through the mode, not from the gesture."""
    logic.snap.align_to_interface()
    before = list(logic.server.state.mpr_origin)

    logic.mpr.scroll_slice("axial", 20.0)

    assert traverse(logic) == 20
    assert logic.server.state.mpr_origin == before

    logic.snap.align_to_interface()
    assert logic.server.state.mpr_origin != before


def test_the_path_is_travelled_from_any_view(logic):
    for view in ("axial", "sagittal", "coronal"):
        logic.server.state.snap_traverse = 50
        logic.mpr.scroll_slice(view, 10.0)
        assert traverse(logic) == 60


def test_travelling_stops_at_the_far_end(logic):
    logic.server.state.snap_traverse = 95

    logic.mpr.scroll_slice("axial", 20.0)

    assert traverse(logic) == 100


def test_travelling_stops_at_the_near_end(logic):
    logic.server.state.snap_traverse = 5

    logic.mpr.scroll_slice("axial", -20.0)

    assert traverse(logic) == 0


def test_holding_at_an_end_does_not_fall_back_to_moving_the_origin(logic):
    """Running out of path is still travelling, not scrolling out of it."""
    logic.server.state.snap_traverse = 100
    before = list(logic.server.state.mpr_origin)

    logic.mpr.scroll_slice("axial", 20.0)

    assert traverse(logic) == 100
    assert logic.server.state.mpr_origin == before


def test_a_slow_trackpad_scroll_still_travels(logic):
    """Fractions of a step are carried, or a slow scroll would round to nothing."""
    logic.server.state.snap_traverse = 0

    for _ in range(4):
        logic.mpr.scroll_slice("axial", 0.25)

    assert traverse(logic) == 1


# --- travelling against the locks --------------------------------------------


def test_a_position_lock_does_not_suspend_travelling(logic):
    """The lock derives its answer from the fraction, so this drives it."""
    logic.server.state.snap_locked = True
    logic.server.state.snap_traverse = 30

    logic.mpr.scroll_slice("axial", 10.0)

    assert traverse(logic) == 40


def test_an_orientation_lock_does_not_suspend_travelling(logic):
    logic.server.state.snap_orientation_locked = True
    logic.server.state.snap_traverse = 30

    logic.mpr.scroll_slice("axial", 10.0)

    assert traverse(logic) == 40


# --- every other mode is unchanged -------------------------------------------


@pytest.mark.parametrize("mode", ["label", "interface"])
def test_other_modes_still_scroll_out_of_the_plane(logic, mode):
    logic.server.state.snap_mode = mode
    before = list(logic.server.state.mpr_origin)

    logic.mpr.scroll_slice("axial", 5.0)

    assert logic.server.state.mpr_origin != before
    assert traverse(logic) == 0


@pytest.mark.parametrize("mode", ["label", "interface"])
def test_a_position_lock_still_suspends_scrolling_off_the_path(logic, mode):
    logic.server.state.snap_mode = mode
    logic.server.state.snap_locked = True
    before = list(logic.server.state.mpr_origin)

    logic.mpr.scroll_slice("axial", 5.0)

    assert logic.server.state.mpr_origin == before


def test_traverse_without_a_path_scrolls_out_of_the_plane(logic):
    """No third group, so there is no line: the gesture keeps its usual meaning."""
    logic.server.state.snap_labels_c = []
    before = list(logic.server.state.mpr_origin)

    logic.mpr.scroll_slice("axial", 5.0)

    assert logic.server.state.mpr_origin != before
    assert traverse(logic) == 0


def test_travelling_moves_along_the_line_it_names(logic):
    """Scrolling to the end lands where aligning at 100 percent would."""
    logic.server.state.snap_traverse = 0

    logic.mpr.scroll_slice("axial", 100.0)
    logic.snap.align_to_interface()
    travelled = list(logic.server.state.mpr_origin)

    logic.server.state.snap_traverse = 100
    logic.snap.align_to_interface()

    assert np.allclose(travelled, logic.server.state.mpr_origin)
