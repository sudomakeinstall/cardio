"""Test the mouse-driven rotation: an in-plane roll of the view being dragged.

The gesture spins the slice frame about the dragged view's own normal -- the
axis ``scroll_slice`` travels along -- so the axial view turns about the
craniocaudal axis and goes on showing the same cut. That normal is a signed
L/A/S direction before any rotation is applied, so a drag lands in
``mpr_rotation_data`` as an ordinary Euler step rather than a quaternion:
readable and editable in the rotations panel, and leaving ``rotation_sequence``
the single source of rotation truth.

How far it turns is the angle the cursor sweeps about the origin, so the tests
drive it with two spokes rather than an angle, the way the view events do.
"""

# System
import math

# Third Party
import numpy as np
import pytest

# Internal
from cardio.logic.rotations import MOUSE_STEP_NAMES
from cardio.orientation import AngleUnits, IndexOrder
from cardio.reslice import VIEW_TRANSFORMS, VIEWS
from tests.fakes import FakeApp, FakeScene

# The axis each view rolls about, named by the ITK axis it lies along.
VIEW_ROLL_AXIS = {"axial": "Z", "sagittal": "X", "coronal": "Y"}


CENTRE = (200.0, 150.0)


class CentredViews:
    """MPRViews with the origin at a known point on screen."""

    def origin_on_screen(self, view: str):
        return CENTRE


def make_app(steps=None, index_order=IndexOrder.ITK, angle_units=AngleUnits.DEGREES):
    return FakeApp(
        FakeScene(
            [],
            index_order=index_order,
            angle_units=angle_units,
            mpr_views=CentredViews(),
        ),
        mpr_rotation_data={"angles_list": list(steps or [])},
        mpr_origin=[0.0, 0.0, 0.0],
    )


def spoke(bearing: float, radius: float = 100.0):
    """A cursor position at ``bearing`` degrees anticlockwise from 3 o'clock."""
    angle = math.radians(bearing)
    return (
        CENTRE[0] + radius * math.cos(angle),
        CENTRE[1] + radius * math.sin(angle),
    )


def sweep(app, view, degrees, radius=100.0, start_bearing=0.0):
    """Drag from one spoke to another, sweeping ``degrees`` clockwise."""
    app.mpr.rotate_view(
        view,
        spoke(start_bearing, radius),
        spoke(start_bearing - degrees, radius),
    )


def steps_of(app):
    return app.rotations.rotation_sequence().angles_list


def mouse_steps(app):
    return [s for s in steps_of(app) if s.name in MOUSE_STEP_NAMES.values()]


def image_after(view, degrees, **kwargs):
    """Where a landmark at the top of the view lands, before and after a sweep."""
    app = make_app()
    frame = VIEW_TRANSFORMS[view]
    top = np.array(frame[:, 1]) * 10.0
    before = frame.T @ top

    sweep(app, view, degrees, **kwargs)

    return before, (app.rotations.rotation_matrix() @ frame).T @ top


def top_of_image_moves(view, degrees):
    """How far that landmark slides in output x."""
    before, after = image_after(view, degrees)
    return after[0] - before[0]


def content_turn(view, degrees, **kwargs):
    """How far the image actually turns, in degrees clockwise."""
    before, after = image_after(view, degrees, **kwargs)
    return -math.degrees(
        math.atan2(
            before[0] * after[1] - before[1] * after[0],
            before[0] * after[0] + before[1] * after[1],
        )
    )


# --- an in-plane roll --------------------------------------------------------


@pytest.mark.parametrize("view", VIEWS)
def test_rolling_leaves_the_cut_alone(view):
    """The whole point: the plane spins, it does not tilt out of itself."""
    app = make_app()
    normal = VIEW_TRANSFORMS[view][:, 2]

    sweep(app, view, 25.0)

    assert np.allclose(app.rotations.rotation_matrix() @ normal, normal)


@pytest.mark.parametrize("view", VIEWS)
def test_rolling_leaves_the_dragged_views_scroll_axis_fixed(view):
    """Roll and scroll share an axis, so rolling never redirects the scroll.

    Compared before against after rather than against the axcode column, which
    would bake in ``scroll_vector``'s sign for the coronal normal.
    """
    app = make_app()
    before = app.mpr.scroll_vector(view)

    sweep(app, view, 25.0)

    assert np.allclose(app.mpr.scroll_vector(view), before)


@pytest.mark.parametrize("view,axis", VIEW_ROLL_AXIS.items())
def test_each_view_rolls_about_its_own_normal(view, axis):
    """Axial turns about the craniocaudal axis, and so on round the three."""
    app = make_app()

    sweep(app, view, 10.0)

    assert [s.name for s in mouse_steps(app)] == [MOUSE_STEP_NAMES[axis]]


@pytest.mark.parametrize("view", VIEWS)
def test_a_drag_writes_one_euler_step_not_a_quaternion(view):
    app = make_app()

    sweep(app, view, 10.0)

    (step,) = mouse_steps(app)
    assert step.quaternion is None
    assert step.axis in ("X", "Y", "Z")


# --- which way it turns ------------------------------------------------------


@pytest.mark.parametrize("view", VIEWS)
def test_dragging_right_spins_clockwise_in_every_view(view):
    """The axcode frames disagree on handedness; the gesture must not."""
    assert top_of_image_moves(view, 20.0) > 0


@pytest.mark.parametrize("view", VIEWS)
def test_dragging_back_the_other_way_spins_back(view):
    assert top_of_image_moves(view, -20.0) < 0


# --- what a drag writes ------------------------------------------------------


def test_dragging_on_about_the_same_axis_accumulates_into_one_step():
    app = make_app()

    sweep(app, "axial", 10.0)
    sweep(app, "axial", 10.0)

    (step,) = mouse_steps(app)
    assert step.angle == pytest.approx(-20.0)


def test_dragging_back_cancels():
    app = make_app()

    sweep(app, "axial", 12.0)
    sweep(app, "axial", -12.0)

    assert np.allclose(app.rotations.rotation_matrix(), np.eye(3))


def test_rolling_two_views_keeps_two_steps():
    app = make_app()

    sweep(app, "axial", 10.0)
    sweep(app, "sagittal", 10.0)

    assert {s.name for s in mouse_steps(app)} == {
        MOUSE_STEP_NAMES["Z"],
        MOUSE_STEP_NAMES["X"],
    }


def test_the_axis_being_dragged_goes_last():
    """Last is composed first, the only place a roll stays in its own plane."""
    app = make_app()

    sweep(app, "axial", 6.0)
    sweep(app, "coronal", 9.0)

    assert [s.name for s in mouse_steps(app)] == [
        MOUSE_STEP_NAMES["Z"],
        MOUSE_STEP_NAMES["Y"],
    ]


def test_changing_axis_starts_a_step_rather_than_reordering():
    """Reordering would recompose the rotation and move the views."""
    app = make_app()

    sweep(app, "axial", 6.0)
    sweep(app, "coronal", 9.0)
    sweep(app, "axial", 6.0)

    assert [s.name for s in mouse_steps(app)] == [
        MOUSE_STEP_NAMES["Z"],
        MOUSE_STEP_NAMES["Y"],
        MOUSE_STEP_NAMES["Z"],
    ]


@pytest.mark.parametrize("prior", VIEWS)
@pytest.mark.parametrize("view", VIEWS)
def test_a_roll_stays_in_plane_whatever_was_rolled_before(prior, view):
    """The regression guard: a step left outside the new one tilts the plane."""
    app = make_app()
    sweep(app, prior, 40.0)

    frame = VIEW_TRANSFORMS[view]
    normal = app.rotations.rotation_matrix() @ frame[:, 2]

    sweep(app, view, 30.0)

    assert np.allclose(app.rotations.rotation_matrix() @ frame[:, 2], normal)


@pytest.mark.parametrize("prior", VIEWS)
@pytest.mark.parametrize("view", VIEWS)
def test_a_roll_turns_by_the_swept_angle_whatever_was_rolled_before(prior, view):
    app = make_app()
    sweep(app, prior, 40.0)

    frame = VIEW_TRANSFORMS[view]
    top = np.array(frame[:, 1]) * 10.0
    before = (app.rotations.rotation_matrix() @ frame).T @ top

    sweep(app, view, 30.0)

    after = (app.rotations.rotation_matrix() @ frame).T @ top
    turned = -math.degrees(
        math.atan2(
            before[0] * after[1] - before[1] * after[0],
            before[0] * after[0] + before[1] * after[1],
        )
    )
    assert turned == pytest.approx(30.0)


def test_a_hidden_mouse_step_is_not_accumulated_into():
    """The eye toggle would otherwise swallow the drag."""
    app = make_app()
    sweep(app, "axial", 10.0)

    # rotation_sequence() parses state afresh, so hide it on one object and
    # publish that same one back
    sequence = app.rotations.rotation_sequence()
    sequence.angles_list[-1].visible = False
    app.rotations.publish(sequence)

    sweep(app, "axial", 10.0)

    assert len(mouse_steps(app)) == 2


def test_a_still_drag_writes_nothing():
    app = make_app()

    sweep(app, "axial", 0.0)

    assert steps_of(app) == []


def test_rotating_an_unknown_view_writes_nothing():
    app = make_app()

    sweep(app, "vr", 10.0)

    assert steps_of(app) == []


def test_steps_the_user_built_are_kept_and_stay_first():
    """A hand-built rotation keeps its meaning underneath the mouse step."""
    app = make_app([{"axis": "Z", "angle": 30.0, "name": "Mine"}])

    sweep(app, "axial", 10.0)

    assert steps_of(app)[0].name == "Mine"
    assert steps_of(app)[0].angle == 30.0


@pytest.mark.parametrize("view", VIEWS)
def test_rotating_leaves_the_frame_orthonormal(view):
    app = make_app()

    sweep(app, view, 21.0)

    rotation = app.rotations.rotation_matrix()
    assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_rotating_does_not_move_the_origin():
    """Turning is not travelling: the crosshair stays on the same point."""
    app = make_app()
    app.server.state.mpr_origin = [1.0, 2.0, 3.0]

    sweep(app, "axial", 20.0)

    assert app.server.state.mpr_origin == [1.0, 2.0, 3.0]


# --- the convention boundary -------------------------------------------------


def test_radians_sequences_store_the_angle_in_radians():
    degrees = make_app(angle_units=AngleUnits.DEGREES)
    radians = make_app(angle_units=AngleUnits.RADIANS)

    sweep(degrees, "axial", 90.0)
    sweep(radians, "axial", 90.0)

    assert np.allclose(
        np.radians(mouse_steps(degrees)[0].angle), mouse_steps(radians)[0].angle
    )


def test_roma_turns_the_same_physical_axis_as_itk():
    """The regression guard for the ITK<->ROMA switch, as scroll and pan have."""
    itk = make_app(index_order=IndexOrder.ITK)
    roma = make_app(index_order=IndexOrder.ROMA)

    sweep(itk, "axial", 12.0)
    sweep(roma, "axial", 12.0)

    assert np.allclose(
        itk.rotations.rotation_matrix(), roma.rotations.rotation_matrix()
    )


# --- the swept angle ---------------------------------------------------------


@pytest.mark.parametrize("view", VIEWS)
def test_the_image_turns_by_exactly_the_angle_swept(view):
    """A direct grab: no sensitivity sits between the cursor and the image."""
    assert content_turn(view, 30.0) == pytest.approx(30.0)


@pytest.mark.parametrize("radius", [30.0, 100.0, 400.0])
def test_the_radius_does_not_change_the_angle(radius):
    """Only the angle between the spokes counts, not how long they are."""
    assert content_turn("axial", 45.0, radius=radius) == pytest.approx(45.0)


@pytest.mark.parametrize("bearing", [0.0, 90.0, 200.0, -75.0])
def test_where_the_drag_starts_does_not_change_the_angle(bearing):
    assert content_turn("axial", 45.0, start_bearing=bearing) == pytest.approx(45.0)


def test_a_radial_drag_does_not_turn_the_image():
    """Straight out from the origin sweeps no angle at all."""
    app = make_app()

    app.mpr.rotate_view("axial", spoke(0.0, 60.0), spoke(0.0, 250.0))

    assert steps_of(app) == []


def test_a_drag_across_the_origin_does_not_spin_wildly():
    """Near the centre a pixel is a huge angle, so the short spokes are ignored."""
    app = make_app()

    app.mpr.rotate_view(
        "axial", (CENTRE[0] + 1, CENTRE[1] + 1), (CENTRE[0] - 1, CENTRE[1] - 1)
    )

    assert steps_of(app) == []


def test_a_sweep_beyond_a_half_turn_reads_as_the_short_way_round():
    """Per-event sweeps are small; the wrap only matters if one is not."""
    assert content_turn("axial", 350.0) == pytest.approx(-10.0)


def test_rotating_before_the_views_exist_writes_nothing():
    app = make_app()
    app.scene.mpr_views = None

    sweep(app, "axial", 30.0)

    assert steps_of(app) == []


def test_rotating_before_the_window_is_sized_writes_nothing():
    """origin_on_screen has no answer until the renderer has a viewport."""

    class UnsizedViews:
        def origin_on_screen(self, view):
            return None

    app = make_app()
    app.scene.mpr_views = UnsizedViews()

    sweep(app, "axial", 30.0)

    assert steps_of(app) == []
