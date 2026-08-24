"""Test the out-of-plane scroll direction and the origin it moves.

``scroll_vector`` is the one convention-crossing computation that had no
coverage: it composes the visible rotations in ITK -- the only order the
rotation math is defined in -- and hands the result back in the order
``mpr_origin`` is stored in. ``test_interaction.py`` covers the right-drag that
calls it, but against a fake controller, so nothing checked the vector itself.
"""

# Third Party
import numpy as np
import pytest

# Internal
from cardio.convention import exchange_point, exchange_step
from cardio.orientation import AngleUnits, IndexOrder
from tests.fakes import FakeApp, FakeScene

# The out-of-plane direction of each view before any rotation, in ITK order.
BASE_NORMALS = {
    "axial": [0.0, 0.0, 1.0],
    "sagittal": [1.0, 0.0, 0.0],
    "coronal": [0.0, 1.0, 0.0],
}

# R_x(90) and R_y(90), written out rather than composed, so the expectations
# below are independent of orientation.py.
X90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
Y90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def make_app(
    steps=None,
    index_order=IndexOrder.ITK,
    angle_units=AngleUnits.DEGREES,
    origin=(0.0, 0.0, 0.0),
):
    """Real controllers over a scene holding only a rotation convention."""
    return FakeApp(
        FakeScene([], index_order=index_order, angle_units=angle_units),
        mpr_rotation_data={"angles_list": list(steps or [])},
        mpr_origin=list(origin),
    )


# --- an unrotated volume -----------------------------------------------------


@pytest.mark.parametrize("view,normal", BASE_NORMALS.items())
def test_unrotated_views_scroll_along_their_own_axis(view, normal):
    assert np.allclose(make_app().mpr.scroll_vector(view), normal)


def test_an_unknown_view_falls_back_to_the_axial_normal():
    """The volume viewport shares the handler but has no slice to scroll."""
    assert np.allclose(make_app().mpr.scroll_vector("vr"), [0.0, 0.0, 1.0])


@pytest.mark.parametrize("view,normal", BASE_NORMALS.items())
def test_roma_returns_the_normal_in_roma_order(view, normal):
    """No rotation, so the only work is the index exchange on the way out."""
    app = make_app(index_order=IndexOrder.ROMA)
    assert np.allclose(app.mpr.scroll_vector(view), exchange_point(normal))


# --- a rotated volume --------------------------------------------------------


@pytest.mark.parametrize(
    "view,expected",
    [
        ("axial", [0.0, -1.0, 0.0]),
        ("sagittal", [1.0, 0.0, 0.0]),
        ("coronal", [0.0, 0.0, 1.0]),
    ],
)
def test_a_single_rotation_turns_every_normal(view, expected):
    app = make_app([{"axis": "X", "angle": 90.0}])
    assert np.allclose(app.mpr.scroll_vector(view), expected)


def test_steps_compose_in_order():
    """X-then-Y and Y-then-X are different rotations, and the order is the
    order the user listed them in."""
    x_then_y = make_app([{"axis": "X", "angle": 90.0}, {"axis": "Y", "angle": 90.0}])
    y_then_x = make_app([{"axis": "Y", "angle": 90.0}, {"axis": "X", "angle": 90.0}])

    assert np.allclose(x_then_y.mpr.scroll_vector("axial"), X90 @ Y90 @ [0, 0, 1])
    assert np.allclose(y_then_x.mpr.scroll_vector("axial"), Y90 @ X90 @ [0, 0, 1])
    assert not np.allclose(
        x_then_y.mpr.scroll_vector("axial"), y_then_x.mpr.scroll_vector("axial")
    )


def test_radians_and_degrees_describe_the_same_rotation():
    degrees = make_app([{"axis": "X", "angle": 90.0}])
    radians = make_app(
        [{"axis": "X", "angle": np.pi / 2}], angle_units=AngleUnits.RADIANS
    )

    assert np.allclose(
        degrees.mpr.scroll_vector("axial"), radians.mpr.scroll_vector("axial")
    )


def test_hidden_steps_do_not_contribute():
    """The eye toggle in the rotation panel must move the slice normal back."""
    app = make_app([{"axis": "X", "angle": 90.0, "visible": False}])
    assert np.allclose(app.mpr.scroll_vector("axial"), [0.0, 0.0, 1.0])


def test_a_quaternion_step_matches_the_euler_step_it_encodes():
    half = np.sqrt(0.5)  # 90 degrees about X, as [x, y, z, w]
    euler = make_app([{"axis": "X", "angle": 90.0}])
    quaternion = make_app([{"quaternion": [half, 0.0, 0.0, half]}])

    assert np.allclose(
        euler.mpr.scroll_vector("axial"),
        quaternion.mpr.scroll_vector("axial"),
        atol=1e-9,
    )


# --- the convention boundary -------------------------------------------------


@pytest.mark.parametrize("view", BASE_NORMALS)
def test_switching_convention_does_not_move_the_slice(view):
    """The regression guard for the ITK<->ROMA switch.

    The same physical rotation, written once in each order, must scroll along
    the same physical direction -- which is the ROMA answer read back through
    the exchange. If either the step conversion or the vector conversion were
    dropped or applied twice, this is what would catch it.
    """
    steps = [{"axis": "X", "angle": 30.0}, {"axis": "Z", "angle": -45.0}]
    itk = make_app(steps)
    roma = make_app([exchange_step(step) for step in steps], IndexOrder.ROMA)

    assert np.allclose(
        exchange_point(roma.mpr.scroll_vector(view)), itk.mpr.scroll_vector(view)
    )


# --- moving the origin -------------------------------------------------------


def test_scrolling_walks_the_origin_along_the_normal():
    app = make_app(origin=(1.0, 2.0, 3.0))

    app.mpr.scroll_slice("axial", 2.5)

    assert app.server.state.mpr_origin == [1.0, 2.0, 5.5]


def test_scrolling_accumulates_and_reverses():
    app = make_app(origin=(0.0, 0.0, 0.0))

    app.mpr.scroll_slice("sagittal", 3.0)
    app.mpr.scroll_slice("sagittal", 3.0)
    app.mpr.scroll_slice("sagittal", -1.0)

    assert app.server.state.mpr_origin == [5.0, 0.0, 0.0]


def test_scrolling_follows_the_rotated_normal():
    app = make_app([{"axis": "X", "angle": 90.0}], origin=(0.0, 0.0, 0.0))

    app.mpr.scroll_slice("axial", 4.0)

    assert np.allclose(app.server.state.mpr_origin, [0.0, -4.0, 0.0])


def test_scrolling_an_unknown_view_leaves_the_origin_alone():
    """Unlike scroll_vector, which has a normal to fall back on, there is no
    sensible slice to move here -- the volume viewport must be inert."""
    app = make_app(origin=(1.0, 2.0, 3.0))

    app.mpr.scroll_slice("vr", 5.0)

    assert app.server.state.mpr_origin == [1.0, 2.0, 3.0]


def test_roma_scrolls_the_same_physical_axis_as_itk():
    """mpr_origin is stored in the user's order, so the axis that moves is the
    exchanged one -- ROMA index 0 and ITK index 2 are both S."""
    itk = make_app(origin=(0.0, 0.0, 0.0))
    roma = make_app(index_order=IndexOrder.ROMA, origin=(0.0, 0.0, 0.0))

    itk.mpr.scroll_slice("axial", 2.0)
    roma.mpr.scroll_slice("axial", 2.0)

    assert itk.server.state.mpr_origin == [0.0, 0.0, 2.0]
    assert roma.server.state.mpr_origin == [2.0, 0.0, 0.0]
