"""Test the in-plane pan direction and the origin it moves.

``pan_view`` is the in-plane sibling of ``scroll_slice``: both move the one
shared ``mpr_origin``, so all three views stay a single point of view on the
volume rather than three that can drift apart. ``pan_vectors`` is the
convention-crossing half, composed in ITK and read back in the order
``mpr_origin`` is stored in, exactly as ``scroll_vector`` is.

The scale a drag is measured at comes from the camera and is covered against
real render windows in ``test_mpr_views.py``; here it is fixed, so these tests
are about direction and arithmetic alone.
"""

# Third Party
import numpy as np
import pytest

# Internal
from cardio.convention import exchange_point, exchange_step
from cardio.orientation import AngleUnits, IndexOrder
from cardio.reslice import VIEWS
from tests.fakes import FakeApp, FakeScene

# The in-plane (right, up) axes of each view before any rotation, in ITK order.
# Columns 0 and 1 of each view's axcode: LAS, ASL, LSA.
BASE_AXES = {
    "axial": ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),  # Left, Anterior
    "sagittal": ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),  # Anterior, Superior
    "coronal": ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),  # Left, Superior
}

X90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class FixedScaleViews:
    """MPRViews with a known world-units-per-pixel, so drags are exact."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def world_per_pixel(self, view: str) -> float:
        return self.scale


def make_app(
    steps=None,
    index_order=IndexOrder.ITK,
    angle_units=AngleUnits.DEGREES,
    origin=(0.0, 0.0, 0.0),
    scale=1.0,
    mpr_views=...,
):
    views = FixedScaleViews(scale) if mpr_views is ... else mpr_views
    return FakeApp(
        FakeScene(
            [], index_order=index_order, angle_units=angle_units, mpr_views=views
        ),
        mpr_rotation_data={"angles_list": list(steps or [])},
        mpr_origin=list(origin),
    )


# --- the in-plane basis ------------------------------------------------------


@pytest.mark.parametrize("view,axes", BASE_AXES.items())
def test_unrotated_views_lie_in_their_own_plane(view, axes):
    right, up = make_app().mpr.pan_vectors(view)
    assert np.allclose(right, axes[0])
    assert np.allclose(up, axes[1])


@pytest.mark.parametrize("view", VIEWS)
def test_the_in_plane_axes_are_perpendicular_to_the_scroll_normal(view):
    """Pan and scroll must divide the space, not overlap in it."""
    app = make_app([{"axis": "X", "angle": 30.0}, {"axis": "Z", "angle": -45.0}])
    right, up = app.mpr.pan_vectors(view)
    normal = app.mpr.scroll_vector(view)

    assert np.dot(right, normal) == pytest.approx(0.0, abs=1e-9)
    assert np.dot(up, normal) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("view", VIEWS)
def test_the_in_plane_axes_are_a_unit_orthogonal_pair(view):
    right, up = make_app([{"axis": "Z", "angle": 20.0}]).mpr.pan_vectors(view)

    assert np.linalg.norm(right) == pytest.approx(1.0)
    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert np.dot(right, up) == pytest.approx(0.0, abs=1e-9)


def test_a_rotation_turns_the_in_plane_axes(view="axial"):
    right, up = make_app([{"axis": "X", "angle": 90.0}]).mpr.pan_vectors(view)

    assert np.allclose(right, X90 @ BASE_AXES[view][0])
    assert np.allclose(up, X90 @ BASE_AXES[view][1])


def test_hidden_steps_do_not_contribute():
    app = make_app([{"axis": "X", "angle": 90.0, "visible": False}])
    right, up = app.mpr.pan_vectors("axial")

    assert np.allclose(right, BASE_AXES["axial"][0])
    assert np.allclose(up, BASE_AXES["axial"][1])


@pytest.mark.parametrize("view", VIEWS)
def test_roma_returns_the_axes_in_roma_order(view):
    right, up = make_app(index_order=IndexOrder.ROMA).mpr.pan_vectors(view)

    assert np.allclose(right, exchange_point(BASE_AXES[view][0]))
    assert np.allclose(up, exchange_point(BASE_AXES[view][1]))


# --- moving the origin -------------------------------------------------------


def test_panning_moves_the_origin_against_the_drag():
    """The origin travels the other way, so the image follows the cursor."""
    app = make_app(origin=(0.0, 0.0, 0.0))

    app.mpr.pan_view("axial", 3.0, 0.0)

    assert np.allclose(app.server.state.mpr_origin, [-3.0, 0.0, 0.0])


def test_panning_combines_both_axes():
    app = make_app(origin=(0.0, 0.0, 0.0))

    app.mpr.pan_view("axial", 3.0, 5.0)

    # -3 * Left - 5 * Anterior
    assert np.allclose(app.server.state.mpr_origin, [-3.0, 5.0, 0.0])


def test_panning_scales_with_the_world_per_pixel():
    """A drag is a grab, so a pixel must cover the world the camera shows."""
    app = make_app(origin=(0.0, 0.0, 0.0), scale=0.25)

    app.mpr.pan_view("axial", 8.0, 0.0)

    assert np.allclose(app.server.state.mpr_origin, [-2.0, 0.0, 0.0])


def test_panning_accumulates_and_reverses():
    app = make_app(origin=(1.0, 2.0, 3.0))

    app.mpr.pan_view("sagittal", 4.0, 2.0)
    app.mpr.pan_view("sagittal", -4.0, -2.0)

    assert np.allclose(app.server.state.mpr_origin, [1.0, 2.0, 3.0])


def test_panning_stays_in_the_plane_it_started_in():
    """Panning must never scroll: the normal component stays untouched."""
    app = make_app([{"axis": "X", "angle": 30.0}], origin=(0.0, 0.0, 0.0))

    app.mpr.pan_view("coronal", 7.0, -3.0)

    normal = app.mpr.scroll_vector("coronal")
    assert np.dot(app.server.state.mpr_origin, normal) == pytest.approx(0.0, abs=1e-9)


def test_panning_follows_the_rotated_axes():
    app = make_app([{"axis": "X", "angle": 90.0}], origin=(0.0, 0.0, 0.0))

    app.mpr.pan_view("axial", 2.0, 0.0)

    assert np.allclose(app.server.state.mpr_origin, -2.0 * (X90 @ [1.0, 0.0, 0.0]))


def test_panning_an_unknown_view_leaves_the_origin_alone():
    """The volume viewport shares the handler but has no plane to slide in."""
    app = make_app(origin=(1.0, 2.0, 3.0))

    app.mpr.pan_view("vr", 5.0, 5.0)

    assert app.server.state.mpr_origin == [1.0, 2.0, 3.0]


def test_panning_before_the_views_exist_leaves_the_origin_alone():
    app = make_app(origin=(1.0, 2.0, 3.0), mpr_views=None)

    app.mpr.pan_view("axial", 5.0, 5.0)

    assert app.server.state.mpr_origin == [1.0, 2.0, 3.0]


def test_panning_an_unsized_view_leaves_the_origin_alone():
    """world_per_pixel reports zero until the window has a viewport."""
    app = make_app(origin=(1.0, 2.0, 3.0), scale=0.0)

    app.mpr.pan_view("axial", 5.0, 5.0)

    assert app.server.state.mpr_origin == [1.0, 2.0, 3.0]


def test_roma_pans_the_same_physical_axes_as_itk():
    """The regression guard for the ITK<->ROMA switch, as scrolling has."""
    steps = [{"axis": "X", "angle": 30.0}, {"axis": "Z", "angle": -45.0}]
    itk = make_app(steps)
    roma = make_app([exchange_step(step) for step in steps], IndexOrder.ROMA)

    itk.mpr.pan_view("axial", 6.0, -2.0)
    roma.mpr.pan_view("axial", 6.0, -2.0)

    assert np.allclose(
        exchange_point(roma.server.state.mpr_origin), itk.server.state.mpr_origin
    )
