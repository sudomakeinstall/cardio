"""The planar geometry a traced region is measured by.

Arrays in, arrays out, so none of this needs a render window or a server. What
it is *about* is that an area read off a cut is the area of the region on the
patient: that the cut's two directions are orthonormal, that the shoelace sum
over the millimetres the user clicked is the same number, and that moving
within the plane leaves both alone.
"""

# Third Party
import numpy as np
import pytest

# Internal
from cardio.orientation import AngleUnits, EulerAxis, euler_angle_to_rotation_matrix
from cardio.planimetry import (
    ContourStyle,
    Cut,
    closed_spline,
    contour_area,
    contour_points,
    cut_from,
    from_lps,
    is_click,
    polygon_area,
    same_plane,
    to_lps,
)
from cardio.reslice import VIEWS

SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]


def turn(axis: str, degrees: float) -> np.ndarray:
    return euler_angle_to_rotation_matrix(EulerAxis(axis), degrees, AngleUnits.DEGREES)


def circle(count: int, radius: float = 1.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


# --- polygon_area ---------------------------------------------------------


def test_the_area_of_a_square_is_its_side_squared():
    assert polygon_area(SQUARE) == pytest.approx(4.0)


def test_the_area_of_a_right_triangle_is_half_its_box():
    assert polygon_area([(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)]) == pytest.approx(6.0)


def test_tracing_the_other_way_round_gives_the_same_area():
    assert polygon_area(SQUARE[::-1]) == pytest.approx(polygon_area(SQUARE))


@pytest.mark.parametrize("points", [[], [(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)]])
def test_fewer_than_three_points_enclose_nothing(points):
    assert polygon_area(points) == 0.0


def test_a_finer_polygon_approaches_the_circle_it_is_drawn_on():
    coarse = polygon_area(circle(8))
    fine = polygon_area(circle(128))
    assert coarse < fine < np.pi
    assert fine == pytest.approx(np.pi, rel=1e-3)


def test_the_area_does_not_move_with_the_region():
    moved = np.asarray(SQUARE) + np.array([17.0, -4.0])
    assert polygon_area(moved) == pytest.approx(polygon_area(SQUARE))


# --- closed_spline --------------------------------------------------------


def test_the_spline_passes_through_every_point_it_was_given():
    curve = closed_spline(SQUARE, per_segment=8)
    for point in SQUARE:
        assert np.isclose(curve, point).all(axis=1).any()


def test_the_spline_is_one_sample_run_per_segment():
    assert len(closed_spline(circle(5), per_segment=8)) == 40


def test_the_spline_closes_rather_than_doubling_a_point():
    curve = closed_spline(SQUARE, per_segment=8)
    assert not np.allclose(curve[0], curve[-1])
    # The last sample is a step short of coming back round to the first.
    assert np.linalg.norm(curve[-1] - curve[0]) < np.linalg.norm(curve[0] - curve[4])


def test_the_spline_bulges_outside_the_polygon_through_the_same_points():
    assert polygon_area(closed_spline(SQUARE)) > polygon_area(SQUARE)


def test_a_spline_recovers_far_more_of_a_circle_than_its_points_do():
    points = circle(8)
    straight = np.pi - polygon_area(points)
    curved = np.pi - polygon_area(closed_spline(points))
    assert 0.0 < curved < straight / 10.0


def test_the_sample_count_is_past_where_the_spline_s_area_settles():
    points = circle(8)
    settled = polygon_area(closed_spline(points, per_segment=256))
    assert polygon_area(closed_spline(points)) == pytest.approx(settled, rel=1e-3)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_too_few_points_to_curve_are_handed_back(count):
    points = circle(count) if count else np.empty((0, 2))
    assert closed_spline(points).shape == points.shape


# --- contour_points / contour_area ----------------------------------------


def test_a_polygon_contour_is_the_points_themselves():
    assert np.array_equal(contour_points(SQUARE, ContourStyle.POLYGON), SQUARE)


def test_the_polygon_area_is_the_area_of_the_polygon():
    assert contour_area(SQUARE, ContourStyle.POLYGON) == pytest.approx(4.0)


def test_the_two_styles_measure_the_curves_they_draw():
    for style in ContourStyle:
        assert contour_area(SQUARE, style) == pytest.approx(
            polygon_area(contour_points(SQUARE, style))
        )


def test_the_style_may_be_named_as_the_document_spells_it():
    assert contour_area(SQUARE, "polygon") == pytest.approx(4.0)


# --- cut_from -------------------------------------------------------------


@pytest.mark.parametrize("view", VIEWS)
def test_a_cut_frame_is_orthonormal(view):
    cut = cut_from([1.0, 2.0, 3.0], turn("Z", 37.0) @ turn("X", 11.0), view)
    right, up, normal = np.array(cut.right), np.array(cut.up), cut.normal

    assert np.linalg.norm(right) == pytest.approx(1.0)
    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert right @ up == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(normal) == pytest.approx(1.0)


@pytest.mark.parametrize("view", VIEWS)
def test_the_cut_origin_is_the_point_the_view_was_aimed_at(view):
    origin = [4.0, -3.0, 12.0]
    cut = cut_from(origin, turn("Y", 25.0), view)
    assert to_lps(cut, [(0.0, 0.0)])[0] == pytest.approx(origin)


# --- to_lps / from_lps ----------------------------------------------------


def test_cut_millimetres_and_patient_millimetres_round_trip():
    cut = cut_from([2.0, -5.0, 7.0], turn("Z", 40.0) @ turn("X", 20.0), "ul")
    points = circle(9, radius=13.0)
    assert from_lps(cut, to_lps(cut, points)) == pytest.approx(points)


def test_a_step_in_the_cut_is_the_same_step_in_the_patient():
    cut = cut_from([0.0, 0.0, 0.0], turn("X", 33.0), "ll")
    lps = to_lps(cut, [(0.0, 0.0), (5.0, 0.0)])
    assert np.linalg.norm(lps[1] - lps[0]) == pytest.approx(5.0)


def test_the_area_survives_the_trip_through_the_patient():
    cut = cut_from([9.0, 9.0, 9.0], turn("Y", 61.0) @ turn("Z", 14.0), "lr")
    points = circle(24, radius=6.0)
    assert polygon_area(from_lps(cut, to_lps(cut, points))) == pytest.approx(
        polygon_area(points)
    )


# --- same_plane -----------------------------------------------------------


def base() -> Cut:
    return cut_from([0.0, 0.0, 0.0], np.eye(3), "ul")


def test_a_cut_is_its_own_plane():
    assert same_plane(base(), base())


def test_a_millimetre_along_the_normal_is_another_plane():
    moved = cut_from(base().normal * 1.0, np.eye(3), "ul")
    assert not same_plane(base(), moved)


def test_a_step_within_the_plane_is_the_same_plane():
    moved = cut_from(np.array(base().right) * 50.0, np.eye(3), "ul")
    assert same_plane(base(), moved)


def test_a_roll_within_the_plane_is_the_same_plane():
    cut = base()
    rolled = cut_from([0.0, 0.0, 0.0], turn("Z", 30.0), "ul")
    assert same_plane(cut, rolled)


def test_a_degree_about_an_in_plane_axis_is_another_plane():
    assert not same_plane(base(), cut_from([0.0, 0.0, 0.0], turn("X", 1.0), "ul"))


def test_a_normal_turned_right_around_is_the_same_plane():
    cut = base()
    flipped = Cut(origin=cut.origin, right=cut.up, up=cut.right)
    assert np.allclose(flipped.normal, -cut.normal)
    assert same_plane(cut, flipped)


def test_the_tolerances_are_the_caller_s_to_tighten():
    moved = cut_from(base().normal * 1.0, np.eye(3), "ul")
    assert same_plane(base(), moved, depth=2.0)


# --- is_click -------------------------------------------------------------


def test_a_press_and_release_in_the_same_place_is_a_click():
    assert is_click((100.0, 50.0), (100.0, 50.0))


def test_a_press_that_travelled_is_not_a_click():
    assert not is_click((100.0, 50.0), (140.0, 50.0))


def test_a_shaky_hand_is_still_a_click():
    assert is_click((100.0, 50.0), (101.0, 51.0))
