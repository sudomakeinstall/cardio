"""The props a traced region is drawn with.

Small claims, but two of them are the kind that only show up as a picture that
is subtly wrong: a contour that votes on the camera's bounds reframes the cut
it was traced on, and one drawn in the plane of the image fights with it.
"""

# Third Party
import numpy as np
import pytest
import vtk

# Internal
from cardio.contour import CONTOUR_DEPTH, TRACED_COLOR, ContourActors

SQUARE = np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])


@pytest.fixture
def actors() -> ContourActors:
    return ContourActors()


def points_of(actor: vtk.vtkActor) -> np.ndarray:
    polydata = actor.GetMapper().GetInput()
    return np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])


def test_nothing_is_drawn_until_it_is_shown(actors):
    assert [prop.GetVisibility() for prop in actors.props] == [0, 0, 0]


def test_the_props_stay_out_of_the_camera_s_reckoning(actors):
    """A fit measures what is drawn, and a contour is not what a view is of."""
    assert [prop.GetUseBounds() for prop in actors.props] == [0, 0, 0]


def test_the_contour_is_lifted_clear_of_the_image(actors):
    """The image is a quad at zero and every camera here looks from +z."""
    actors.set_points(SQUARE, SQUARE, closed=True)

    assert CONTOUR_DEPTH > 0.0
    assert points_of(actors.line)[:, 2] == pytest.approx(CONTOUR_DEPTH)
    assert points_of(actors.marks)[:, 2] == pytest.approx(CONTOUR_DEPTH)


def test_the_contour_is_drawn_in_the_millimetres_it_was_given(actors):
    actors.set_points(SQUARE, SQUARE, closed=True)

    assert points_of(actors.line)[:, :2] == pytest.approx(SQUARE)


def test_a_closed_region_comes_back_round_to_its_first_point(actors):
    actors.set_points(SQUARE, SQUARE, closed=True)
    line = actors.line.GetMapper().GetInput().GetLines()

    ids = vtk.vtkIdList()
    line.InitTraversal()
    line.GetNextCell(ids)

    assert ids.GetNumberOfIds() == len(SQUARE) + 1
    assert ids.GetId(0) == ids.GetId(ids.GetNumberOfIds() - 1)


def test_a_region_still_being_traced_is_left_open(actors):
    actors.set_points(SQUARE, SQUARE, closed=False)
    line = actors.line.GetMapper().GetInput().GetLines()

    ids = vtk.vtkIdList()
    line.InitTraversal()
    line.GetNextCell(ids)

    assert ids.GetNumberOfIds() == len(SQUARE)


def test_the_handles_are_the_points_placed_rather_than_the_curve(actors):
    """They differ for a splined region, and it is the clicks a hand knows."""
    curve = np.repeat(SQUARE, 4, axis=0)
    actors.set_points(curve, SQUARE, closed=True)

    assert len(points_of(actors.line)) == len(curve)
    assert len(points_of(actors.marks)) == len(SQUARE)


def test_the_props_can_be_re_pointed_rather_than_rebuilt(actors):
    """What makes a cine cost a point array per frame instead of a pipeline."""
    mapper = actors.line.GetMapper()
    actors.set_points(SQUARE, SQUARE, closed=True)
    actors.set_points(SQUARE * 2.0, SQUARE * 2.0, closed=True)

    assert actors.line.GetMapper() is mapper
    assert points_of(actors.line)[:, :2] == pytest.approx(SQUARE * 2.0)


def test_a_label_is_only_drawn_once_there_is_one(actors):
    actors.set_points(SQUARE, SQUARE, closed=True)
    actors.set_visible(True)

    assert actors.line.GetVisibility()
    assert not actors.label.GetVisibility(), "no text yet"

    actors.set_label("16.0 mm²", (2.0, 2.0))
    actors.set_visible(True)

    assert actors.label.GetVisibility()


def test_a_colour_reaches_every_prop(actors):
    actors.set_color(TRACED_COLOR)

    assert actors.line.GetProperty().GetColor() == pytest.approx(TRACED_COLOR)
    assert actors.marks.GetProperty().GetColor() == pytest.approx(TRACED_COLOR)
    assert actors.label.GetTextProperty().GetColor() == pytest.approx(TRACED_COLOR)


def test_adding_the_props_twice_leaves_one_of_each(actors):
    """Which is what lets a redraw put them back without asking whether it must."""
    renderer = vtk.vtkRenderer()

    actors.add_to(renderer)
    actors.add_to(renderer)

    assert renderer.GetViewProps().GetNumberOfItems() == len(actors.props)


def test_the_props_can_be_taken_off_again(actors):
    renderer = vtk.vtkRenderer()
    actors.add_to(renderer)

    actors.remove_from(renderer)

    assert renderer.GetViewProps().GetNumberOfItems() == 0
