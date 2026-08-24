"""Test that MPRViews wires all three renderers the way the unrolled code did."""

# Third Party
import pytest
import vtk

# Internal
from cardio.mpr_views import MPRViews
from cardio.reslice import VIEWS, ResliceSet
from tests.test_reslice import make_image


def make_crosshairs() -> dict:
    """Two 2D line actors per view, shaped like Volume.create_crosshair_actors."""
    crosshairs = {}
    for view in VIEWS:
        crosshairs[view] = {
            name: {"actor": vtk.vtkActor2D()} for name in ("line1", "line2")
        }
    return crosshairs


def slices(background: float = -1000.0) -> ResliceSet:
    return ResliceSet(make_image(), interpolation="linear", background_level=background)


@pytest.fixture
def views() -> MPRViews:
    return MPRViews()


def prop_count(views: MPRViews, view: str) -> int:
    return views.renderer(view).GetViewProps().GetNumberOfItems()


def test_builds_one_window_per_orientation(views):
    assert set(views.windows) == set(VIEWS)
    for view in VIEWS:
        assert views[view].GetOffScreenRendering() == 1
        assert views.renderer(view) is not None


def test_set_image_adds_and_shows_the_actor_in_every_view(views):
    image = slices()
    views.set_image(image)

    for view in VIEWS:
        assert prop_count(views, view) == 1
        assert image[view]["actor"].GetVisibility() == 1


def test_clear_empties_every_renderer(views):
    views.set_image(slices())
    views.clear()

    for view in VIEWS:
        assert prop_count(views, view) == 0


def test_show_replaces_rather_than_accumulates(views):
    views.show(slices())
    views.show(slices())

    for view in VIEWS:
        assert prop_count(views, view) == 1


def test_show_adds_crosshairs_with_the_requested_visibility(views):
    crosshairs = make_crosshairs()
    views.show(slices(), crosshairs, crosshairs_visible=False)

    for view in VIEWS:
        # one image actor plus the two crosshair lines
        assert prop_count(views, view) == 3
        for line in crosshairs[view].values():
            assert line["actor"].GetVisibility() == 0


def test_show_tolerates_a_volume_with_no_crosshairs(views):
    views.show(slices(), {}, True)

    for view in VIEWS:
        assert prop_count(views, view) == 1


def test_overlays_stack_on_top_of_the_image(views):
    views.show(slices())
    overlay = slices(background=0.0)
    views.add_overlay(overlay)

    for view in VIEWS:
        assert prop_count(views, view) == 2
        assert overlay[view]["actor"].GetVisibility() == 1


def test_reset_cameras_reaches_every_view(views):
    views.set_image(slices())
    before = [views.renderer(v).GetActiveCamera().GetPosition() for v in VIEWS]

    views.reset_cameras()

    after = [views.renderer(v).GetActiveCamera().GetPosition() for v in VIEWS]
    assert before != after


def test_iterating_yields_the_view_names(views):
    assert set(iter(views)) == set(VIEWS)
