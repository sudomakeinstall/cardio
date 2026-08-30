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
    """A fresh set of windows, released again on the way out.

    Each one holds three render windows. Left to the garbage collector they
    accumulate for the length of the run, which is enough to wedge a machine
    whose GLX cannot service them.
    """
    windows = MPRViews()
    yield windows
    for view in windows:
        windows[view].Finalize()


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


def framed(views: MPRViews) -> MPRViews:
    """Views showing a frame, sized and fitted to it as the app has them.

    The size matters: a renderer only learns its viewport from the window, and
    projection is degenerate until it has one.
    """
    for view in VIEWS:
        views[view].SetSize(400, 300)
    views.show(slices())
    views.reset_cameras()
    return views


def test_world_per_pixel_is_positive_once_the_view_is_sized(views):
    assert framed(views).world_per_pixel("axial") > 0.0


def test_world_per_pixel_is_zero_before_the_window_is_sized(views):
    """Every display point projects to the same spot, so a pan must not run."""
    views.show(slices())
    views.reset_cameras()

    assert views.world_per_pixel("axial") == 0.0


def test_world_per_pixel_spans_the_fitted_image(views):
    """A fit puts the image across the viewport, so a pixel is a fraction of it."""
    framed(views)
    width = slices()["axial"]["actor"].GetBounds()[1] * 2

    assert 0.0 < views.world_per_pixel("axial") < width


def test_zooming_out_makes_each_pixel_cover_more_world(views):
    framed(views)
    before = views.world_per_pixel("axial")
    camera = views.renderer("axial").GetActiveCamera()
    camera.Dolly(0.5)

    assert views.world_per_pixel("axial") > before


def test_world_per_pixel_is_measured_per_view(views):
    """Each view frames its own extent, so the scales are independent."""
    framed(views)
    scales = {view: views.world_per_pixel(view) for view in VIEWS}

    assert all(scale > 0.0 for scale in scales.values())
    assert len(set(scales.values())) > 1


def test_zoom_moves_every_view(views):
    """All three share a zoom so the MPRs stay comparable."""
    framed(views)
    before = {view: views.world_per_pixel(view) for view in VIEWS}

    views.zoom(2.0)

    for view in VIEWS:
        assert views.world_per_pixel(view) < before[view]


def test_zoom_scales_every_view_by_the_same_factor(views):
    """Each keeps its own fit; only the factor is shared."""
    framed(views)
    before = {view: views.world_per_pixel(view) for view in VIEWS}

    views.zoom(2.0)

    ratios = [views.world_per_pixel(view) / before[view] for view in VIEWS]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_zooming_out_undoes_zooming_in(views):
    framed(views)
    before = views.world_per_pixel("axial")

    views.zoom(1.5)
    views.zoom(1 / 1.5)

    assert views.world_per_pixel("axial") == pytest.approx(before)


def test_zoom_leaves_the_focal_point_alone(views):
    """The origin sits at the focal point, so the crosshair must not drift."""
    framed(views)
    camera = views.renderer("axial").GetActiveCamera()
    before = camera.GetFocalPoint()

    views.zoom(2.0)

    assert camera.GetFocalPoint() == pytest.approx(before)


@pytest.mark.parametrize("factor", [0.0, -1.0])
def test_a_degenerate_zoom_is_ignored(views, factor):
    framed(views)
    before = views.world_per_pixel("axial")

    views.zoom(factor)

    assert views.world_per_pixel("axial") == pytest.approx(before)
