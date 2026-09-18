"""Test that MPRViews wires all three renderers the way the unrolled code did."""

# System
import math

# Third Party
import pytest
import vtk

# Internal
from cardio.camera import cut_point, visible_rectangle
from cardio.mpr_views import MPRViews
from cardio.reslice import VIEWS, ResliceSet
from tests.phantoms import make_image, rotation_about_z


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
    assert framed(views).world_per_pixel("ul") > 0.0


def test_world_per_pixel_is_zero_before_the_window_is_sized(views):
    """Every display point projects to the same spot, so a pan must not run."""
    views.show(slices())
    views.reset_cameras()

    assert views.world_per_pixel("ul") == 0.0


def test_world_per_pixel_spans_the_fitted_image(views):
    """A fit puts the image across the viewport, so a pixel is a fraction of it."""
    framed(views)
    width = slices()["ul"]["actor"].GetBounds()[1] * 2

    assert 0.0 < views.world_per_pixel("ul") < width


def test_zooming_out_makes_each_pixel_cover_more_world(views):
    framed(views)
    before = views.world_per_pixel("ul")
    camera = views.renderer("ul").GetActiveCamera()
    camera.Dolly(0.5)

    assert views.world_per_pixel("ul") > before


def test_world_per_pixel_is_measured_per_view(views):
    """Each view frames its own extent, so the scales are independent."""
    framed(views)
    scales = {view: views.world_per_pixel(view) for view in VIEWS}

    assert all(scale > 0.0 for scale in scales.values())
    assert len(set(scales.values())) > 1


# What the view is showing, which a data capture is cropped to


def test_the_visible_rectangle_spans_the_whole_viewport(views):
    """Edge to edge, in world units: what a capture has to cover for the
    picture and the data behind it to be framed alike."""
    framed(views)
    renderer = views.renderer("ul")
    width, height = renderer.GetSize()
    scale = views.world_per_pixel("ul")

    (low_x, high_x), (low_y, high_y) = visible_rectangle(renderer)

    assert high_x - low_x == pytest.approx(scale * width, rel=1e-3)
    assert high_y - low_y == pytest.approx(scale * height, rel=1e-3)


def test_the_visible_rectangle_is_centred_on_what_the_view_looks_at(views):
    """The cameras are never moved off the cut's origin, so the rectangle is
    centred on it and a crop about it keeps the crosshair in the middle."""
    framed(views)
    renderer = views.renderer("ul")
    focus = renderer.GetActiveCamera().GetFocalPoint()

    (low_x, high_x), (low_y, high_y) = visible_rectangle(renderer)

    assert (low_x + high_x) / 2 == pytest.approx(focus[0], abs=1e-6)
    assert (low_y + high_y) / 2 == pytest.approx(focus[1], abs=1e-6)


def test_zooming_in_narrows_the_visible_rectangle(views):
    framed(views)
    renderer = views.renderer("ul")
    (low, high), _ = visible_rectangle(renderer)

    views.zoom(2.0)

    (zoomed_low, zoomed_high), _ = visible_rectangle(renderer)
    assert zoomed_high - zoomed_low < high - low


def test_there_is_no_visible_rectangle_before_the_window_is_sized(views):
    """Every display point projects onto the same spot, so there is nothing to
    crop to and the capture is written whole."""
    views.show(slices())
    views.reset_cameras()

    assert visible_rectangle(views.renderer("ul")) is None


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
    before = views.world_per_pixel("ul")

    views.zoom(1.5)
    views.zoom(1 / 1.5)

    assert views.world_per_pixel("ul") == pytest.approx(before)


def test_zoom_leaves_the_focal_point_alone(views):
    """The origin sits at the focal point, so the crosshair must not drift."""
    framed(views)
    camera = views.renderer("ul").GetActiveCamera()
    before = camera.GetFocalPoint()

    views.zoom(2.0)

    assert camera.GetFocalPoint() == pytest.approx(before)


@pytest.mark.parametrize("factor", [0.0, -1.0])
def test_a_degenerate_zoom_is_ignored(views, factor):
    framed(views)
    before = views.world_per_pixel("ul")

    views.zoom(factor)

    assert views.world_per_pixel("ul") == pytest.approx(before)


# The fit


def posed_away_from_the_centre(image) -> ResliceSet:
    """Cuts aimed somewhere other than the middle of their own image.

    Which is the case the fit used to get wrong: the auto-cropped extent is
    centred on the image, so it names the posed origin only by coincidence.
    """
    cuts = ResliceSet(image, interpolation="linear", background_level=-1000.0)
    cuts.set_pose(
        [value + shift for value, shift in zip(image.GetCenter(), (7.0, -5.0, 11.0))]
    )
    return cuts


def half_height(renderer) -> float:
    """What the camera covers above and below the focal point, in world units."""
    camera = renderer.GetActiveCamera()
    focal, position = camera.GetFocalPoint(), camera.GetPosition()
    distance = math.dist(focal, position)
    return distance * math.tan(math.radians(camera.GetViewAngle() / 2.0))


def half_span(renderer) -> float:
    """How far what is drawn reaches from the origin, along the view's y."""
    bounds = renderer.ComputeVisiblePropBounds()
    return max(abs(bounds[2]), abs(bounds[3]))


def test_reset_cameras_looks_at_the_reslice_origin(views):
    """Every view centres on the point the cuts were posed on.

    The origin is always the reslice output's (0, 0, 0), which is the whole
    reason the crosshairs can be drawn at the middle of the viewport.
    """
    views.show(posed_away_from_the_centre(make_image()), reset_camera=True)

    for view in VIEWS:
        camera = views.renderer(view).GetActiveCamera()
        assert camera.GetFocalPoint() == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_reset_cameras_centres_an_oblique_image(views):
    """An obliquely acquired image is centred too, not merely nearly.

    Its auto-cropped extent is rounded out to whole voxels asymmetrically, so
    fitting to the extent leaves a fraction of a millimetre behind -- small
    enough to miss until the view is zoomed in on something small.
    """
    image = make_image(direction=rotation_about_z(25.0))
    views.show(posed_away_from_the_centre(image), reset_camera=True)

    for view in VIEWS:
        camera = views.renderer(view).GetActiveCamera()
        assert camera.GetFocalPoint() == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_reset_cameras_still_shows_the_whole_cut(views):
    """Centring on the origin widens the fit rather than cropping it."""
    views.show(posed_away_from_the_centre(make_image()), reset_camera=True)

    for view in VIEWS:
        renderer = views.renderer(view)
        assert half_height(renderer) >= half_span(renderer) - 1e-9


# --- a click on a cut ---------------------------------------------------------

# ``cut_point`` is the one place a click becomes a measurement.  What is checked
# here is the claim it rests on: a cut renderer's world coordinates are the
# cut's own millimetres, so the centre of the viewport is the point the cut was
# posed at and one pixel is worth ``world_per_pixel`` of them.


def test_the_centre_of_the_view_is_the_point_the_cut_was_posed_at(views):
    framed(views)
    renderer = views.renderer("ul")
    width, height = renderer.GetSize()

    assert cut_point(renderer, width / 2.0, height / 2.0) == pytest.approx(
        (0.0, 0.0), abs=1e-6
    )


def test_a_pixel_of_the_view_is_worth_one_world_unit_of_the_cut(views):
    framed(views)
    renderer = views.renderer("ul")
    scale = views.world_per_pixel("ul")

    here = cut_point(renderer, 200.0, 150.0)
    right = cut_point(renderer, 201.0, 150.0)
    up = cut_point(renderer, 200.0, 151.0)

    assert right[0] - here[0] == pytest.approx(scale, rel=1e-3)
    assert right[1] == pytest.approx(here[1], abs=1e-6)
    assert up[1] - here[1] == pytest.approx(scale, rel=1e-3)
    assert up[0] == pytest.approx(here[0], abs=1e-6)


def test_the_cut_reads_right_and_up_the_way_the_display_does(views):
    """A click further right and further up is further along both axes.

    Display coordinates arrive with y running up, which is VTK's own
    convention, so nothing flips one on the way in -- and a contour traced from
    clicks that had been quietly mirrored would enclose the right area in the
    wrong place.
    """
    framed(views)
    renderer = views.renderer("ul")

    low = cut_point(renderer, 120.0, 90.0)
    high = cut_point(renderer, 280.0, 210.0)

    assert high[0] > low[0]
    assert high[1] > low[1]


def test_the_corners_of_the_view_are_the_rectangle_it_shows(views):
    """The same two corners ``visible_rectangle`` reports, reached one click at
    a time -- so a region traced to the edge of a view is a region a capture of
    that view covers."""
    framed(views)
    renderer = views.renderer("ul")
    width, height = renderer.GetSize()
    (low_x, high_x), (low_y, high_y) = visible_rectangle(renderer)

    assert cut_point(renderer, 0.0, 0.0) == pytest.approx((low_x, low_y), abs=1e-6)
    assert cut_point(renderer, float(width), float(height)) == pytest.approx(
        (high_x, high_y), abs=1e-6
    )


def test_a_click_on_a_zoomed_view_is_measured_at_the_zoomed_scale(views):
    """The scale is read through the camera, so a zoom needs nothing told to it."""
    framed(views)
    renderer = views.renderer("ul")
    before = cut_point(renderer, 250.0, 150.0)

    views.zoom(2.0)

    assert cut_point(renderer, 250.0, 150.0)[0] == pytest.approx(
        before[0] / 2.0, rel=1e-2
    )
