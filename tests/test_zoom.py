"""Fitting the MPR views to a set of labels.

The camera looks at the reslice origin and never moves off it, so the fit frames
the labels by sliding the origin onto the middle of their shadow and then sizing
the viewport to the half-span from there. The slide runs along the fitted plane's
own two axes, which leaves that plane cutting where it did. What is checked here
is that arithmetic, and that the cloud it measures really does cover the voxels
-- at every frame, not only the one on screen.
"""

# Third Party
import numpy as np
import pytest

# Internal
from cardio.camera import fit_factor
from cardio.reslice import VIEW_TRANSFORMS
from cardio.segmentation import (
    index_to_world,
    label_mask,
    plane_shadow,
    trimmed_bounds,
    voxel_corner_cloud,
    voxel_shell,
)
from tests.phantoms import (
    MOVING_FRAMES,
    make_image,
    moving_stack_segmentation,
    rotation_about_z,
    stacked_array,
    to_world,
)
from tests.test_app_smoke import build_ready

# ------------------------------------------------------------ the arithmetic ---


def test_a_fit_fills_the_dimension_the_box_is_tightest_against():
    # A square box in a landscape viewport: the height runs out first.
    factor = fit_factor((10.0, 10.0), (400, 200), 1.0, 1.0)

    assert factor == pytest.approx(10.0)


def test_a_wide_box_is_limited_by_the_width():
    factor = fit_factor((50.0, 10.0), (400, 200), 1.0, 1.0)

    assert factor == pytest.approx(4.0)


def test_the_fill_fraction_leaves_the_margin_it_names():
    full = fit_factor((10.0, 10.0), (400, 200), 1.0, 1.0)
    eighty = fit_factor((10.0, 10.0), (400, 200), 1.0, 0.8)

    assert eighty == pytest.approx(0.8 * full)


def test_a_finer_scale_asks_for_more_zoom():
    coarse = fit_factor((10.0, 10.0), (400, 200), 1.0, 1.0)
    fine = fit_factor((10.0, 10.0), (400, 200), 0.5, 1.0)

    assert fine == pytest.approx(coarse / 2.0)


def test_a_box_with_no_extent_asks_for_no_zoom_at_all():
    assert fit_factor((0.0, 0.0), (400, 200), 1.0, 1.0) is None


def test_a_flat_box_is_fitted_by_the_direction_it_does_reach_in():
    # A single slab of voxels seen edge-on still has a width to fit.
    assert fit_factor((10.0, 0.0), (400, 200), 1.0, 1.0) == pytest.approx(20.0)


def test_a_window_that_was_never_sized_cannot_be_fitted():
    assert fit_factor((10.0, 10.0), (0, 0), 0.0, 1.0) is None
    assert fit_factor((10.0, 10.0), (400, 200), 0.0, 1.0) is None


# -------------------------------------------------------------- the shadow ---


def test_a_lopsided_cloud_is_a_centre_offset_from_the_origin():
    """What the fit moves the origin by, and what it then measures against."""
    cloud = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])

    centre, half_span = plane_shadow(cloud, np.eye(3), np.zeros(3))

    assert centre == pytest.approx((15.0, 0.0))
    assert half_span == pytest.approx((5.0, 0.0))


def test_the_extent_is_taken_in_the_plane_s_own_axes():
    """A point on the axial normal casts no shadow on the axial plane."""
    frame = VIEW_TRANSFORMS["axial"]
    normal = frame[:, 2]

    centre, half_span = plane_shadow(np.array([normal * 30.0]), frame, np.zeros(3))

    assert centre == pytest.approx((0.0, 0.0))
    assert half_span == pytest.approx((0.0, 0.0))


def test_an_untrimmed_shadow_covers_every_point():
    cloud = np.array([[0.0, 0.0, 0.0]] * 999 + [[100.0, 0.0, 0.0]])
    _, half_span = plane_shadow(cloud, np.eye(3), np.zeros(3))
    assert half_span[0] == pytest.approx(50.0)


def test_trimming_drops_the_tail_that_sets_the_extent():
    """A stray voxel is a vanishing share of a cloud and yet its whole extent."""
    cloud = np.array([[0.0, 0.0, 0.0]] * 999 + [[100.0, 0.0, 0.0]])
    _, half_span = plane_shadow(cloud, np.eye(3), np.zeros(3), percentile=99.0)
    assert half_span[0] == pytest.approx(0.0, abs=1e-9)


def test_trimming_is_off_until_it_is_asked_for():
    projected = np.array([[0.0], [1.0], [100.0]])
    low, high = trimmed_bounds(projected)
    assert (low, high) == (pytest.approx([0.0]), pytest.approx([100.0]))


def test_a_trim_takes_the_same_share_off_each_end():
    projected = np.array([[float(value)] for value in range(101)])
    low, high = trimmed_bounds(projected, 98.0)
    assert low == pytest.approx([1.0])
    assert high == pytest.approx([99.0])


def test_a_turned_plane_sees_a_turned_shadow():
    frame = rotation_about_z(45.0) @ VIEW_TRANSFORMS["axial"]
    cloud = np.array([[10.0, 10.0, 0.0]])

    turned, _ = plane_shadow(cloud, frame, np.zeros(3))
    square, _ = plane_shadow(cloud, VIEW_TRANSFORMS["axial"], np.zeros(3))

    assert max(abs(turned)) == pytest.approx(np.hypot(10.0, 10.0))
    assert max(abs(square)) == pytest.approx(10.0)


def test_an_empty_cloud_has_no_extent():
    assert plane_shadow(None, np.eye(3), np.zeros(3)) is None
    assert plane_shadow(np.zeros((0, 3)), np.eye(3), np.zeros(3)) is None


# --------------------------------------------------------------- the cloud ---


def _fill(image, array):
    """Write a ``(k, j, i)`` label array into an allocated image."""
    scalars = image.GetPointData().GetScalars()
    for index, value in enumerate(array.ravel()):
        scalars.SetTuple1(index, int(value))
    return image


def _solid_block():
    """A 6x6x6 image whose middle 4x4x4 carries label 1."""
    image = make_image(dims=(6, 6, 6), spacing=(1.0, 1.0, 1.0))
    array = np.zeros((6, 6, 6), dtype=np.uint8)
    array[1:5, 1:5, 1:5] = 1
    return _fill(image, array), array


def test_the_shell_drops_what_is_walled_in_on_every_side():
    _, array = _solid_block()
    mask = array.astype(bool)

    shell = voxel_shell(mask)

    # A 4-cubed block keeps its 56 surface voxels and loses its 8-cubed core.
    assert mask.sum() == 64
    assert shell.sum() == 56


def test_the_shell_reaches_exactly_as_far_as_the_solid_did():
    _, array = _solid_block()
    mask = array.astype(bool)

    solid = np.array(np.nonzero(mask))
    shell = np.array(np.nonzero(voxel_shell(mask)))

    assert (shell.min(axis=1) == solid.min(axis=1)).all()
    assert (shell.max(axis=1) == solid.max(axis=1)).all()


def test_the_cloud_covers_the_voxels_rather_than_their_centres():
    image, _ = _solid_block()

    cloud = voxel_corner_cloud(image, [1])

    # Centres run 1..4 in each axis; the corners reach half a voxel past both.
    assert cloud.min(axis=0) == pytest.approx([0.5, 0.5, 0.5])
    assert cloud.max(axis=0) == pytest.approx([4.5, 4.5, 4.5])


def test_the_cloud_is_carried_by_the_image_s_own_geometry():
    spacing, origin, direction = (
        (2.0, 3.0, 4.0),
        (-5.0, 7.0, 11.0),
        rotation_about_z(30.0),
    )
    image = make_image(dims=(6, 6, 6), spacing=spacing, direction=direction)
    image.SetOrigin(*origin)
    array = np.zeros((6, 6, 6), dtype=np.uint8)
    array[1:5, 1:5, 1:5] = 1
    _fill(image, array)

    cloud = voxel_corner_cloud(image, [1])

    corner = to_world((0.5, 0.5, 0.5), spacing, origin, direction)
    assert np.isclose(cloud, corner).all(axis=1).any()


def test_an_absent_label_has_no_cloud():
    image, _ = _solid_block()

    assert voxel_corner_cloud(image, [7]) is None
    assert voxel_corner_cloud(image, []) is None


def test_the_index_affine_agrees_with_vtk_s_own():
    image, _ = _solid_block()
    image.SetOrigin(-5.0, 7.0, 11.0)

    matrix, origin = index_to_world(image)

    expected = [0.0, 0.0, 0.0]
    image.TransformContinuousIndexToPhysicalPoint(1.5, 2.5, 3.5, expected)
    assert matrix @ np.array([1.5, 2.5, 3.5]) + origin == pytest.approx(expected)


# ------------------------------------------------------------ across frames ---


def test_the_cloud_spans_every_frame_of_a_moving_label(tmp_path):
    """The stack travels two voxels per frame, and the fit has to hold for all."""
    seg = moving_stack_segmentation(tmp_path)

    whole = seg.label_cloud([1])
    first = voxel_corner_cloud(seg._label_images[0], [1])
    last = voxel_corner_cloud(seg._label_images[MOVING_FRAMES - 1], [1])

    assert whole.min(axis=0) == pytest.approx(np.minimum(first.min(0), last.min(0)))
    assert whole.max(axis=0) == pytest.approx(np.maximum(first.max(0), last.max(0)))
    assert whole.max(axis=0)[0] > first.max(axis=0)[0]


def test_the_cloud_is_read_once_and_remembered(tmp_path):
    seg = moving_stack_segmentation(tmp_path)

    first = seg.label_cloud([1])

    assert seg.label_cloud([1]) is first
    assert seg.label_cloud([1, 1]) is first


def test_a_label_the_series_never_carries_has_no_cloud(tmp_path):
    seg = moving_stack_segmentation(tmp_path)

    assert seg.label_cloud([9]) is None


def test_the_mask_names_the_voxels_the_array_put_there(tmp_path):
    seg = moving_stack_segmentation(tmp_path)

    mask = label_mask(seg._label_images[0], [1])

    assert mask.sum() == int((stacked_array(30.0) == 1).sum())


# ---------------------------------------------------------------- the lock ---


def locked_app(tmp_path, **overrides):
    """A built app whose views are sized, holding a fit on label 1."""
    server, scene, logic, _ = build_ready(tmp_path, zoom={"labels": [1], **overrides})
    for name in scene.mpr_views:
        scene.mpr_views[name].SetSize(600, 400)
        scene.mpr_views[name].Render()
    scene.mpr_views.reset_cameras()
    return server, scene, logic


def scales(scene):
    return [scene.mpr_views.world_per_pixel(view) for view in scene.mpr_views]


def reach(logic):
    """The labels' farthest edge from the origin, in the plane being fitted."""
    centre, half_span, _ = logic.zoom.shadow()
    return np.abs(centre) + half_span


def filled(logic, scene):
    """The share of the fitted viewport the labels take up in each direction."""
    views = scene.mpr_views
    plane = logic.server.state.zoom_plane
    width, height = views.renderer(plane).GetSize()
    per_pixel = views.world_per_pixel(plane)
    edge = reach(logic)
    return edge[0] / (per_pixel * width / 2), edge[1] / (per_pixel * height / 2)


def test_the_fit_brings_the_labels_to_the_share_of_the_viewport_asked_for(tmp_path):
    _, scene, logic = locked_app(tmp_path, fill=80)

    logic.dispatch("zoom_to_labels")

    share = filled(logic, scene)
    assert max(share) == pytest.approx(0.80)
    assert max(share) >= min(share)


def test_the_fit_slides_the_crosshair_onto_the_middle_of_the_shadow(tmp_path):
    """The whole reason the fit is measured against the half-span."""
    _, scene, logic = locked_app(tmp_path)
    assert max(abs(value) for value in logic.zoom.shadow()[0]) > 0.0

    logic.dispatch("zoom_to_labels")

    assert logic.zoom.shadow()[0] == pytest.approx((0.0, 0.0), abs=1e-9)
    # The camera is what stays put: it looks at the reslice frame's own origin,
    # and the origin moving is the picture sliding under it.
    for view in scene.mpr_views:
        camera = scene.mpr_views.renderer(view).GetActiveCamera()
        assert camera.GetFocalPoint() == pytest.approx((0.0, 0.0, 0.0))


def test_the_slide_stays_in_the_plane_being_fitted(tmp_path):
    """The fitted plane goes on cutting where it did; only the crosshair moves."""
    server, _, logic = locked_app(tmp_path)
    convention = logic.zoom.convention
    before = np.array(convention.point_to_itk(server.state.mpr_origin))
    normal = logic.zoom.shadow()[2][:, 2]

    logic.dispatch("zoom_to_labels")

    after = np.array(convention.point_to_itk(server.state.mpr_origin))
    assert np.linalg.norm(after - before) > 0.0
    assert (after - before) @ normal == pytest.approx(0.0, abs=1e-9)


def test_fitting_a_second_time_changes_nothing(tmp_path):
    """The slide is onto a centre the fit has already reached."""
    server, scene, logic = locked_app(tmp_path)
    logic.dispatch("zoom_to_labels")
    settled, left = list(server.state.mpr_origin), scales(scene)

    logic.dispatch("zoom_to_labels")

    assert server.state.mpr_origin == pytest.approx(settled)
    assert scales(scene) == pytest.approx(left)


def test_all_three_views_move_by_the_one_factor(tmp_path):
    """Each keeps its own fit; only the factor is shared, as a drag shares it."""
    _, scene, logic = locked_app(tmp_path)
    before = scales(scene)

    logic.dispatch("zoom_to_labels")

    factors = [b / a for b, a in zip(before, scales(scene))]
    assert factors[0] != pytest.approx(1.0)
    assert factors == pytest.approx([factors[0]] * 3)


def test_a_held_fit_pulls_the_origin_back_onto_the_centre(tmp_path):
    """The lock owns the framing, so an origin moved under it is moved back.

    The scale does not have to change for that: the labels span what they span
    however far the crosshair has wandered off their middle.
    """
    server, scene, logic = locked_app(tmp_path)
    server.state.zoom_locked = True
    server.state.flush()
    convention = logic.zoom.convention
    held = scales(scene)
    settled = np.array(convention.point_to_itk(server.state.mpr_origin))
    frame = logic.zoom.shadow()[2]

    server.state.mpr_origin = [value + 4.0 for value in server.state.mpr_origin]
    server.state.flush()

    moved = np.array(convention.point_to_itk(server.state.mpr_origin)) - settled
    # Back onto the centre along the plane's own axes; the third axis is left
    # where it was put, which is what a snap lock underneath still owns.
    assert moved @ frame[:, 0] == pytest.approx(0.0, abs=1e-9)
    assert moved @ frame[:, 1] == pytest.approx(0.0, abs=1e-9)
    assert moved @ frame[:, 2] != pytest.approx(0.0)
    assert scales(scene) == pytest.approx(held)
    assert logic.zoom.shadow()[0] == pytest.approx((0.0, 0.0), abs=1e-9)


def test_a_fit_that_is_not_held_stays_where_the_user_left_it(tmp_path):
    server, scene, logic = locked_app(tmp_path)
    logic.dispatch("zoom_to_labels")
    left = scales(scene)

    server.state.mpr_origin = [value + 4.0 for value in server.state.mpr_origin]
    server.state.flush()

    assert scales(scene) == pytest.approx(left)


def test_a_held_fit_survives_a_turn(tmp_path):
    """A rotation re-cuts the plane, so the shadow it casts is a different one."""
    server, scene, logic = locked_app(tmp_path)
    server.state.zoom_locked = True
    server.state.flush()

    logic.dispatch("add_rotation", axis="X")
    # Written whole rather than edited in place: the sequence is one value, and
    # a mutation inside it is not a change trame can see.
    data = {**server.state.mpr_rotation_data}
    data["angles_list"] = [{**step, "angle": 0.7} for step in data["angles_list"]]
    server.state.mpr_rotation_data = data
    server.state.flush()

    assert max(filled(logic, scene)) == pytest.approx(0.80)


def test_a_configured_lock_fits_once_the_views_are_real(tmp_path):
    """Seeding happens before the MPR windows exist, so the fit has to wait."""
    server, scene, logic = locked_app(tmp_path, locked=True)

    assert server.state.zoom_locked is True

    logic.zoom.refit()

    assert max(filled(logic, scene)) == pytest.approx(0.80)


def test_nothing_is_fitted_to_a_selection_that_is_empty(tmp_path):
    server, scene, logic = locked_app(tmp_path)
    server.state.zoom_labels = []
    left = scales(scene)

    logic.dispatch("zoom_to_labels")

    assert logic.zoom.shadow() is None
    assert scales(scene) == pytest.approx(left)


def test_the_labels_survive_the_startup_that_seeded_them(tmp_path):
    """The segmentation listener fires on the first flush and must clear nothing."""
    server, _, _ = locked_app(tmp_path)

    assert server.state.zoom_labels == [1]


def test_changing_segmentation_drops_labels_that_indexed_into_the_old_one(tmp_path):
    """The labels number the file they came from, so they cannot follow it."""
    server, _, logic = locked_app(tmp_path)

    # A name no segmentation carries leaves the picker and the labels alone.
    logic.zoom._on_segmentation_changed("mesh")
    assert server.state.zoom_labels == [1]

    logic.zoom._published_seg_label = "another"
    logic.zoom._on_segmentation_changed("seg")
    assert server.state.zoom_labels == []
