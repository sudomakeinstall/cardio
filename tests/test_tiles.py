"""Test the tile grid: several cuts sampled along the traverse path."""

import numpy as np
import pytest
import vtk

from cardio.logic.snap import ALIGN_STEP_NAME, alignment_rotation
from cardio.logic.tiles import poses_along, sample_fractions
from cardio.orientation import (
    euler_angle_to_rotation_matrix,
    minimal_rotation,
    quaternion_to_rotation_matrix,
)
from cardio.reslice import VIEW_TRANSFORMS
from cardio.rotation import RotationStep
from cardio.state import ObjectState
from tests.fakes import FakeApp, FakeScene, align_at, snap_state, traverse_logic
from tests.geometry import angle_between, matrix_array, tilted_plane
from tests.phantoms import stacked_segmentation

# Sampling


@pytest.mark.parametrize("count", [2, 3, 9, 12, 36])
def test_samples_span_the_whole_path(count):
    fractions = sample_fractions(count)
    assert len(fractions) == count
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0


def test_samples_are_evenly_spaced():
    assert sample_fractions(9) == pytest.approx(
        [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    )


def test_a_single_tile_sits_in_the_middle():
    """Neither endpoint is a fairer choice than the other, so take neither."""
    assert sample_fractions(1) == [0.5]


def test_no_tiles_yields_no_samples():
    assert sample_fractions(0) == []


def test_poses_along_returns_one_pose_per_tile():
    def pose_at(fraction):
        return [fraction, 0.0, 0.0], np.eye(3)

    poses = poses_along(pose_at, 4)
    assert [origin[0] for origin, _ in poses] == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_poses_along_gives_up_on_the_first_gap():
    def pose_at(fraction):
        return None if fraction > 0.5 else ([0.0, 0.0, 0.0], np.eye(3))

    assert poses_along(pose_at, 5) is None


# Alignment arithmetic


def test_alignment_rotation_matches_the_published_quaternion(tmp_path):
    """The tiles compose their own rotation; it must agree with Align's."""
    logic = traverse_logic(stacked_segmentation(tmp_path))
    align_at(logic, 40)

    step = logic.server.state.mpr_rotation_data["angles_list"][0]
    assert step["name"] == ALIGN_STEP_NAME

    _, axes = logic.snap.traverse_pose(0, 0.4)
    assert quaternion_to_rotation_matrix(step["quaternion"]) == pytest.approx(
        alignment_rotation(axes), abs=1e-9
    )


def test_alignment_rotation_carries_the_basis_onto_the_axial_view():
    axes = tilted_plane(25)
    rotation = alignment_rotation(axes)
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


# Poses along the traverse path


def test_traverse_pose_reproduces_both_interfaces(tmp_path):
    logic = traverse_logic(stacked_segmentation(tmp_path))

    _, start_axes = logic.snap.traverse_pose(0, 0.0)
    _, end_axes = logic.snap.traverse_pose(0, 1.0)

    assert start_axes == pytest.approx(logic.snap._interface_plane(0, 0)[1], abs=1e-12)
    assert end_axes == pytest.approx(logic.snap._interface_plane(0, 1)[1], abs=1e-12)


def test_traverse_pose_halves_the_tilt_at_the_midpoint(tmp_path):
    logic = traverse_logic(stacked_segmentation(tmp_path))

    _, start_axes = logic.snap.traverse_pose(0, 0.0)
    _, middle_axes = logic.snap.traverse_pose(0, 0.5)
    _, end_axes = logic.snap.traverse_pose(0, 1.0)

    tilt = angle_between(start_axes[:, 2], end_axes[:, 2])
    assert tilt > 20.0
    assert angle_between(middle_axes[:, 2], start_axes[:, 2]) == pytest.approx(
        tilt / 2, abs=1e-6
    )


def test_traverse_pose_origins_walk_in_a_straight_line(tmp_path):
    logic = traverse_logic(stacked_segmentation(tmp_path))

    origins = np.array([logic.snap.traverse_pose(0, f)[0] for f in sample_fractions(5)])
    steps = np.diff(origins, axis=0)
    assert steps == pytest.approx(np.repeat(steps[:1], len(steps), axis=0), abs=1e-9)


def test_consecutive_tiles_only_tilt(tmp_path):
    """A spin between neighbouring tiles would make the grid unreadable."""
    logic = traverse_logic(stacked_segmentation(tmp_path))
    bases = [logic.snap.traverse_pose(0, f)[1] for f in sample_fractions(6)]

    for before, after in zip(bases, bases[1:]):
        carried = minimal_rotation(before[:, 2], after[:, 2]) @ before
        assert after == pytest.approx(carried, abs=1e-9)


def test_traverse_pose_needs_a_complete_selection(tmp_path):
    logic = traverse_logic(stacked_segmentation(tmp_path), snap_labels_c=[])
    assert logic.snap.traverse_pose(0, 0.5) is None


def test_traverse_pose_leaves_the_slider_alone(tmp_path):
    """Sampling the path must not disturb the pose the quad view is showing."""
    logic = traverse_logic(stacked_segmentation(tmp_path))
    align_at(logic, 25)
    origin = list(logic.server.state.mpr_origin)

    for fraction in sample_fractions(9):
        logic.snap.traverse_pose(0, fraction)

    assert logic.server.state.snap_traverse == 25
    assert logic.server.state.mpr_origin == origin


# The grid


class FakeVolume:
    """A volume the tile pipelines can reslice, and nothing more."""

    kind = "volume"

    def __init__(self, label="vol"):
        self.label = label
        source = vtk.vtkImageSinusoidSource()
        source.SetWholeExtent(0, 31, 0, 31, 0, 31)
        source.Update()
        self._image = source.GetOutput()

    def mpr_image_data(self, frame: int = 0):
        return self._image


def tiled(segmentation, **overrides) -> FakeApp:
    """A FakeApp whose tile grid is built and on screen."""
    state = snap_state(
        snap_mode="traverse",
        snap_labels_c=[3],
        mpr_window=800.0,
        mpr_level=200.0,
        mpr_segmentation_opacity=0.7,
        maximized_view="tile",
        active_volume_label="vol",
        tile_rows=3,
        tile_cols=3,
    )
    # MPRController.register seeds these in the real app; the fake has no register
    state[ObjectState.of(segmentation).mpr_overlay] = False
    state.update(overrides)
    scene = FakeScene([segmentation], volumes=[FakeVolume()])
    scene.setup_tile_render_window()
    return FakeApp(scene, **state)


def posed_matrices(logic: FakeApp) -> list[np.ndarray]:
    key = ("volume:vol", logic.server.state.frame)
    tiles = logic.tiles._tile_sets[key]
    return [matrix_array(parts["reslice"].GetResliceAxes()) for parts in tiles.values()]


def test_the_grid_fills_with_one_tile_per_cell(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)

    assert len(logic.scene.tile_views) == 9
    assert len(posed_matrices(logic)) == 9
    assert not logic.server.state.snap_no_interface


def test_changing_the_grid_resizes_both_the_tiles_and_the_renderers(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)

    logic.server.state.tile_rows = 4
    logic.server.state.tile_cols = 2
    logic.tiles.update_tiles(0)

    assert len(logic.scene.tile_views) == 8
    assert len(posed_matrices(logic)) == 8


def test_the_grid_is_capped(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path), tile_rows=9, tile_cols=9)
    assert logic.tiles.tile_count == 36


def test_the_first_and_last_tile_sit_on_the_two_interfaces(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)
    matrices = posed_matrices(logic)

    for tile, interface in ((0, 0), (-1, 1)):
        expected = logic.snap._interface_plane(0, interface)[1]
        assert matrices[tile][:3, 2] == pytest.approx(expected[:, 2], abs=1e-9)


def test_tiles_walk_the_path_in_order(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)
    origins = np.array([m[:3, 3] for m in posed_matrices(logic)])

    direction = origins[-1] - origins[0]
    projections = origins @ direction
    assert np.all(np.diff(projections) > 0)


def test_a_tile_matches_the_quad_view_at_the_same_fraction(tmp_path):
    """The tile grid must show exactly what scrubbing the slider would show."""
    logic = tiled(stacked_segmentation(tmp_path), tile_rows=1, tile_cols=5)
    logic.tiles.update_tiles(0)
    tiles = posed_matrices(logic)

    for tile, percent in enumerate((0, 25, 50, 75, 100)):
        align_at(logic, percent)
        rotation = logic.rotations.rotation_matrix()
        origin = logic.mpr.convention.point_to_itk(logic.server.state.mpr_origin)
        expected = np.eye(4)
        expected[:3, :3] = rotation @ VIEW_TRANSFORMS["axial"]
        expected[:3, 3] = origin
        assert tiles[tile] == pytest.approx(expected, abs=1e-9)


def test_a_user_rotation_carries_into_every_tile(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)
    before = posed_matrices(logic)

    logic.rotations.edit_steps(
        lambda steps: [*steps, RotationStep(axis="Z", angle=20.0)]
    )
    logic.tiles.update_tiles(0)
    after = posed_matrices(logic)

    # The reslice matrix is cumulative @ T_axial, so compare the cumulative
    # rotations rather than the matrices, which carry the view transform too.
    spin = euler_angle_to_rotation_matrix("Z", 20.0)
    axial = VIEW_TRANSFORMS["axial"]
    for was, now in zip(before, after):
        assert now[:3, :3] @ axial.T == pytest.approx(
            was[:3, :3] @ axial.T @ spin, abs=1e-9
        )


def test_an_incomplete_selection_leaves_the_grid_empty(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path), snap_labels_c=[])
    logic.tiles.update_tiles(0)

    assert logic.server.state.snap_no_interface
    assert all(
        r.GetViewProps().GetNumberOfItems() == 0
        for r in logic.scene.tile_views.renderers
    )


def test_nothing_is_built_while_the_grid_is_off_screen(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path), maximized_view="")
    logic.tiles.update_tiles(0)

    assert logic.tiles._tile_sets == {}


def test_tile_sets_are_cached_per_frame(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)
    first = logic.tiles._tile_sets[("volume:vol", 0)]

    logic.tiles.update_tiles(0)
    assert logic.tiles._tile_sets[("volume:vol", 0)] is first

    logic.tiles.update_tiles(1)
    assert logic.tiles._tile_sets[("volume:vol", 0)] is first
    assert ("volume:vol", 1) in logic.tiles._tile_sets


def test_the_cache_is_dropped_when_the_grid_changes(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)

    logic.server.state.tile_cols = 4
    logic.tiles.update_tiles(0)

    assert len(logic.tiles._tile_sets[("volume:vol", 0)]) == 12


# Fitting the cameras


def scales(logic: FakeApp) -> list[float]:
    return [
        round(r.GetActiveCamera().GetParallelScale(), 6)
        for r in logic.scene.tile_views.renderers
    ]


def fitted(logic: FakeApp) -> bool:
    """Whether the tiles are framed rather than left at the camera default."""
    return all(scale != pytest.approx(1.0) for scale in scales(logic))


def test_the_first_tiles_shown_are_framed(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)

    assert fitted(logic)


def choose_group_c(logic: FakeApp, labels: list[int]):
    """Complete the selection, firing the listeners Logic wires to that write."""
    logic.server.state.snap_labels_c = labels
    logic.snap._on_snap_selection_changed()
    logic.tiles._on_path_changed()


def test_tiles_are_framed_when_the_path_arrives_after_the_grid(tmp_path):
    """Entering tile mode before choosing the groups still frames the tiles.

    The empty first pass used to record the grid, so the pass that finally had
    something to show saw no reshape and never fitted a camera -- leaving every
    tile at the default half-height of one world unit, zoomed far into the
    middle of the cut.
    """
    logic = tiled(stacked_segmentation(tmp_path), snap_labels_c=[])
    logic.tiles.update_tiles(0)
    assert not fitted(logic)

    choose_group_c(logic, [3])

    assert fitted(logic)


def test_an_empty_pass_does_not_consume_the_framing(tmp_path):
    """Whatever wakes it, the first pass that draws tiles has to frame them."""
    logic = tiled(stacked_segmentation(tmp_path), snap_labels_c=[])
    logic.tiles.update_tiles(0)

    logic.server.state.snap_labels_c = [3]
    logic.snap._invalidate_lock_cache()
    logic.tiles.update_tiles(0)

    assert fitted(logic)


def test_every_tile_is_framed_at_one_scale(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)

    assert len(set(scales(logic))) == 1


def test_a_frame_change_leaves_the_framing_alone(tmp_path):
    """Refitting per frame would make a cine pulse."""
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)
    before = scales(logic)

    for renderer in logic.scene.tile_views.renderers:
        renderer.GetActiveCamera().SetParallelScale(999.0)
    logic.tiles.update_tiles(0)

    assert scales(logic) != before
    assert scales(logic) == [999.0] * len(logic.scene.tile_views)


def test_a_new_grid_is_framed(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)

    logic.server.state.tile_cols = 5
    logic.tiles.update_tiles(0)

    assert fitted(logic)
    assert len(set(scales(logic))) == 1


def test_reset_cameras_reframes_on_demand(tmp_path):
    logic = tiled(stacked_segmentation(tmp_path))
    logic.tiles.update_tiles(0)
    for renderer in logic.scene.tile_views.renderers:
        renderer.GetActiveCamera().SetParallelScale(999.0)

    logic.tiles.reset_cameras()

    assert fitted(logic)


def test_an_enabled_overlay_lands_on_every_tile(tmp_path):
    seg = stacked_segmentation(tmp_path)
    logic = tiled(seg, **{ObjectState.of(seg).mpr_overlay: True})
    logic.tiles.update_tiles(0)

    for renderer in logic.scene.tile_views.renderers:
        # the volume cut and the overlay on top of it
        assert renderer.GetViewProps().GetNumberOfItems() == 2
