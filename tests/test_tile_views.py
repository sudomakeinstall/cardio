"""Test that TileViews lays out one renderer per tile in a single window."""

import numpy as np
import pytest
import vtk

from cardio.orientation import create_vtk_reslice_matrix
from cardio.reslice import VIEW_TRANSFORMS, TileSet
from cardio.tile_views import MAX_COLS, MAX_ROWS, TileViews, tile_viewport
from tests.test_reslice import make_image


def tiles(count: int, background: float = -1000.0) -> TileSet:
    return TileSet(
        make_image(), count, interpolation="linear", background_level=background
    )


@pytest.fixture
def views() -> TileViews:
    grid = TileViews()
    grid.set_grid(3, 4)
    return grid


def prop_count(views: TileViews, tile: int) -> int:
    return views.renderers[tile].GetViewProps().GetNumberOfItems()


def window_renderer_count(views: TileViews) -> int:
    return views.window.GetRenderers().GetNumberOfItems()


# Layout


def test_the_window_is_offscreen():
    assert TileViews().window.GetOffScreenRendering() == 1


def test_the_grid_holds_one_renderer_per_tile(views):
    assert len(views) == 12
    assert window_renderer_count(views) == 12


def test_tile_zero_is_top_left():
    x0, y0, x1, y1 = tile_viewport(0, 3, 4)
    assert (x0, x1) == (0.0, 0.25)
    assert (y0, y1) == pytest.approx((2 / 3, 1.0))


def test_tiles_cover_the_window_without_gaps_or_overlap():
    rows, cols = 3, 4
    covered = np.zeros((rows, cols))
    for index in range(rows * cols):
        x0, y0, x1, y1 = tile_viewport(index, rows, cols)
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0
        covered[round(y0 * rows), round(x0 * cols)] += 1
    assert covered.sum() == rows * cols
    assert covered.max() == 1


def test_tiles_run_left_to_right_then_down():
    first, second = tile_viewport(0, 2, 2), tile_viewport(1, 2, 2)
    third = tile_viewport(2, 2, 2)
    assert second[0] > first[0] and second[1] == first[1]
    assert third[1] < first[1] and third[0] == first[0]


def test_growing_the_grid_adds_renderers_to_the_window(views):
    views.set_grid(4, 4)
    assert len(views) == 16
    assert window_renderer_count(views) == 16


def test_shrinking_the_grid_removes_them_again(views):
    views.set_grid(2, 2)
    assert len(views) == 4
    assert window_renderer_count(views) == 4


def test_the_grid_is_clamped_to_something_renderable():
    grid = TileViews()
    grid.set_grid(0, 99)
    assert (grid.rows, grid.cols) == (1, MAX_COLS)
    grid.set_grid(99, 0)
    assert (grid.rows, grid.cols) == (MAX_ROWS, 1)


def test_tiles_use_parallel_projection(views):
    """Comparing sizes across tiles only means anything without perspective."""
    for renderer in views.renderers:
        assert renderer.GetActiveCamera().GetParallelProjection() == 1


# Contents


def test_set_images_shows_one_actor_per_tile(views):
    grid = tiles(12)
    views.set_images(grid)

    for tile in range(12):
        assert prop_count(views, tile) == 1
        assert grid[tile]["actor"].GetVisibility() == 1


def test_clear_empties_every_tile(views):
    views.set_images(tiles(12))
    views.clear()

    for tile in range(12):
        assert prop_count(views, tile) == 0


def test_show_replaces_rather_than_accumulates(views):
    views.show(tiles(12))
    views.show(tiles(12))

    for tile in range(12):
        assert prop_count(views, tile) == 1


def test_overlays_stack_on_top_of_the_image(views):
    views.show(tiles(12))
    overlay = tiles(12, background=0.0)
    views.add_overlay(overlay)

    for tile in range(12):
        assert prop_count(views, tile) == 2


def test_reset_cameras_puts_every_tile_on_one_scale(views):
    views.set_images(tiles(12))
    views.reset_cameras()

    scales = {round(r.GetActiveCamera().GetParallelScale(), 9) for r in views.renderers}
    assert len(scales) == 1


def test_reset_cameras_tolerates_an_empty_grid():
    TileViews().reset_cameras()


# TileSet


def test_a_tile_set_builds_one_pipeline_per_tile():
    grid = tiles(9)
    assert len(grid) == 9
    assert all(
        isinstance(parts["reslice"], vtk.vtkImageReslice) for parts in grid.values()
    )


def test_set_poses_writes_the_axial_matrix_each_tile_asks_for():
    grid = tiles(3)
    poses = [
        ([float(i), 1.0, 2.0], np.diag([1.0, -1.0, -1.0]) if i else np.eye(3))
        for i in range(3)
    ]
    grid.set_poses(poses)

    for tile, (origin, rotation) in enumerate(poses):
        expected = create_vtk_reslice_matrix(
            rotation @ VIEW_TRANSFORMS["axial"], origin
        )
        actual = grid[tile]["reslice"].GetResliceAxes()
        for row in range(4):
            for column in range(4):
                assert actual.GetElement(row, column) == pytest.approx(
                    expected.GetElement(row, column)
                )


def test_set_poses_ignores_poses_it_has_no_tile_for():
    grid = tiles(2)
    grid.set_poses([([0.0, 0.0, 0.0], np.eye(3))] * 5)
    assert len(grid) == 2
