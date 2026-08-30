"""The tile grid: several cuts along a path, shown side by side."""

# System
import typing as ty

# Third Party
import numpy as np

# Internal
from ..reslice import TileSet
from ..segmentation import set_label_opacity
from ..state import ObjectState
from ..tile_views import MAX_COLS, MAX_ROWS
from .base import Controller
from .snap import ALIGN_STEP_NAME, alignment_rotation

# The value of ``maximized_view`` that puts the grid on screen.
TILE_LAYOUT = "tile"

Pose = tuple[list[float], np.ndarray]
PoseAt = ty.Callable[[float], Pose | None]


def sample_fractions(count: int) -> list[float]:
    """``count`` positions spread evenly over [0, 1], endpoints included.

    A single tile sits at the middle rather than at either end, which is the
    only position that does not privilege one landmark over the other.
    """
    if count < 1:
        return []
    if count == 1:
        return [0.5]
    return [i / (count - 1) for i in range(count)]


def poses_along(pose_at: PoseAt, count: int) -> list[Pose] | None:
    """One ``(origin, rotation)`` per sample, or None if any sample is missing.

    ``pose_at`` is the only thing that knows what the path is. Today it walks
    the traverse path between two interface planes; a parallel-slice source
    would step a fixed plane along its own normal, and nothing downstream would
    need to change.
    """
    poses = []
    for fraction in sample_fractions(count):
        pose = pose_at(fraction)
        if pose is None:
            return None
        poses.append(pose)
    return poses


class TileController(Controller):
    """The tile grid: what each tile shows, and how many there are."""

    def __init__(self, app):
        super().__init__(app)
        self._tile_sets: dict[tuple[str, int], TileSet] = {}
        self._grid: tuple[int, int] | None = None
        self._fitted = False

    def register(self):
        state = self.server.state
        state.tile_rows = self.scene.tile_rows
        state.tile_cols = self.scene.tile_cols
        state.tile_sizes = list(range(1, MAX_ROWS + 1))

        state.change("tile_rows", "tile_cols")(self.refresh)
        state.change("active_volume_label")(self._on_volume_changed)
        state.change("maximized_view")(self.refresh)
        state.change("mpr_rotation_data", "mpr_window", "mpr_level")(self.refresh)
        state.change("mpr_segmentation_opacity")(self.refresh)
        state.change(
            "snap_seg_label",
            "snap_mode",
            "snap_labels_a",
            "snap_labels_b",
            "snap_labels_c",
        )(self._on_path_changed)
        for seg in self.scene.segmentations:
            state.change(ObjectState.of(seg).mpr_overlay)(self.refresh)

        self.server.controller.reset_tile_cameras = self.reset_cameras

    @property
    def active(self) -> bool:
        """Whether the tile grid is the layout currently on screen."""
        return getattr(self.server.state, "maximized_view", "") == TILE_LAYOUT

    @property
    def tile_count(self) -> int:
        state = self.server.state
        rows = max(1, min(MAX_ROWS, int(getattr(state, "tile_rows", 1))))
        cols = max(1, min(MAX_COLS, int(getattr(state, "tile_cols", 1))))
        return rows * cols

    def tile_poses(self, frame: int) -> list[Pose] | None:
        """One ``(origin, rotation)`` per tile, in ITK, or None if unavailable.

        The rotation is the same composition the quad view gets: the plane at
        that point of the path, with whatever rotations the user has stacked on
        top of the alignment step applied after it. A tile is therefore the quad
        view's axial cut, taken at its own fraction.
        """
        user_rotation = self.app.rotations.rotation_matrix(exclude=ALIGN_STEP_NAME)

        def pose_at(fraction: float) -> Pose | None:
            pose = self.app.snap.traverse_pose(frame, fraction)
            if pose is None:
                return None
            origin, axes = pose
            return origin, alignment_rotation(axes) @ user_rotation

        return poses_along(pose_at, self.tile_count)

    def update_tiles(self, frame: int, reset_cameras: bool = False):
        """Repose and redraw every tile for ``frame``.

        A no-op unless the grid is on screen: the tiles are as expensive as the
        MPR views times the tile count, and nobody is looking at them.
        """
        views = self.scene.tile_views
        if views is None or not self.active:
            return

        volume = self._active_volume()
        if volume is None:
            return

        self._sync_grid(views)

        poses = self.tile_poses(frame)
        if poses is None:
            views.clear()
            self._fitted = False
            self.server.state.snap_no_interface = True
            self.server.controller.view_update()
            return

        state = self.server.state
        state.snap_no_interface = False

        tiles = self._tile_set(volume, frame, len(poses))
        tiles.set_poses(poses)
        tiles.set_window_level(state.mpr_window, state.mpr_level)
        views.show(tiles, reset_cameras=reset_cameras or not self._fitted)
        self._fitted = True

        for seg in self.scene.segmentations:
            if not state[ObjectState.of(seg).mpr_overlay]:
                continue
            overlay = self._tile_set(seg, frame, len(poses))
            overlay.set_poses(poses)
            set_label_opacity(overlay.values(), state.mpr_segmentation_opacity)
            views.add_overlay(overlay)

        self.server.controller.view_update()

    def refresh(self, **kwargs):
        """Re-tile at the current frame, for any change that alters the cuts."""
        self.update_tiles(getattr(self.server.state, "frame", 0))

    def reset_cameras(self, **kwargs):
        """Refit every tile, on demand.

        Frame changes deliberately do not refit -- a cine would pulse -- so
        this is also the way out if a fit ever goes stale.
        """
        self.update_tiles(getattr(self.server.state, "frame", 0), reset_cameras=True)

    def _on_path_changed(self, **kwargs):
        """A different path cuts a different shape, so refit to it."""
        self._fitted = False
        self.refresh()

    def _on_volume_changed(self, **kwargs):
        self._tile_sets.clear()
        self._fitted = False
        self.refresh()

    def _sync_grid(self, views):
        """Reshape the grid if the user has changed it.

        A reshape drops the cached tile sets, which are sized to the old count,
        and any renderer it adds arrives with a default camera, so the fit no
        longer describes what is on screen.
        """
        state = self.server.state
        grid = (
            max(1, min(MAX_ROWS, int(getattr(state, "tile_rows", 1)))),
            max(1, min(MAX_COLS, int(getattr(state, "tile_cols", 1)))),
        )
        if grid == self._grid:
            return

        views.set_grid(*grid)
        self._tile_sets.clear()
        self._grid = grid
        self._fitted = False

    def _tile_set(self, obj, frame: int, count: int) -> TileSet:
        """The reslice pipelines for one object's tiles, built once per frame."""
        key = (f"{obj.kind}:{obj.label}", frame)
        cached = self._tile_sets.get(key)
        if cached is not None and len(cached) == count:
            return cached

        image_data = obj.mpr_image_data(frame)
        if obj.kind == "segmentation":
            tiles = TileSet(
                image_data,
                count,
                interpolation="nearest",
                background_level=0,
                output_filter=obj.label_color_filter(image_data),
            )
        else:
            tiles = TileSet(
                image_data, count, interpolation="linear", background_level=-1000.0
            )

        self._tile_sets[key] = tiles
        return tiles
