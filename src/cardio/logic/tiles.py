"""The tile grid: several cuts along a path, shown side by side."""

# System
import logging
import typing as ty

# Third Party
import numpy as np

# Internal
from ..action import action
from ..reslice import VIEW_TRANSFORMS, VIEWS, TileSet
from ..segmentation import plane_depth, set_label_opacity
from ..state import ObjectState
from ..tile import TileSource
from ..tile_views import MAX_COLS, MAX_ROWS
from .base import Controller
from .snap import ALIGN_STEP_NAME, alignment_rotation

logger = logging.getLogger(__name__)

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


def poses_along(
    pose_at: PoseAt, count: int, reverse: bool = False
) -> list[Pose] | None:
    """One ``(origin, rotation)`` per sample, or None if any sample is missing.

    ``pose_at`` is the only thing that knows what the path is: the traverse
    path between two interface planes, or a fixed plane stepped along its own
    normal. Nothing downstream can tell which it was handed.

    ``reverse`` walks the same path from the other end. It reverses the order
    the samples are taken in and nothing else, which is what makes it different
    from turning the cut around: a half turn about an in-plane axis would flip
    the normal, but it would flip one of the in-plane axes with it and hand
    back every tile mirrored.
    """
    fractions = sample_fractions(count)
    if reverse:
        fractions.reverse()

    poses = []
    for fraction in fractions:
        pose = pose_at(fraction)
        if pose is None:
            return None
        poses.append(pose)
    return poses


class TileController(Controller):
    """The tile grid: what each tile shows, and how many there are."""

    # tile_source is absent: two of its three values need a segmentation, so a
    # scene without one is seeded onto the third and the registry says so.
    seeds = (
        "tile_rows",
        "tile_cols",
        "tile_plane",
        "tile_spacing",
        "tile_labels",
        "tile_reverse",
    )

    def __init__(self, app):
        super().__init__(app)
        self._tile_sets: dict[tuple[str, int], TileSet] = {}
        self._grid: tuple[int, int] | None = None
        self._fitted = False
        self._published_seg_label = None

    def register(self):
        state = self.server.state
        state.change("tile_rows", "tile_cols")(self.refresh)
        state.change("active_volume_label")(self._on_volume_changed)
        state.change("maximized_view")(self.refresh)
        # The parallel sources are anchored to the origin, so they follow it the
        # way they follow the rotation.
        state.change("mpr_origin")(self.refresh)
        state.change("mpr_rotation_data", "mpr_window", "mpr_level")(self.refresh)
        state.change(
            "tile_source",
            "tile_plane",
            "tile_spacing",
            "tile_labels",
            "tile_reverse",
            "label_percentile",
        )(self._on_path_changed)
        state.change("tile_seg_label")(self._on_segmentation_changed)
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

    def seed(self):
        state = self.server.state
        state.tile_seg_label = self.scene.tile.segmentation_label
        # A scene with no segmentation opens on the one source that does not
        # need one, whatever it asked for; the panel offers no other there.
        state.tile_source = (
            self.scene.tile.source if self.scene.segmentations else TileSource.SPACING
        )
        super().seed()
        state.tile_sizes = list(range(1, MAX_ROWS + 1))
        state.tile_plane_items = [
            {"title": view.capitalize(), "value": view} for view in VIEWS
        ]
        self._publish_labels()

    def _publish_labels(self):
        """Fill the label picker, and filter what the config asked for."""
        if not self.scene.segmentations:
            return

        self.server.state.tile_available_labels = []
        seg = self._selected_segmentation()
        if seg is None:
            return

        self._publish_available_labels(seg)
        self._publish_configured_labels(seg)

    def _selected_segmentation(self, label: str | None = None):
        """The segmentation ``label`` names, defaulting to the selected one.

        An empty label takes the first, which is what makes a one-segmentation
        scene need no picker at all.
        """
        if label is None:
            label = getattr(self.server.state, "tile_seg_label", "")
        if not label:
            return self.scene.segmentations[0] if self.scene.segmentations else None
        return next((s for s in self.scene.segmentations if s.label == label), None)

    def _publish_available_labels(self, seg) -> bool:
        """Write the label picker's options for ``seg`` at the current frame.

        Returns whether this is a different segmentation from the one the
        picker was last built for.
        """
        self.server.state.tile_available_labels = [
            {"title": str(value), "value": value}
            for value in seg.get_labels(self._frame)
        ]
        changed = seg.label != self._published_seg_label
        self._published_seg_label = seg.label
        return changed

    def _publish_configured_labels(self, seg):
        """Apply the configured labels, dropping the ones ``seg`` lacks.

        A typo leaves the source with nothing to span, which the panel already
        shows as a grid it cannot fill; that is friendlier than refusing to
        launch over a label that is merely absent.
        """
        present = set(seg.get_labels(self._frame))
        asked = self.scene.tile.labels
        missing = [value for value in asked if value not in present]
        if missing:
            logger.warning(
                f"Tile labels {missing} are not present in segmentation {seg.label}."
            )
        self.server.state.tile_labels = [value for value in asked if value in present]

    def _on_segmentation_changed(self, tile_seg_label=None, **kwargs):
        """Repopulate the picker, and clear labels that no longer apply.

        The labels index into the segmentation being left, so they cannot
        survive a change of segmentation. The listener also fires on the first
        flush, for the value seeding wrote -- which is the segmentation the
        picker was just built for, so nothing is cleared.
        """
        seg = self._selected_segmentation(tile_seg_label)
        if seg is None:
            self.server.state.tile_available_labels = []
            return

        if self._publish_available_labels(seg):
            self.server.state.tile_labels = []
        self._on_path_changed()

    @property
    def active(self) -> bool:
        """Whether the tile grid is the layout currently on screen."""
        return getattr(self.server.state, "maximized_view", "") == TILE_LAYOUT

    @property
    def source(self) -> str:
        """Which path the tiles are currently taken along."""
        return getattr(self.server.state, "tile_source", TileSource.TRAVERSE)

    @property
    def reverse(self) -> bool:
        """Whether the tiles are taken from the far end of the path."""
        return bool(getattr(self.server.state, "tile_reverse", False))

    @property
    def cut_plane(self) -> str:
        """The plane every tile is cut in.

        The traverse source has no plane to pick: its cut is the interface the
        path interpolates, which its poses already carry.
        """
        if self.source == TileSource.TRAVERSE:
            return "ul"
        plane = getattr(self.server.state, "tile_plane", "ul")
        return plane if plane in VIEW_TRANSFORMS else "ul"

    @property
    def tile_count(self) -> int:
        state = self.server.state
        rows = max(1, min(MAX_ROWS, int(getattr(state, "tile_rows", 1))))
        cols = max(1, min(MAX_COLS, int(getattr(state, "tile_cols", 1))))
        return rows * cols

    def tile_poses(self, frame: int) -> list[Pose] | None:
        """One ``(origin, rotation)`` per tile, in ITK, or None if unavailable."""
        pose_at = self._pose_at(frame)
        if pose_at is None:
            return None
        return poses_along(pose_at, self.tile_count, self.reverse)

    def _pose_at(self, frame: int) -> PoseAt | None:
        """The path the current source samples, as a function of fraction.

        None when the source cannot describe a path at all, which is a
        different thing from a path with a gap in it.
        """
        if self.source == TileSource.TRAVERSE:
            return self._traverse_pose_at(frame)
        reach = self._reach(frame)
        return None if reach is None else self._parallel_pose_at(*reach)

    def _traverse_pose_at(self, frame: int) -> PoseAt:
        """The path between the two interface planes.

        The rotation is the same composition the quad view gets: the plane at
        that point of the path, with whatever rotations the user has stacked on
        top of the alignment step applied after it. A tile is therefore the quad
        view's upper-left cut, taken at its own fraction.
        """
        user_rotation = self.app.rotations.rotation_matrix(exclude=ALIGN_STEP_NAME)

        def pose_at(fraction: float) -> Pose | None:
            pose = self.app.snap.traverse_pose(frame, fraction)
            if pose is None:
                return None
            origin, axes = pose
            return origin, alignment_rotation(axes) @ user_rotation

        return pose_at

    def _reach(self, frame: int) -> tuple[float, float] | None:
        """How far either way along the normal a parallel stack should reach.

        Offsets from the current origin, in millimetres, for the tile at
        fraction 0 and the tile at fraction 1. None when the source has nothing
        to measure itself against.
        """
        if self.source == TileSource.LABELS:
            depth = self.depth()
            if depth is None:
                return None
            centre, half_span = depth
            return centre - half_span, centre + half_span

        half = self.server.state.tile_spacing * (self.tile_count - 1) / 2.0
        return -half, half

    def depth(self) -> tuple[float, float] | None:
        """The chosen labels' extent along the chosen plane's normal.

        ``(centre, half_span)``, in millimetres and relative to the current
        origin, so the outermost tiles land on the bounds of what the labels
        are believed down to, and the spacing between them is whatever the grid
        size makes it.

        The cloud spans every frame at once, so a stack sized on one frame
        still crosses the labels on the next rather than breathing with them.
        """
        labels = list(getattr(self.server.state, "tile_labels", []) or [])
        seg = self._selected_segmentation()
        if seg is None or not labels:
            return None

        rotation = self.app.rotations.rotation_matrix()
        normal = (rotation @ VIEW_TRANSFORMS[self.cut_plane])[:, 2]
        origin = self.convention.point_to_itk(self.server.state.mpr_origin)
        return plane_depth(
            seg.label_cloud(labels),
            normal,
            origin,
            self.server.state.label_percentile,
        )

    def _parallel_pose_at(self, low: float, high: float) -> PoseAt:
        """A stack of parallel cuts, stepped along the chosen plane's normal.

        Unlike the traverse source, the rotation is the whole of the user's --
        the alignment step included, since there is no interface plane here to
        replace it. Every tile is the quad view's own cut, moved off it.
        """
        rotation = self.app.rotations.rotation_matrix()
        normal = (rotation @ VIEW_TRANSFORMS[self.cut_plane])[:, 2]
        origin = np.asarray(
            self.convention.point_to_itk(self.server.state.mpr_origin), dtype=float
        )

        def pose_at(fraction: float) -> Pose:
            offset = origin + (low + fraction * (high - low)) * normal
            return [float(value) for value in offset], rotation

        return pose_at

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

        state = self.server.state
        poses = self.tile_poses(frame)
        # Only the traverse source has an interface to be missing; the parallel
        # sources say what they are short of in the panel, not in snap's key.
        traversing = self.source == TileSource.TRAVERSE

        if poses is None:
            views.clear()
            self._fitted = False
            if traversing:
                state.snap_no_interface = True
            self.server.controller.view_update()
            return

        if traversing:
            state.snap_no_interface = False

        plane = self.cut_plane
        tiles = self._tile_set(volume, frame, len(poses))
        tiles.set_poses(poses, plane)
        tiles.set_window_level(state.mpr_window, state.mpr_level)

        refitting = reset_cameras or not self._fitted
        views.show(tiles, reset_cameras=refitting)
        self._fitted = True

        for seg in self.scene.segmentations:
            if not state[ObjectState.of(seg).mpr_overlay]:
                continue
            overlay = self._tile_set(seg, frame, len(poses))
            overlay.set_poses(poses, plane)
            set_label_opacity(overlay.values(), state.mpr_segmentation_opacity)
            views.add_overlay(overlay)

        # A refit frames the whole cut. Where a fit is being held, that is not
        # what is on screen and the held one gets the last word -- the same
        # thing the lock does when the origin or the rotation moves under it.
        if refitting:
            self.app.zoom.refit()

        self.server.controller.view_update()

    def refresh(self, **kwargs):
        """Re-tile at the current frame, for any change that alters the cuts."""
        self.update_tiles(self._frame)

    @action("zoom_tiles")
    def zoom_tiles(self, factor: float):
        """Zoom the whole grid, leaving the fit that put the tiles on one scale."""
        views = self.scene.tile_views
        if views is None or not self.active:
            return

        views.zoom(factor)
        self.server.controller.view_update()

    @action("reset_tile_cameras")
    def reset_cameras(self, **kwargs):
        """Refit every tile, on demand.

        Frame changes deliberately do not refit -- a cine would pulse -- so
        this is also the way out if a fit ever goes stale.
        """
        self.update_tiles(self._frame, reset_cameras=True)

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
