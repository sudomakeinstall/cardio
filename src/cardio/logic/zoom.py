"""Fitting the MPR views to a chosen set of labels."""

# System
import logging

# Third Party
import numpy as np

# Internal
from ..action import action
from ..camera import fit_factor
from ..reslice import VIEW_TRANSFORMS, VIEWS
from ..segmentation import plane_shadow
from .base import Controller

logger = logging.getLogger(__name__)

# How far the origin has to be off the shadow's centre before the fit moves it.
# A tenth of a millimetre is below the finest voxel anybody reformats at.
RECENTRE_TOLERANCE = 0.1


class ZoomController(Controller):
    """The label set the views are fitted to, and the fit itself."""

    # zoom_seg_label is absent: an empty one means the first segmentation,
    # which is a fallback rather than a field, and the registry says so.
    SELECTION = ("zoom_labels", "zoom_plane", "zoom_fill")

    # Not part of the selection: it says how much of a label cloud to believe,
    # which is a fact about the series rather than a thing this panel picks.
    # TileController reads it for its stack the same way.
    MEASUREMENT = "label_percentile"

    # Apart from the selection for the reason snap keeps its own apart: the
    # lock acts the moment it is written, so it goes on last, once there is a
    # settled selection to act on.
    LOCK = "zoom_locked"

    seeds = (*SELECTION, MEASUREMENT, LOCK)

    def __init__(self, app):
        super().__init__(app)
        self._published_seg_label = None
        self._refitting = False

    def register(self):
        if not self.scene.segmentations:
            return

        state = self.server.state
        state.change("zoom_seg_label")(self._on_segmentation_changed)
        state.change(*self.SELECTION, self.MEASUREMENT, self.LOCK)(self.refit)
        # A fit is measured from the origin and along the rotated frame, so it
        # stops being the fit that was asked for the moment either moves. The
        # frame is not listened for: the cloud already spans every one of them,
        # and playback reaches this through the origin whenever snap moves it.
        #
        # Nor is the layout: the client resizes a render window after telling
        # the server which view it maximized, so a refit here would measure
        # against the size the viewport is about to stop having.
        state.change("mpr_origin", "mpr_rotation_data")(self.refit)

    def seed(self):
        """The selection, the pickers it is filtered against, and the lock.

        The selection and the lock are written whether or not there is a
        segmentation to fit to. They are document state, and a session that
        left them out would not be one a config could reopen -- a lock with
        nothing to fit is inert, and ``Zoom.warn_ineffective_lock`` already
        treats asking for one as a warning rather than an error.
        """
        self.server.state.zoom_seg_label = self.scene.zoom.segmentation_label
        self.write_seeds(*self.SELECTION, self.MEASUREMENT)
        self._publish_pickers()
        self.write_seeds(self.LOCK)
        self.refit()

    def _publish_pickers(self):
        """Fill the plane and label pickers, and filter what the config asked for."""
        if not self.scene.segmentations:
            return

        state = self.server.state
        state.zoom_plane_items = [
            {"title": view.capitalize(), "value": view} for view in VIEWS
        ]
        state.zoom_available_labels = []

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
            label = getattr(self.server.state, "zoom_seg_label", "")
        if not label:
            return self.scene.segmentations[0] if self.scene.segmentations else None
        return next((s for s in self.scene.segmentations if s.label == label), None)

    def _publish_available_labels(self, seg) -> bool:
        """Write the label picker's options for ``seg`` at the current frame.

        Returns whether this is a different segmentation from the one the
        picker was last built for.
        """
        self.server.state.zoom_available_labels = [
            {"title": str(value), "value": value}
            for value in seg.get_labels(self._frame)
        ]
        changed = seg.label != self._published_seg_label
        self._published_seg_label = seg.label
        return changed

    def _publish_configured_labels(self, seg):
        """Apply the configured labels, dropping the ones ``seg`` lacks.

        A typo leaves the fit with nothing to frame, which the panel already
        shows as a button that cannot be pressed; that is friendlier than
        refusing to launch over a label that is merely absent.
        """
        present = set(seg.get_labels(self._frame))
        asked = self.scene.zoom.labels
        missing = [value for value in asked if value not in present]
        if missing:
            logger.warning(
                f"Zoom labels {missing} are not present in segmentation {seg.label}."
            )
        self.server.state.zoom_labels = [value for value in asked if value in present]

    def _on_segmentation_changed(self, zoom_seg_label=None, **kwargs):
        """Repopulate the picker, and clear labels that no longer apply.

        The labels index into the segmentation being left, so they cannot
        survive a change of segmentation. The listener also fires on the first
        flush, for the value seeding wrote -- which is the segmentation the
        picker was just built for, so nothing is cleared.
        """
        seg = self._selected_segmentation(zoom_seg_label)
        if seg is None:
            self.server.state.zoom_available_labels = []
            return

        if self._publish_available_labels(seg):
            self.server.state.zoom_labels = []

    @property
    def fit_plane(self) -> str:
        """The plane the fit is measured in.

        The tile grid cuts in a plane of its own, and while it is up that is
        the plane being looked at -- ``zoom_plane`` names one of three
        viewports that are not being drawn at all.
        """
        if self.app.tiles.active:
            return self.app.tiles.cut_plane
        return getattr(self.server.state, "zoom_plane", "")

    def _viewport(self):
        """The renderer the fit is sized against, and its world-per-pixel.

        Whichever viewport is on screen. A fit measured against one that is
        not being drawn frames nothing, which is what the tile grid used to
        get: it was sized against an MPR window and applied to an MPR camera,
        and the tiles kept whatever scale their own refit had given them.
        """
        tiles = self.scene.tile_views
        if self.app.tiles.active and tiles is not None and len(tiles):
            return tiles.renderers[0], tiles.world_per_pixel()

        views = self.scene.mpr_views
        if views is None:
            return None
        return views.renderer(self.fit_plane), views.world_per_pixel(self.fit_plane)

    def shadow(self):
        """The chosen labels' shadow on the fitted plane, and that plane's basis.

        ``(centre, half_span, frame)``, the first two in the plane's own right
        and up axes and relative to the current origin. None whenever there is
        nothing to fit: no segmentation, no labels, or a label set the series
        never carries.

        While the tile grid is up the frame is the one the parallel sources cut
        in, which is also the traverse source's whenever its alignment is the
        rotation the views carry. The traverse tiles each tilt a little off it,
        which is the same reason the grid holds one shared scale rather than
        one fit per tile.
        """
        state = self.server.state
        labels = list(getattr(state, "zoom_labels", []) or [])
        seg = self._selected_segmentation()
        if seg is None or not labels:
            return None

        plane = self.fit_plane
        if plane not in VIEW_TRANSFORMS:
            return None

        # The basis the cuts are actually taken in, composed the way
        # ``MPRController.pan_vectors`` composes it, and in ITK as all the
        # rotation math is.
        frame = self.app.rotations.rotation_matrix() @ VIEW_TRANSFORMS[plane]
        origin = self.convention.point_to_itk(state.mpr_origin)
        shadow = plane_shadow(
            seg.label_cloud(labels), frame, origin, state.label_percentile
        )
        if shadow is None:
            return None

        centre, half_span = shadow
        return centre, half_span, frame

    def zoom_factor(self, half_span) -> float | None:
        """The factor that brings ``half_span`` inside the viewport on screen.

        Measured about the origin, which ``zoom_to_labels`` has put on the
        shadow's centre by the time this is asked.

        None whenever the window the fit is measured against has never been
        sized -- which it has not been until a client has connected and laid
        the viewports out.
        """
        viewport = self._viewport()
        if viewport is None:
            return None

        renderer, per_pixel = viewport
        return fit_factor(
            tuple(half_span),
            renderer.GetSize(),
            per_pixel,
            self.server.state.zoom_fill / 100.0,
        )

    @action("zoom_to_labels")
    def zoom_to_labels(self):
        """Frame the chosen labels' shadow on the chosen plane.

        The origin slides onto the shadow's centre first, and the fit is then
        measured against the half-span rather than against the farthest edge:
        a crosshair snapped to one end of what is being framed would otherwise
        cost the fit a viewport's worth of empty space at the other.

        The slide is along the plane's own two axes, so the plane being fitted
        goes on cutting where it did and only the crosshair moves within it.
        The other two views do re-cut, that pair of axes being the axes they
        scroll along -- with a snap lock also on, snap owns the origin along
        the third axis and the fit owns it along these two.

        Sized against one plane and applied to all three, which is how the MPR
        views already share a zoom: each keeps the absolute scale its own fit
        gave it and only the factor is passed around, so the three stay in
        whatever relation the user last put them in.

        Not routed through ``zoom_views``, which refuses to move while the fit
        is locked -- that guard is there to stop a drag fighting the lock, and
        this is the thing the lock exists to apply.
        """
        shadow = self.shadow()
        if shadow is None:
            return

        centre, half_span, frame = shadow
        factor = self.zoom_factor(half_span)
        if factor is None:
            return

        self._recentre(centre, frame)
        self._zoom_views(factor)
        self.server.controller.view_update()

    def _zoom_views(self, factor: float):
        """Apply the fit to whatever it was sized against, and only that.

        The two sets of cameras hold their own absolute scales, so a factor
        measured against one viewport frames nothing in the other.
        """
        if self.app.tiles.active and self.scene.tile_views is not None:
            self.scene.tile_views.zoom(factor)
            return
        self.scene.mpr_views.zoom(factor)

    def _recentre(self, centre, frame):
        """Slide the origin onto ``centre``, given in ``frame``'s first two axes.

        A shift small enough to make no difference to the picture is skipped
        rather than written: the fit settles on a centre it has already reached,
        and re-writing the origin there would only wake every listener on it.
        """
        if np.linalg.norm(centre) < RECENTRE_TOLERANCE:
            return

        origin = self.convention.point_to_itk(self.server.state.mpr_origin)
        moved = np.asarray(origin, dtype=float) + frame[:, :2] @ centre
        self.app.mpr.set_origin([float(value) for value in moved])

    @property
    def locked(self) -> bool:
        """Whether the fit is the standing owner of the MPR zoom.

        A lock re-applies the fit every time the views move, so a zoom gesture
        would be thrown away again the next time anything else did -- which is
        why the gesture is refused outright rather than left to lose.
        """
        return bool(getattr(self.server.state, "zoom_locked", False))

    def refit(self, **kwargs):
        """Fit again, if the fit is being held.

        The listeners and ``finalize_mpr_initialization`` both come here. The
        MPR windows are built after Logic is, so at seeding time a configured
        lock has nothing to fit against; the second call, once the views are
        real, is what gives it something.

        The guard is for the origin listener: the fit writes the origin, and a
        held fit would otherwise be woken by its own recentring. One pass is
        all it would take -- the shift is zero once the origin is on the centre
        -- but the reentry would be inside the flush that provoked it.
        """
        if not self.locked or self._refitting:
            return

        self._refitting = True
        try:
            self.zoom_to_labels()
        finally:
            self._refitting = False
