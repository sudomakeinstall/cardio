"""Fitting the MPR views to a chosen set of labels."""

# System
import logging

# Internal
from ..action import action
from ..camera import fit_factor
from ..reslice import VIEW_TRANSFORMS, VIEWS
from ..segmentation import plane_half_extent
from .base import Controller

logger = logging.getLogger(__name__)


class ZoomController(Controller):
    """The label set the views are fitted to, and the fit itself."""

    # zoom_seg_label is absent: an empty one means the first segmentation,
    # which is a fallback rather than a field, and the registry says so.
    SELECTION = ("zoom_labels", "zoom_plane", "zoom_fill")

    # Apart from the selection for the reason snap keeps its own apart: the
    # lock acts the moment it is written, so it goes on last, once there is a
    # settled selection to act on.
    LOCK = "zoom_locked"

    seeds = (*SELECTION, LOCK)

    def __init__(self, app):
        super().__init__(app)
        self._published_seg_label = None

    def register(self):
        if not self.scene.segmentations:
            return

        state = self.server.state
        state.change("zoom_seg_label")(self._on_segmentation_changed)
        state.change(*self.SELECTION, self.LOCK)(self.refit)
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
        self.write_seeds(*self.SELECTION)
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

    def half_extent(self):
        """How far the chosen labels reach from the origin, in the chosen plane.

        None whenever there is nothing to fit: no segmentation, no labels, or a
        label set the series never carries.
        """
        state = self.server.state
        labels = list(getattr(state, "zoom_labels", []) or [])
        seg = self._selected_segmentation()
        if seg is None or not labels:
            return None

        plane = getattr(state, "zoom_plane", "")
        if plane not in VIEW_TRANSFORMS:
            return None

        # The basis the cuts are actually taken in, composed the way
        # ``MPRController.pan_vectors`` composes it, and in ITK as all the
        # rotation math is.
        frame = self.app.rotations.rotation_matrix() @ VIEW_TRANSFORMS[plane]
        origin = self.convention.point_to_itk(state.mpr_origin)
        return plane_half_extent(seg.label_cloud(labels), frame, origin)

    def zoom_factor(self) -> float | None:
        """The factor that brings the chosen labels inside the chosen viewport.

        None whenever there is nothing to fit, and whenever the window the fit
        is measured against has never been sized -- which it has not been until
        a client has connected and laid the viewports out.
        """
        views = self.scene.mpr_views
        reach = self.half_extent()
        if views is None or reach is None:
            return None

        plane = self.server.state.zoom_plane
        return fit_factor(
            reach,
            views.renderer(plane).GetSize(),
            views.world_per_pixel(plane),
            self.server.state.zoom_fill / 100.0,
        )

    @action("zoom_to_labels")
    def zoom_to_labels(self):
        """Fit the views to the chosen labels' shadow on the chosen plane.

        Sized against one plane and applied to all three, which is how the MPR
        views already share a zoom: each keeps the absolute scale its own fit
        gave it and only the factor is passed around, so the three stay in
        whatever relation the user last put them in.

        Not routed through ``zoom_views``, which refuses to move while the fit
        is locked -- that guard is there to stop a drag fighting the lock, and
        this is the thing the lock exists to apply.
        """
        factor = self.zoom_factor()
        if factor is None:
            return

        self.scene.mpr_views.zoom(factor)
        self.server.controller.view_update()

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
        """
        if self.locked:
            self.zoom_to_labels()
