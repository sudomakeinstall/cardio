"""Shared plumbing for the parts of Logic."""

# Internal
from ..convention import Convention
from ..state import ObjectState


class Controller:
    """One concern of the application logic.

    Each controller owns a slice of trame state: it registers its own variables,
    change listeners and controller functions in ``register()``. Siblings are
    reached through ``self.app``, which is the Logic facade that composes them.
    """

    def __init__(self, app):
        self.app = app
        self.server = app.server
        self.scene = app.scene

    def register(self):
        """Declare this controller's state, listeners and controller functions."""

    @property
    def convention(self) -> Convention:
        """The index order and angle units the MPR state is currently written in."""
        return Convention.from_metadata(self.scene.mpr_rotation_sequence.metadata)

    @property
    def _frame(self) -> int:
        """The frame being shown.

        ``frame`` reaches state only when the playback slider is built, which
        happens after Logic; until then the scene's configured frame is the one
        that will be shown.
        """
        frame = getattr(self.server.state, "frame", None)
        return self.scene.current_frame if frame is None else frame

    def _active_volume(self):
        """The volume the MPR views are showing, or None if there isn't one."""
        label = self.server.state.active_volume_label
        return next((v for v in self.scene.volumes if v.label == label), None)

    def _overlaid_segmentations(self):
        """The segmentations switched on over the slices."""
        return [
            seg
            for seg in self.scene.segmentations
            if self.server.state[ObjectState.of(seg).mpr_overlay]
        ]
