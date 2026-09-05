"""Shared plumbing for the parts of Logic."""

# Internal
from ..convention import Convention
from ..state import ObjectState


class Controller:
    """One concern of the application logic.

    Each controller owns a slice of trame state: it declares its listeners and
    controller functions in ``register()`` and writes the state itself in
    ``seed()``. Siblings are reached through ``self.app``, which is the Logic
    facade that composes them.
    """

    def __init__(self, app):
        self.app = app
        self.server = app.server
        self.scene = app.scene

    def register(self):
        """Declare this controller's listeners and controller functions."""

    def seed(self):
        """Write this controller's state, as the scene configures it.

        Split from ``register`` so that opening a config, restoring a saved
        session and resetting to what the config asks for are one pass rather
        than three: every controller's ``seed`` runs together, in one order, on
        a server whose listeners are already in place.
        """

    @property
    def convention(self) -> Convention:
        """The index order and angle units the MPR state is currently written in."""
        return Convention.from_metadata(self.scene.mpr_rotation_sequence.metadata)

    @property
    def _frame(self) -> int:
        """The frame being shown."""
        return self.server.state.frame

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
