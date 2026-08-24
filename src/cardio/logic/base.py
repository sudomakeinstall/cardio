"""Shared plumbing for the parts of Logic."""

# Internal
from ..convention import Convention


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

    def _active_volume(self):
        """The volume the MPR views are showing, or None if there isn't one."""
        label = self.server.state.active_volume_label
        return next((v for v in self.scene.volumes if v.label == label), None)
