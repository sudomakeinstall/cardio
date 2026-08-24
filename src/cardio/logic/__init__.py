"""Application logic, composed from one controller per concern.

``Logic`` keeps the public surface it always had -- app.py constructs it, the UI
binds its controller functions -- but the work now lives in the controllers,
each owning its own slice of trame state.
"""

# Internal
from ..scene import Scene
from .base import Controller
from .capture import CaptureController
from .clipping import ClippingController
from .mpr import MPRController
from .playback import PlaybackController
from .rotations import RotationController
from .snap import ALIGN_STEP_NAME, SnapController
from .visibility import VisibilityController

__all__ = [
    "Logic",
    "ALIGN_STEP_NAME",
    "Controller",
    "CaptureController",
    "ClippingController",
    "MPRController",
    "PlaybackController",
    "RotationController",
    "SnapController",
    "VisibilityController",
]


class Logic:
    """Composes the controllers and wires them to the server.

    Registration order matters: trame fires a change listener only for writes
    that happen after it is registered, so each controller registers its
    listeners and its defaults together, in the order the original constructor
    established.
    """

    def __init__(self, server, scene: Scene):
        self.server = server
        self.scene = scene

        self.rotations = RotationController(self)
        self.mpr = MPRController(self)
        self.snap = SnapController(self)
        self.playback = PlaybackController(self)
        self.visibility = VisibilityController(self)
        self.clipping = ClippingController(self)
        self.capture = CaptureController(self)

        for controller in self.controllers:
            controller.register()

        # Depends on every controller's listeners already being in place
        self.mpr.register_initial_view()
        self.clipping._initialize_clipping_state()
        self.snap.register_initial_labels()

    @property
    def controllers(self) -> list[Controller]:
        return [
            self.rotations,
            self.mpr,
            self.playback,
            self.visibility,
            self.clipping,
            self.snap,
            self.capture,
        ]
