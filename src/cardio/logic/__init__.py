"""Application logic, composed from one controller per concern.

``Logic`` keeps the public surface it always had -- app.py constructs it, the UI
binds its controller functions -- but the work now lives in the controllers,
each owning its own slice of trame state.
"""

# Internal
import cardio.registry as registry

from ..action import Journal, Registry
from ..document import to_toml
from ..scene import Scene
from .base import Controller
from .camera import CameraController
from .capture import CaptureController
from .clipping import ClippingController
from .console import ConsoleController
from .mpr import MPRController
from .playback import PlaybackController
from .rotations import RotationController
from .snap import ALIGN_STEP_NAME, SnapController
from .tiles import TileController
from .view import ViewController
from .visibility import VisibilityController
from .zoom import ZoomController

__all__ = [
    "ALIGN_STEP_NAME",
    "CameraController",
    "CaptureController",
    "ClippingController",
    "ConsoleController",
    "Controller",
    "Logic",
    "MPRController",
    "PlaybackController",
    "RotationController",
    "SnapController",
    "TileController",
    "ViewController",
    "VisibilityController",
    "ZoomController",
]


class Logic:
    """Composes the controllers and wires them to the server.

    Declaring and seeding are two passes rather than one. Every controller
    registers its listeners and controller functions first; only then does
    ``apply_scene`` write the state the scene configures. Trame looks its
    change callbacks up at flush time and nothing here flushes, so the order
    the two are interleaved in never mattered to the listeners. What it does
    decide is whether a controller reading a sibling's state finds it written
    yet -- ``snap`` snaps against the MPR origin, the tiles are cut along the
    snap path -- and one ordered pass is where that can be seen.
    """

    def __init__(self, server, scene: Scene):
        self.server = server
        self.scene = scene

        # Taken before anything runs, because an action moves the scene under
        # it -- a rotation added is a rotation the sequence now holds. A script
        # replayed against where the session ended up would do it all twice, so
        # what a script needs is the config that opens where it began.
        self.opened_as = to_toml(scene)

        self.console = ConsoleController(self)
        self.view = ViewController(self)
        self.rotations = RotationController(self)
        self.mpr = MPRController(self)
        self.snap = SnapController(self)
        self.playback = PlaybackController(self)
        self.visibility = VisibilityController(self)
        self.clipping = ClippingController(self)
        self.tiles = TileController(self)
        self.capture = CaptureController(self)
        self.zoom = ZoomController(self)
        self.camera = CameraController(self)

        self.journal = Journal(
            server.state, self.document_keys, refresh=self.camera.publish
        )
        self.actions = Registry(self.journal)
        for controller in self.controllers:
            controller.register()
            self.actions.add(controller)
        self.actions.bind(self.server.controller)

        self.apply_scene()

    def document_keys(self) -> list[str]:
        """Every state key that says what the app is showing."""
        return registry.document_keys(self.scene)

    def dispatch(self, name: str, *positional, **arguments):
        """Do the named thing.

        The one way an action is called, whichever asked for it -- a button, a
        gesture, a script, a typed line. Anything that has to happen around
        every action goes here and nowhere else.

        Positional arguments are passed through rather than refused: the
        registry resolves them against the signature, so a hand-typed call may
        give its arguments the way the method spells them.
        """
        return self.actions.run(name, *positional, **arguments)

    def apply_scene(self):
        """Write every state variable the scene configures.

        The one way state is put where a ``Scene`` says it should be, whether
        that scene came from a config file at startup or from somewhere else
        later.
        """
        for controller in self.controllers:
            controller.seed()

    @property
    def controllers(self) -> list[Controller]:
        """The controllers, in the order they register and then seed.

        ``console`` is first because it reads and writes nothing any sibling
        owns, and being first is how that is said. ``snap`` is late because a
        configured lock snaps the moment it is seeded, which reads the origin,
        the rotation and the frame that the controllers above it write; ``zoom``
        follows it, fitting the views against the origin and the rotation that
        lock has just settled; ``camera`` is last because it reads where every
        camera ended up once all of that has happened.
        """
        return [
            self.console,
            self.view,
            self.rotations,
            self.mpr,
            self.playback,
            self.visibility,
            self.clipping,
            self.tiles,
            self.capture,
            self.snap,
            self.zoom,
            self.camera,
        ]
