"""The app without a page: a scene, its logic, and a way to ask it for things.

``CardioApp`` is this plus a browser. Everything the app can do it can do here,
because the actions and the state were never the page's -- the page only ever
pressed the buttons. What a session adds is the two things a browser was
quietly providing: something to arm the change listeners, and somewhere for an
action that goes to the background to run.
"""

# System
import asyncio
import pathlib as pl

# Third Party
import trame as tm
import trame.app

# Internal
from . import toml
from .document import scene_from_state, to_toml
from .logic import Logic
from .scene import Scene
from .view import VIEW_FUNCTIONS


def until_settled(work):
    """Run ``work`` and whatever it starts in the background, to completion.

    A capture is a coroutine put on the event loop, and returns before it has
    written anything. A browser has a loop already running and something to
    return to; a script has neither, and wants the files on disk before the
    next line.
    """

    async def drive():
        work()
        pending = [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ]
        if pending:
            await asyncio.gather(*pending)

    asyncio.run(drive())


class Session:
    """A scene and its logic on a server that no browser has to connect to."""

    def __init__(self, scene: Scene, server=None):
        self.server = tm.app.get_server(server, client_type="vue3")
        self.scene = scene
        self.logic = Logic(self.server, scene)

        # Without a page nothing assigns these, and trame raises on a
        # controller function that has no implementation rather than passing
        # quietly.
        for name in VIEW_FUNCTIONS:
            getattr(self.server.controller, name).can_be_empty = True

    @classmethod
    def from_config(cls, config_file: pl.Path | str, server=None, **overrides):
        """A session on the scene a TOML config describes."""
        return cls(Scene.load(config_file=config_file, **overrides), server=server)

    @property
    def actions(self):
        """Everything this session can be asked to do."""
        return self.logic.actions

    def ready(self):
        """Arm the session, as a browser connecting would.

        Nothing flushes until the state is ready, so until this is called an
        action writes state that never reaches the renderer. The render windows
        the page would have built on its way past are built here instead.
        """
        if self.server.state.is_ready:
            return

        self.scene.setup_mpr_render_windows()
        self.scene.setup_tile_render_window()
        self.server.state.ready()
        self.server.controller.finalize_mpr_initialization()

    def scene_now(self) -> Scene:
        """The scene as the session currently stands, ready to be reopened."""
        self.logic.camera.publish()
        return scene_from_state(self.server.state, self.scene)

    def save(self, path) -> pl.Path:
        """Write the session out as a config file, and say where it went."""
        return toml.write(path, to_toml(self.scene_now()))

    def do(self, name: str, **arguments):
        """Ask for one action, and wait for whatever it started.

        Inside a state block, as a browser's call arrives: an action mostly
        writes state, and it is the flush that turns those writes into slices
        resampled and a preset dropped. Without one the action would go through
        and almost nothing would come of it.
        """
        self.ready()

        def act():
            with self.server.state:
                self.logic.dispatch(name, **arguments)

        until_settled(act)

    def run(self, actions) -> None:
        """Ask for a sequence of ``(name, arguments)`` in order.

        Each is settled before the next begins, so a script reads the way it
        runs.
        """
        for name, arguments in actions:
            self.do(name, **(arguments or {}))
