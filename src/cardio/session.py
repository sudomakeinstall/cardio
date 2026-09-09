"""The app without a page: a scene, its logic, and a way to ask it for things.

``CardioApp`` is this plus a browser. Everything the app can do it can do here,
because the actions and the state were never the page's -- the page only ever
pressed the buttons. What a session adds is the two things a browser was
quietly providing: something to arm the change listeners, somewhere for an
action that goes to the background to run, and something to draw.
"""

# System
import asyncio
import functools as ft
import pathlib as pl

# Third Party
import trame as tm
import trame.app

# Internal
from . import toml
from .document import scene_from_state, to_toml
from .logic import Logic
from .scene import Scene
from .view import RENDER_VIEWS, VIEW_FUNCTIONS

# The views a session draws, named as the scene names their windows rather
# than as the controller names the function that draws each.
DRAWN_VIEWS = tuple(name.removesuffix("_update") for name in RENDER_VIEWS)


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

        self.bind_views()

    def bind_views(self):
        """Draw what a page would have drawn, there being no page to draw it.

        ``VtkRemoteView.update`` renders its window and sends the picture on.
        With nobody assigning these they were left empty, so nothing ever
        rendered: a capture read a frame buffer that had never been drawn into
        and wrote a picture of nothing, one file per viewport, all of them
        black. What a session wants of those functions is the first half.

        ``view_reset_camera`` is the half that is only a page's -- the app
        poses its own cameras -- and stays empty, which trame otherwise raises
        on rather than passing quietly.
        """
        controller = self.server.controller
        for name, view in zip(RENDER_VIEWS, DRAWN_VIEWS):
            setattr(controller, name, ft.partial(self.render, view))
        controller.view_update = self.render

        for name in set(VIEW_FUNCTIONS) - set(RENDER_VIEWS) - {"view_update"}:
            getattr(controller, name).can_be_empty = True

    def render(self, *views, **kwargs):
        """Draw the named views, or every one that has been built.

        Called with whatever trame hands a controller function, which for the
        ones that are also change listeners is the whole state.
        """
        for view in views or DRAWN_VIEWS:
            window = self.window(view)
            if window is not None:
                window.Render()

    def window(self, view: str):
        """The render window ``view`` draws into, or None before it is built."""
        if view == "volume":
            return self.scene.renderWindow
        if view == "tile":
            return self.scene.tile_views.window if self.scene.tile_views else None
        return self.scene.mpr_views[view] if self.scene.mpr_views else None

    @classmethod
    def from_config(cls, config_file: pl.Path | str, server=None, **overrides):
        """A session on the scene a TOML config describes."""
        return cls(Scene.load(config_file=config_file, **overrides), server=server)

    @property
    def actions(self):
        """Everything this session can be asked to do."""
        return self.logic.actions

    @property
    def opened_as(self) -> str:
        """The config that opens the scene as this session started it."""
        return self.logic.opened_as

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
        self.size_windows()
        self.server.state.ready()
        self.server.controller.finalize_mpr_initialization()

    def size_windows(self):
        """Give every window a size, since no browser is going to.

        A page sizes its render windows over the wire as it lays them out, and
        without one they stay at nothing by nothing -- which renders, and
        captures, an image zero pixels wide. Done before the state is armed so
        that the fits which follow measure the window a script will actually
        get.
        """
        for view in DRAWN_VIEWS:
            self.window(view).SetSize(*self.scene.headless_size)

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
