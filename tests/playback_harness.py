"""A virtual-clock harness for the cine playback loop.

``_play_loop`` is asynchronous and timing-dependent, so the tests drive it on a
virtual clock: ``playback.time`` and ``playback.asyncio`` are swapped for shims,
so a sleep advances the clock instead of the wall and a render charges its cost
to the same clock. The loop then runs at full speed and deterministically.

The sleep shim also caps how many times the loop may await. That cap is the
runaway detector -- a loop that will not stop raises ``Runaway`` rather than
hanging the suite.
"""

# System
import asyncio

# Internal
from cardio.image_quality import (
    DEFAULT_PLAYBACK_QUALITY,
    DEFAULT_PLAYBACK_RESOLUTION,
)
from cardio.logic import playback as playback_module
from cardio.logic.playback import PlaybackController

# The constant _play_loop uses for its pause-check granularity.
CHECK_INTERVAL = 0.01


class Runaway(RuntimeError):
    """The loop awaited more times than the scenario allowed."""


class Clock:
    """Virtual time. Sleeps and renders both advance it."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


class SleepShim:
    """Stands in for ``asyncio`` inside the playback module."""

    CancelledError = asyncio.CancelledError

    def __init__(self, clock: Clock, budget: int):
        self.clock = clock
        self.budget = budget
        self.count = 0

    async def sleep(self, seconds):
        self.count += 1
        if self.count > self.budget:
            raise Runaway(f"the loop awaited more than {self.budget} times")
        self.clock.advance(seconds)
        await asyncio.sleep(0)


class FakeState:
    """trame's state semantics, as far as the playback loop depends on them.

    Assignment marks a key pending; ``flush`` fires the change listeners for
    the pending keys and is a no-op while already flushing, which is what
    trame's ``skip_flushing`` does.
    """

    def __init__(self, **initial):
        self.__dict__["_data"] = dict(initial)
        self.__dict__["_callbacks"] = {}
        self.__dict__["_pending"] = set()
        self.__dict__["_flushing"] = False
        self.__dict__["flushes"] = 0

    def __getattr__(self, name):
        try:
            return self.__dict__["_data"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        data = self.__dict__["_data"]
        if name not in data or data[name] != value:
            self.__dict__["_pending"].add(name)
        data[name] = value

    def __getitem__(self, key):
        return self.__dict__["_data"][key]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def change(self, *keys):
        def register(func):
            for key in keys:
                self.__dict__["_callbacks"].setdefault(key, []).append(func)
            return func

        return register

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.flush()

    def flush(self):
        if self.__dict__["_flushing"]:
            return
        self.__dict__["flushes"] += 1
        self.__dict__["_flushing"] = True
        try:
            while self.__dict__["_pending"]:
                keys = self.__dict__["_pending"]
                self.__dict__["_pending"] = set()
                for key in sorted(keys):
                    for func in self.__dict__["_callbacks"].get(key, []):
                        func(**self.__dict__["_data"])
        finally:
            self.__dict__["_flushing"] = False


class FakeController:
    """``view_update`` records the render and charges it to the clock."""

    def __init__(self, clock: Clock, render_seconds: float, app):
        self._clock = clock
        self._render_seconds = render_seconds
        self._app = app
        self.renders = []

    def view_update(self, **kwargs):
        self.renders.append(
            {
                "time": self._clock.now,
                "frame": self._app.server.state.frame,
                "shown": self._app.mpr.last_frame,
                "rendering": self._app.playback._is_rendering,
            }
        )
        self._clock.advance(self._render_seconds)


class FakeCamera:
    def __init__(self):
        self.azimuths = []

    def Azimuth(self, degrees):
        self.azimuths.append(degrees)


class FakeRenderer:
    def __init__(self):
        self.camera = FakeCamera()

    def GetActiveCamera(self):
        return self.camera


class FakeScene:
    def __init__(self, nframes: int):
        self.nframes = nframes
        self.renderables = []
        self.renderer = FakeRenderer()
        self.hides = 0

    def hide_all_frames(self):
        self.hides += 1


class RecordingSnap:
    def __init__(self):
        self.locks = []

    def apply_frame_lock(self, frame):
        self.locks.append(frame)


class RecordingMPR:
    """Stands in for the MPR controller, and remembers the frame it was shown.

    ``last_frame`` starts at the scene's opening frame: the app renders once on
    startup, so the views are already showing it before playback begins.
    """

    def __init__(self, initial_frame: int):
        self.frames = []
        self.last_frame = initial_frame

    def update_mpr_frame(self, frame):
        self.frames.append(frame)
        self.last_frame = frame


class FakeServer:
    """``protocol`` is None until a client connects, as on a real server."""

    def __init__(self, state, controller):
        self.name = "playback-harness"
        self.state = state
        self.controller = controller
        self.protocol = None


class PlaybackApp:
    """A real PlaybackController over fakes, wired the way Logic wires it."""

    def __init__(self, clock, nframes=10, render_seconds=0.0, **overrides):
        state_values = {
            "frame": 0,
            "playing": False,
            "incrementing": True,
            "rotating": False,
            "bpm": 60,
            "bpr": 3,
            "playback_quality": DEFAULT_PLAYBACK_QUALITY,
            "playback_resolution": DEFAULT_PLAYBACK_RESOLUTION,
        }
        state_values.update(overrides)

        self.scene = FakeScene(nframes)
        self.snap = RecordingSnap()
        self.mpr = RecordingMPR(state_values["frame"])
        controller = FakeController(clock, render_seconds, self)
        self.server = FakeServer(FakeState(**state_values), controller)
        self.playback = PlaybackController(self)
        self.playback.register()

    @property
    def renders(self):
        return self.server.controller.renders

    @property
    def state(self):
        return self.server.state


def install_shims(monkeypatch, clock, budget):
    """Point the playback module at the virtual clock and the capped sleep."""
    shim = SleepShim(clock, budget)
    monkeypatch.setattr(playback_module, "asyncio", shim)
    monkeypatch.setattr(playback_module, "time", clock)
    return shim


async def pump(condition, budget=20000):
    """Yield to the event loop until ``condition`` holds, or give up."""
    for _ in range(budget):
        if condition():
            return True
        await asyncio.sleep(0)
    return False


async def drain(task):
    """Let ``task`` finish, surfacing anything but cancellation."""
    try:
        await task
    except asyncio.CancelledError:
        pass
