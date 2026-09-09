"""The log of what the app was asked to do, and the prompt that asks it.

The console is a reader of the action registry rather than a part of it: it
observes every call, renders it as the call that would ask for it again, and
takes one back. Nothing else in the app knows it is there.

It watches through ``Registry.observe`` and not through the journal. The
journal answers what an action *moved*, and copies every document key twice to
find out; a drag dispatches once per mouse move, and the console has to be
armed for the whole session. What it needs is only the name and the arguments,
which cost nothing to be told.
"""

# System
import contextlib as cl
import copy
import datetime as dt
import time
import typing as ty

# Third Party
import pydantic as pc

# Internal
from .. import keypath, scripting
from ..action import action
from ..console import TIME_FORMAT, Log, parse_call
from .base import Controller

# The actions that are about the console rather than about the app. A typed
# command logs the action it asked for, exactly as the button that does the
# same thing would; logging the wrapper as well would say everything twice.
# Saving the log is not a thing anybody means to replay either.
SILENT = frozenset({"run_command", "clear_console", "save_script"})

# How often a coalescing drag republishes. A new line goes out at once; a line
# whose count is merely climbing can wait, and waiting is what keeps a gesture
# from writing the whole list sixty times a second.
PUBLISH_INTERVAL = 0.2

# The action a document key moving by itself is written down as, and the one a
# script uses to move it back.
SET_STATE = "set_state"

# Keys nobody ever asks for, which move as a consequence of something else.
# The state's cameras are only ever a note of where VTK's ended up -- refitted
# when the layout changes, turned by a trackball drag -- and the one time a
# person means to move one, place_camera writes it down as the action it was.
FOLLOWS = frozenset({"cameras"})

# Keys a running loop drives, and what says the loop is running. A cine writes
# the frame thirty times a second and nobody asked for any of them: what was
# asked for was to play, and that is already a line of its own.
DRIVEN = {"frame": ("playing", "capture_running")}

# Keys that another key's change rewrites wholesale, and what rewrites them.
# Switching the units or the index order re-expresses every angle in the
# sequence; that switch is already a line, and the rewrite is what the line
# means rather than a second thing anybody asked for.
#
# The rewrite is a listener's doing, so it lands in the flush after the one that
# set it off rather than in the same one -- which is why what was written down
# last time counts here as well as what moved this time.
CONSEQUENCE = {"mpr_rotation_data": ("angle_units", "index_order")}


def _message(exc: Exception) -> str:
    """What a refusal says, without the quotes ``KeyError`` prints around it."""
    if isinstance(exc, KeyError):
        return str(exc.args[0])
    return str(exc)


class ConsoleController(Controller):
    """The action log, and running a typed call against the registry."""

    seeds = ("console_visible",)

    def __init__(self, app):
        super().__init__(app)
        self.log = Log()
        self._published = 0.0
        self._applying = 0
        self._armed = False
        self._seen: dict[str, ty.Any] = {}
        self._written = set()

    def register(self):
        self.app.actions.observe(self._observed)

        # A log gathered while the console was shut is still a log of what
        # happened, so opening it publishes what it missed.
        self.server.state.change("console_visible")(self.publish)

        # Most of the drawer binds its state directly, so most of what a person
        # does reaches no action at all. Watching the keys rather than the
        # widgets is what catches those -- and catches the per-object ones,
        # which no widget names, for nothing extra.
        self.server.state.change(*self.app.document_keys())(self._document_moved)

        # Bringing the app up writes the whole document, and none of it is
        # anybody's doing. This is what both a page and a bare session run last
        # once everything is up, and an added function runs after the one it
        # was added to -- so by the time the log starts, the startup writes
        # have been and gone.
        self.server.controller.finalize_mpr_initialization.add(self._arm)

    def seed(self):
        super().seed()
        self.server.state.console_entries = []
        self.server.state.console_input = ""
        self.server.state.script_saved_at = None
        self.server.state.script_summary = ""

    @property
    def showing(self) -> bool:
        """Whether anybody is looking.

        Nothing is published while the dock is shut: the log fills in python
        either way, and the state variable is what costs a round trip.
        """
        return bool(getattr(self.server.state, "console_visible", False))

    def publish(self, **kwargs):
        """Put the log where the page can read it."""
        if not self.showing:
            return

        self._published = time.monotonic()
        with self.server.state as state:
            state.console_entries = self.log.entries

    @cl.contextmanager
    def _observed(self, name: str, arguments: dict):
        if name in SILENT:
            yield
            return

        self._applying += 1
        went_through = True
        try:
            # The action's own writes are flushed in here, so that the listener
            # below sees them while this is standing and knows to leave them
            # alone: they are the action about to be written down, not
            # something done to the app behind its back.
            with self.server.state:
                yield
        except Exception:
            went_through = False
            raise
        finally:
            self._applying -= 1

            # Written down on the way out rather than on the way in, so that a
            # call which raised is not left looking like one that happened --
            # and, which matters more, is not left in the script.
            started = self.log.record(name, arguments, ok=went_through)
            if started or time.monotonic() - self._published >= PUBLISH_INTERVAL:
                self.publish()

    def _arm(self, **kwargs):
        """Start recording: what came before was the app being built."""
        self._armed = True
        self._remember(self.app.document_keys())

    def _asked_for(self, key: str, moved: set[str]) -> bool:
        """Whether ``key`` moving is a thing anybody asked for."""
        if key in FOLLOWS:
            return False
        if moved & set(CONSEQUENCE.get(key, ())):
            return False
        return not any(self.server.state[guard] for guard in DRIVEN.get(key, ()))

    def _remember(self, keys) -> None:
        """Keep what a later change to ``keys`` will be compared against.

        Only the keys a path could name something inside, which is one small
        dict today. Copying every document key is what the journal does and
        what this must not: the console is armed for the whole session, and a
        drag reaches it once per mouse move.
        """
        for key in keys:
            value = self.server.state[key]
            if keypath.nested(value):
                self._seen[key] = copy.deepcopy(value)
            else:
                self._seen.pop(key, None)

    def _calls(self, key: str) -> list[tuple[str, ty.Any]]:
        """What to write down about ``key`` having moved.

        The whole of it, unless there is a previous value to compare against
        and both are structures -- in which case the parts that moved, each
        named by where it sits, which is a line short enough to read and to
        type back.
        """
        value = self.server.state[key]
        if key not in self._seen:
            return [(key, value)]
        return keypath.changes(self._seen[key], value, key)

    def _document_moved(self, **state):
        """Write down what moved with nobody having asked for it by name.

        A widget bound straight to a document key is most of the drawer, and
        every one of those writes is something the user did that no action saw.
        Recorded as the action that would do it again, so that a log of a
        session driven entirely from the drawer still replays.
        """
        moved = set(self.server.state.modified_keys) & set(self.app.document_keys())

        # Re-based even when nothing is written down, so that the next change a
        # person makes is compared with what an action left rather than with
        # whatever was there before it ran.
        if not self._armed or self._applying:
            self._remember(moved)
            self._written = set()
            return

        started = False
        written = set()
        for key in sorted(moved):
            if not self._asked_for(key, moved | self._written):
                continue
            for path, value in self._calls(key):
                started |= self.log.record(
                    SET_STATE,
                    {"key": path, "value": value},
                    group=f"{SET_STATE}:{path}",
                )
            written.add(key)

        self._remember(moved)
        self._written = written

        if started or time.monotonic() - self._published >= PUBLISH_INTERVAL:
            self.publish()

    @action("toggle_console")
    def toggle_console(self):
        """Open or close the action console."""
        state = self.server.state
        state.console_visible = not state.console_visible

    @action(SET_STATE)
    def set_state(self, key: str, value: ty.Any):
        """Put one document key, or one place inside it, where it is asked to be.

        The way back in for everything the drawer does by binding state
        directly. The widgets still write their keys themselves -- what this is
        for is that the line the console wrote about it is a line that can be
        run, which is the whole of what makes a log a script.

        ``key`` may name a place within the value rather than the whole of it:
        one rotation's visibility is a thing to ask for, and the sequence it
        sits in is not a thing to have to spell out to ask.

        Confined to the document: session state is not a thing a script has any
        business reaching into, and a mistyped key should say so.
        """
        name, segments = keypath.split(key)
        if name not in self.app.document_keys():
            raise ValueError(f"{name!r} is not a document key")

        if segments:
            value = keypath.write(self.server.state[name], segments, value, name)
        self.server.state[name] = value

    @action("clear_console")
    def clear_console(self):
        """Throw away the log so far."""
        self.log.clear()
        self.publish()

    @action("save_script")
    def save_script(self):
        """Write the log out as a script that does the same session again.

        Two files: the calls, and the scene they are asked of. The scene is the
        one the app opened with rather than the one it is showing -- an action
        moves the scene under it, so a script replayed against where it ended
        up would do everything a second time.
        """
        recorded = dt.datetime.now().astimezone()
        actions = self.log.script
        path = scripting.save(
            self.scene.scripts_directory,
            recorded.strftime(self.scene.timestamp_format),
            actions,
            self.app.opened_as,
            recorded,
            self.log.dropped,
        )

        with self.server.state as state:
            state.script_saved_at = recorded.strftime(TIME_FORMAT)
            state.script_summary = f"{len(actions)} calls to {path.name}"

    @action("run_command")
    def run_command(self, text: str):
        """Do what one typed call asks for, or say why it cannot be done.

        The command is not itself logged -- the action it asks for is, in the
        same line a button press would have written. What a bad one leaves
        behind is the refusal, which the registry and the argument model
        already word better than this could.
        """
        try:
            name, positional, keyword = parse_call(text)
            self.app.dispatch(name, *positional, **keyword)
        except (ValueError, TypeError, KeyError, pc.ValidationError) as exc:
            self.log.error(_message(exc))
            self.publish()
        else:
            self.server.state.console_input = ""
