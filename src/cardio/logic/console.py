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
import time

# Third Party
import pydantic as pc

# Internal
from ..action import action
from ..console import Log, parse_call
from .base import Controller

# The actions that are about the console rather than about the app. A typed
# command logs the action it asked for, exactly as the button that does the
# same thing would; logging the wrapper as well would say everything twice.
SILENT = frozenset({"run_command", "clear_console"})

# How often a coalescing drag republishes. A new line goes out at once; a line
# whose count is merely climbing can wait, and waiting is what keeps a gesture
# from writing the whole list sixty times a second.
PUBLISH_INTERVAL = 0.2


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

    def register(self):
        self.app.actions.observe(self._observed)

        # A log gathered while the console was shut is still a log of what
        # happened, so opening it publishes what it missed.
        self.server.state.change("console_visible")(self.publish)

    def seed(self):
        super().seed()
        self.server.state.console_entries = []
        self.server.state.console_input = ""

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

    def _observed(self, name: str, arguments: dict):
        if name in SILENT:
            return

        started = self.log.record(name, arguments)
        if started or time.monotonic() - self._published >= PUBLISH_INTERVAL:
            self.publish()

    @action("toggle_console")
    def toggle_console(self):
        """Open or close the action console."""
        state = self.server.state
        state.console_visible = not state.console_visible

    @action("clear_console")
    def clear_console(self):
        """Throw away the log so far."""
        self.log.clear()
        self.publish()

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
