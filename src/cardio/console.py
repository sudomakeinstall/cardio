"""The action log, and the one syntax it is both written in and read from.

An action is a name and a dict of arguments -- what ``Registry.run`` binds and
what ``Session.run`` takes. That pair is the whole of what a console line means,
and rendering it back as the call that asks for it is what makes the log
something to copy rather than only something to read: ``format_call`` and
``parse_call`` are inverses, so a line the console printed is a line the prompt
accepts and a line a script may contain.

Nothing here knows about trame. ``Log`` is the contents of the console; whose
state variable it lands in is the controller's business.
"""

# System
import ast
import collections
import dataclasses as dc
import datetime as dt
import typing as ty

# Third Party
import pydantic_core

# How many lines the console keeps. A drag coalesces into one, so this is a
# count of things asked for rather than of events.
LIMIT = 500

# How close together two of the same action have to be to be one gesture. A
# drag dispatches around sixty times a second and two button presses cannot be
# anywhere near that, so this separates the two without either having to be
# named. Getting it wrong costs a line that reads as one thing done twice; it
# costs nothing in the script, which keeps every call either way.
COALESCE_WITHIN = dt.timedelta(seconds=0.5)

TIME_FORMAT = "%H:%M:%S"


def _now() -> dt.datetime:
    """Local wall-clock time, as the capture timestamps use."""
    return dt.datetime.now().astimezone()


def literal(value):
    """``value`` as something ``ast.literal_eval`` can read back.

    The same conversion the state registry makes on the way in, for the same
    reason: an enum argument is its value, not its repr, and a tuple is a list.
    """
    return pydantic_core.to_jsonable_python(value)


def format_call(name: str, arguments: dict) -> str:
    """One action as the call that asks for it."""
    spelled = ", ".join(f"{key}={literal(value)!r}" for key, value in arguments.items())
    return f"{name}({spelled})"


def _value(node, where: str):
    """One argument's value, which has to be a literal and nothing else."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        raise ValueError(
            f"{where} is {ast.unparse(node)!r}, which is not a plain value"
        ) from None


def parse_call(text: str) -> tuple[str, list, dict]:
    """A typed call as ``Registry.run`` takes it: name, positional, keyword.

    Deliberately not ``eval``. What is accepted is an action's name, on its own
    or applied to literal arguments, and every other thing Python's grammar
    allows in that position is refused by name -- so a mistyped command says
    what was wrong with it rather than running.
    """
    text = text.strip()
    if not text:
        raise ValueError("Nothing to run")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{text!r} is not a call: {exc.msg}") from None

    node = tree.body

    if isinstance(node, ast.Name):
        return node.id, [], {}

    if not isinstance(node, ast.Call):
        raise ValueError(f"{text!r} is not a call")

    if not isinstance(node.func, ast.Name):
        raise ValueError(
            f"{ast.unparse(node.func)!r} is not an action name; "
            "a call is a bare name applied to literal arguments"
        )

    name = node.func.id
    positional = [_value(arg, f"argument {i + 1}") for i, arg in enumerate(node.args)]

    keyword = {}
    for entry in node.keywords:
        if entry.arg is None:
            raise ValueError(f"{name} cannot be given ** arguments")
        keyword[entry.arg] = _value(entry.value, f"{entry.arg}")

    return name, positional, keyword


ACTION = "action"
ERROR = "error"


@dc.dataclass
class Entry:
    """One line of the log: what was asked for, when, and how many times.

    A line may stand for several calls. A gesture dispatches its action once
    per mouse move, and forty-seven lines saying the same thing is not a log of
    anything -- so they collapse into one, showing the latest and counting the
    rest. Every call is still kept: what collapses is the reading of them, not
    the record, and a script made from this replays the drag rather than its
    last twitch.
    """

    n: int
    seen: dt.datetime
    kind: str
    name: str = ""
    calls: list[dict] = dc.field(default_factory=list)
    message: str = ""

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def text(self) -> str:
        """The line as it reads: the latest call, or the refusal."""
        if self.kind == ERROR:
            return self.message
        return format_call(self.name, self.calls[-1])

    def joins(self, name: str, at: dt.datetime) -> bool:
        """Whether a call now is part of this line rather than the next one."""
        return (
            self.kind == ACTION
            and self.name == name
            and at - self.seen < COALESCE_WITHIN
        )

    def shown(self) -> dict:
        """The entry as the console renders it."""
        return {
            "n": self.n,
            "at": self.seen.strftime(TIME_FORMAT),
            "text": self.text,
            "kind": self.kind,
            "count": self.count,
        }


class Log:
    """The console's contents.

    Bounded, so a long session does not grow without end, and newest-first on
    the way out, which is the order the panel stacks them in.
    """

    def __init__(self, limit: int = LIMIT, now: ty.Callable[[], dt.datetime] = _now):
        self._entries: collections.deque[Entry] = collections.deque(maxlen=limit)
        self._now = now
        self._counted = 0

    def _add(self, kind: str, at: dt.datetime, **fields) -> Entry:
        self._counted += 1
        entry = Entry(n=self._counted, seen=at, kind=kind, **fields)
        self._entries.append(entry)
        return entry

    def record(self, name: str, arguments: dict) -> bool:
        """Write down one action, and say whether it started a new line.

        A repeat of the action on the line above, close enough behind it to be
        the same gesture, joins that line instead of starting one.
        """
        at = self._now()
        latest = self._entries[-1] if self._entries else None

        if latest is not None and latest.joins(name, at):
            latest.calls.append(dict(arguments))
            latest.seen = at
            return False

        self._add(ACTION, at, name=name, calls=[dict(arguments)])
        return True

    def error(self, message: str) -> None:
        """Write down something that was asked for and could not be done."""
        self._add(ERROR, self._now(), message=message)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entries(self) -> list[dict]:
        """Every line, newest first, as the console renders them."""
        return [entry.shown() for entry in reversed(self._entries)]

    @property
    def script(self) -> list[tuple[str, dict]]:
        """Every action in the order it happened, as ``Session.run`` takes them.

        A line that collapsed a drag contributes each of its calls, not the one
        it shows: what collapsed was the reading, and a script that dropped the
        rest would not put the app back where the drag left it. Errors are not
        in it, having done nothing to repeat.
        """
        return [
            (entry.name, dict(arguments))
            for entry in self._entries
            if entry.kind == ACTION
            for arguments in entry.calls
        ]

    def __len__(self) -> int:
        return len(self._entries)
