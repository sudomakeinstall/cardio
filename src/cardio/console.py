"""The action log, and the one syntax it is both written in and read from.

An action is a name and a dict of arguments -- what ``Registry.run`` binds and
what ``Session.run`` takes. That pair is the whole of what a console line means,
and rendering it back as the call that asks for it is what makes the log
something to copy rather than only something to read: ``format_call`` and
``parse_call`` are inverses, so a line the console printed is a line the prompt
accepts and a line a script may contain -- the same characters in all three,
``do.`` included.

Nothing here knows about trame. ``Log`` is the contents of the console and
``History`` is where the prompt is while it walks back through them; whose
state variables they land in is the controller's business.
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

# What a call is asked of, in the console and in a generated script alike. The
# console prints it because a line is there to be copied, and a line that has
# to be edited before it will run is one the copying did not finish.
PROXY = "do"
PREFIX = f"{PROXY}."


def _now() -> dt.datetime:
    """Local wall-clock time, as the capture timestamps use."""
    return dt.datetime.now().astimezone()


def literal(value):
    """``value`` as something ``ast.literal_eval`` can read back.

    The same conversion the state registry makes on the way in, for the same
    reason: an enum argument is its value, not its repr, and a tuple is a list.
    """
    return pydantic_core.to_jsonable_python(value)


def format_call(name: str, arguments: dict, prefix: str = "") -> str:
    """One action as the call that asks for it.

    ``prefix`` is what the call is asked of. The console and a script both
    write ``PREFIX``, so a line of one is a line of the other and a log is
    copied into a script rather than translated into one. It defaults to
    nothing for the sake of what only needs to read the call itself.
    """
    spelled = ", ".join(f"{key}={literal(value)!r}" for key, value in arguments.items())
    return f"{prefix}{name}({spelled})"


def _value(node, where: str):
    """One argument's value, which has to be a literal and nothing else."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        raise ValueError(
            f"{where} is {ast.unparse(node)!r}, which is not a plain value"
        ) from None


def _named(node) -> str | None:
    """The action ``node`` names, whether or not it is asked of the proxy.

    ``do.place_camera`` and ``place_camera`` name the same action. The first is
    what the console prints and what a script holds; the second is what is
    quicker to type. Taking both is what lets a printed line be pasted back
    without being edited first.
    """
    if isinstance(node, ast.Name):
        return node.id
    proxied = isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    if proxied and node.value.id == PROXY:
        return node.attr
    return None


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

    bare = _named(node)
    if bare is not None:
        return bare, [], {}

    if not isinstance(node, ast.Call):
        raise ValueError(f"{text!r} is not a call")

    name = _named(node.func)
    if name is None:
        raise ValueError(
            f"{ast.unparse(node.func)!r} is not an action name; "
            "a call is a bare name applied to literal arguments"
        )

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
    group: str = ""
    calls: list[dict] = dc.field(default_factory=list)
    message: str = ""
    asked: str = ""
    ok: bool = True

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def text(self) -> str:
        """The line as it reads: the latest call, or the refusal.

        A refusal is not asked of anything, so it carries no prefix: it is what
        the app said back, rather than a line there would be any sense in
        running.
        """
        if self.kind == ERROR:
            return self.message
        return format_call(self.name, self.calls[-1], prefix=PREFIX)

    @property
    def recallable(self) -> str:
        """The line the prompt would be given back, or nothing.

        A call is recalled as it was printed, a call that raised included: what
        it did not do is a reason to run it again rather than a reason not to.
        A refusal is recalled as it was asked rather than as it was answered --
        what there is any point arrowing back to is the command, not the
        complaint about it.
        """
        return self.asked if self.kind == ERROR else self.text

    def joins(self, group: str, at: dt.datetime) -> bool:
        """Whether a call now is part of this line rather than the next one.

        A call that raised joins nothing and is joined by nothing: it is the
        one thing on the line that did not happen, and burying it in a count
        beside forty that did would misrepresent both.
        """
        return (
            self.kind == ACTION
            and self.ok
            and self.group == group
            and at - self.seen < COALESCE_WITHIN
        )

    def shown(self) -> dict:
        """The entry as the console renders it."""
        return {
            "n": self.n,
            "at": self.seen.strftime(TIME_FORMAT),
            "text": self.text,
            "kind": self.kind if self.ok else ERROR,
            "count": self.count,
        }


class Log:
    """The console's contents.

    Bounded, so a long session does not grow without end, and in the order
    things happened on the way out -- which is the order the panel reads in,
    the order a script runs in, and the order a selection follows.
    """

    def __init__(self, limit: int = LIMIT, now: ty.Callable[[], dt.datetime] = _now):
        self._entries: collections.deque[Entry] = collections.deque(maxlen=limit)
        self._now = now
        self._counted = 0
        self._dropped = 0

    def _add(self, kind: str, at: dt.datetime, **fields) -> Entry:
        if len(self._entries) == self._entries.maxlen:
            self._dropped += 1
        self._counted += 1
        entry = Entry(n=self._counted, seen=at, kind=kind, **fields)
        self._entries.append(entry)
        return entry

    def record(
        self, name: str, arguments: dict, group: str = "", ok: bool = True
    ) -> bool:
        """Write down one action, and say whether it started a new line.

        A repeat of the line above, close enough behind it to be the same
        gesture, joins that line instead of starting one.

        What counts as a repeat is ``group``, which is the action's name unless
        the caller knows better. ``set_state`` is why it can: one action name
        now stands for every document key, and dragging a slider is not the
        same gesture as the frame ticking behind it.
        """
        at = self._now()
        group = group or name
        latest = self._entries[-1] if self._entries else None

        if latest is not None and latest.joins(group, at):
            latest.calls.append(dict(arguments))
            latest.seen = at
            return False

        self._add(ACTION, at, name=name, group=group, calls=[dict(arguments)], ok=ok)
        return True

    def error(self, message: str, asked: str = "") -> None:
        """Write down something that was asked for and could not be done.

        ``asked`` is what was typed, which the message does not hold and the
        prompt wants back: a command refused is the one most worth arrowing to,
        there being something in it to fix.
        """
        self._add(ERROR, self._now(), message=message, asked=asked)

    def clear(self) -> None:
        self._dropped += len(self._entries)
        self._entries.clear()

    @property
    def dropped(self) -> int:
        """How many lines the log no longer holds, whether cleared or aged out.

        A script is only what the log still has. Once anything has left it --
        pushed off the far end, or thrown away -- the script no longer starts
        where the session started, and something that says so is better than a
        file that looks complete.
        """
        return self._dropped

    @property
    def entries(self) -> list[dict]:
        """Every line, oldest first, as the console renders them.

        The order they happened in, which is the order they are read in. The
        panel puts the newest at the bottom without reversing this: what is
        reversed there is the box, so that a selection dragged down the log
        still runs the way the lines do.
        """
        return [entry.shown() for entry in self._entries]

    @property
    def recallable(self) -> list[str]:
        """Every line the prompt could be given back, oldest first.

        The whole log rather than the part of it that was typed: a drawer
        control is written down as the call that would move it again, which
        makes it as good a thing to arrow back to as anything typed was.
        """
        return [line for entry in self._entries if (line := entry.recallable)]

    @property
    def script(self) -> list[tuple[str, dict]]:
        """Every action in the order it happened, as ``Session.run`` takes them.

        A line that collapsed a drag contributes each of its calls, not the one
        it shows: what collapsed was the reading, and a script that dropped the
        rest would not put the app back where the drag left it. Neither a
        refusal nor a call that raised is in it, both having done nothing that
        there is any point repeating.
        """
        return [
            (entry.name, dict(arguments))
            for entry in self._entries
            if entry.kind == ACTION and entry.ok
            for arguments in entry.calls
        ]

    def __len__(self) -> int:
        return len(self._entries)


def matching(lines: list[str], typed: str) -> list[str]:
    """The lines a walk from ``typed`` steps through, oldest first.

    Narrowed the way a terminal narrows: what is already typed is the start of
    what is wanted, and nothing else is worth stepping past. Typed nothing, the
    walk is every line, which is the plain walk back.

    A line matches what it starts with, and also what it starts with once the
    prefix is off the front of it -- the prompt takes a bare name, so a bare
    name has to find the lines the console printed under ``do.``.

    Identical lines collapse to the most recent of them. A log is mostly
    repetition, and a walk that stops eleven times at the same text is a walk
    nobody finishes; what collapses is the repeats, never a line saying
    something the walk would not otherwise reach.
    """
    found = [
        line
        for line in lines
        if line.startswith(typed) or line.removeprefix(PREFIX).startswith(typed)
    ]
    return list(reversed(list(dict.fromkeys(reversed(found)))))


class History:
    """Where the prompt is while it is being walked backwards.

    A walk is a prefix and the line last handed over; where it is among the
    lines matching that prefix is looked up rather than kept, because the log
    is not what a shell's history is. A shell's grows only when a command is
    run, and the prompt sits still between. This one grows while the prompt
    sits there -- every drawer control is a line -- so a walk holding the
    matches it started with would be walking a log the app has moved on from.

    A walk lasts until something is typed, which is how editing a recalled line
    searches for the edit rather than going on with the search it interrupted.
    What says something was typed is that the prompt no longer holds what was
    last handed to it: that needs no listening to keystrokes, and is true of
    every way the box can change under a walk.
    """

    def __init__(self):
        self._prefix = ""
        self._lines: list[str] | None = None
        self._handed: str | None = None

    def forget(self) -> None:
        """End the walk, so the next arrow starts a new one."""
        self._handed = None

    def walk(self, lines: list[str], typed: str, step: int) -> str:
        """What the prompt should hold after one press of an arrow.

        ``step`` is -1 for a step back through the log and 1 for a step toward
        the present. The position runs over the matches and one place past the
        newest of them, which is where the walk started: stepping off that end
        hands back what was being typed rather than leaving the box holding the
        last thing recalled. Both ends stop rather than wrap, and a walk with
        nothing to show leaves the prompt exactly as it was.

        The log moving on ends the walk, as running a command does. What is
        left in the box then is a recall rather than anything anybody typed, so
        it is not what the next walk searches for: the whole log is, which is
        what an arrow after doing something is asking for.
        """
        moved_on = lines != self._lines
        walking = typed == self._handed and not moved_on

        if not walking:
            self._prefix = "" if typed == self._handed else typed

        self._lines = lines
        matches = matching(lines, self._prefix)

        at = matches.index(typed) if walking and typed in matches else len(matches)
        at = min(len(matches), max(0, at + step))

        self._handed = matches[at] if at < len(matches) else self._prefix
        return self._handed
