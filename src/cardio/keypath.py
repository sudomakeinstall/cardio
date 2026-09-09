"""A place inside a document key, and the difference between two of them.

Most document keys hold one value, and saying what one of those became is the
whole of what the console has to write down. A few hold a document of their own
-- the rotation sequence is a dict of a dict and a list of dicts -- and saying
what *those* became means repeating the entire structure to report the one
field that moved.

A path is what lets the log name the field instead. Dotted, with a numeric
segment indexing a list, which is how the dotted ``Scene`` field paths in
``registry`` already read.

``changes`` is the half that decides how far in to go. Descending into
everything would be worse, not better: a range slider holds two floats, and two
lines saying which end moved is a worse account of a drag than one line holding
the pair. So a path stops where the structure stops being a structure.
"""

# System
import copy


def split(key: str) -> tuple[str, list[str]]:
    """A path as the document key it starts at, and the way in from there.

    A bare key has no way in, which is what every key but one is.
    """
    name, _, rest = key.partition(".")
    return name, rest.split(".") if rest else []


def _index(container, segment: str, where: str):
    """``segment`` as something ``container`` can be subscripted by.

    Refuses by saying what was actually there, which is the difference between
    a path that can be corrected and a ``KeyError`` with a name in it.
    """
    if isinstance(container, dict):
        if segment not in container:
            spelled = ", ".join(sorted(map(str, container))) or "nothing"
            raise ValueError(f"{where} has no {segment!r}; it holds {spelled}")
        return segment

    if isinstance(container, list):
        try:
            position = int(segment)
        except ValueError:
            raise ValueError(
                f"{where} is a list, and {segment!r} is not a position in one"
            ) from None
        if not -len(container) <= position < len(container):
            raise ValueError(
                f"{where} has {len(container)} entries, so there is no {segment}"
            )
        return position

    raise ValueError(f"{where} is {container!r}, which nothing is inside of")


def _describe(name: str, segments: list[str], depth: int) -> str:
    """How far in a refusal happened, spelled as the path that got there."""
    return ".".join([name, *segments[:depth]])


def read(value, segments: list[str], name: str = ""):
    """What sits at ``segments`` inside ``value``."""
    for depth, segment in enumerate(segments):
        value = value[_index(value, segment, _describe(name, segments, depth))]
    return value


def write(value, segments: list[str], new, name: str = ""):
    """``value`` with ``new`` at ``segments``, as a structure of its own.

    Copied rather than written through: trame is told a key moved by being
    handed a different object, and the console holds the last value it saw to
    compare the next one against. Writing in place would defeat both.
    """
    if not segments:
        return new

    where = _index(value, segments[0], name)
    copied = copy.copy(value)
    copied[where] = write(value[where], segments[1:], new, _describe(name, segments, 1))
    return copied


def _walkable(one, other) -> bool:
    """Whether a path may go further in, or has reached what it came to say.

    Two containers of the same shape are walkable; a scalar is not, and neither
    is a list of them. The pair of floats behind a range slider is one value a
    person dragged, not two values they set, and the log should say so.
    """
    if isinstance(one, dict) and isinstance(other, dict):
        return set(one) == set(other)

    if isinstance(one, list) and isinstance(other, list):
        return len(one) == len(other) and all(
            isinstance(item, (dict, list)) for item in one
        )

    return False


def nested(value) -> bool:
    """Whether a path could ever name something inside ``value``.

    What decides whether the console keeps a copy of a key to compare the next
    one against. The same rule ``_walkable`` applies, seen from one side.
    """
    if isinstance(value, dict):
        return True
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, (dict, list)) for item in value)
    )


def same(one, other) -> bool:
    """Whether two state values are the same one.

    Some of them are arrays, which do not answer ``==`` with a bool. Lives here
    rather than beside the journal that also asks: this module is the one that
    knows nothing else, so both can reach it.
    """
    if one is other:
        return True
    try:
        return bool(one == other)
    except (ValueError, TypeError):
        return repr(one) == repr(other)


def changes(before, after, name: str = "") -> list[tuple[str, object]]:
    """The fewest ``(path, value)`` pairs that say how ``before`` became ``after``.

    One pair naming the whole thing when it is not something to go inside of,
    and otherwise one per part of it that moved. Nothing at all when nothing
    did.
    """
    if not _walkable(before, after):
        return [] if same(before, after) else [(name, after)]

    keys = after if isinstance(after, dict) else range(len(after))
    found = []
    for key in keys:
        found.extend(
            changes(before[key], after[key], f"{name}.{key}" if name else str(key))
        )
    return found
