"""The things a user can do, named, so that something other than a button can do them.

An action used to be a bound method assigned onto trame's controller under
whatever name the widget that called it happened to use. That is enough for a
button and for nothing else: there was no way to ask what the app can do, no
signature to check an argument against, and no single place a call passes
through. A script wants the first two and undo wants the third.

The methods have not moved. What is new is that each one says its name where it
is defined, and that ``Registry`` is the one thing that calls it.
"""

# System
import contextlib as cl
import copy
import dataclasses as dc
import functools as ft
import inspect
import typing as ty

# Third Party
import pydantic as pc
from trame.app import asynchronous

# Internal
from .keypath import same


def action(name: str):
    """Declare a controller method as the named action ``name``."""

    def mark(method):
        method._action_name = name
        return method

    return mark


def background(method):
    """Run an async action as a task, keeping the signature it was written with.

    ``trame.app.asynchronous.task`` returns a bare ``*args, **kwargs`` wrapper,
    which would tell the registry that every async action takes no arguments.
    """

    @ft.wraps(method)
    def run(*args, **kwargs):
        asynchronous.create_task(method(*args, **kwargs))

    return run


def parameter_model(name: str, method) -> type[pc.BaseModel]:
    """A model of what ``method`` may be called with.

    Built from the signature rather than declared separately, so the two cannot
    disagree. ``**kwargs`` is skipped: several actions are change listeners as
    well, and trame hands those the whole state.

    Extras are forbidden, as they are on ``Scene``: a misspelled argument in a
    hand-written script should say so rather than quietly do nothing.
    """
    fields = {}
    for parameter in inspect.signature(method).parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        fields[parameter.name] = (
            ty.Any if parameter.annotation is parameter.empty else parameter.annotation,
            ... if parameter.default is parameter.empty else parameter.default,
        )

    return pc.create_model(
        f"{name}_arguments", __config__=pc.ConfigDict(extra="forbid"), **fields
    )


@dc.dataclass(frozen=True)
class Action:
    """One named thing the app can be asked to do."""

    name: str
    call: ty.Callable
    arguments: type[pc.BaseModel]

    def bind(self, positional, keyword) -> dict:
        """The call's arguments by name, checked against the model.

        A widget passes them positionally and a script passes them by name, and
        what is recorded should not depend on which. Only the positions are
        resolved here; everything else is left to the model, which says what
        was expected rather than merely that something was wrong.
        """
        names = list(self.arguments.model_fields)
        if len(positional) > len(names):
            raise TypeError(
                f"{self.name} takes {len(names)} arguments, given {len(positional)}"
            )

        arguments = dict(zip(names, positional)) | dict(keyword)
        validated = self.arguments(**arguments)
        return {field: getattr(validated, field) for field in names}


@dc.dataclass(frozen=True)
class Change:
    """What one action did: the document keys it moved, and where from.

    ``before`` and ``after`` hold only the keys that actually differ. Restoring
    ``before`` and flushing redraws the old picture, because the render follows
    the state through the change listeners rather than being kept beside it.
    """

    action: str
    arguments: dict
    before: dict
    after: dict

    @property
    def changed(self) -> bool:
        return bool(self.before or self.after)


class Journal:
    """Watches what each action changes, for whoever wants to know.

    Every action is compared with itself: the document keys before it ran, and
    the same keys once the change listeners it set off have settled -- which is
    why the action runs inside a state block, so that the flush happens before
    the second look rather than after it.

    Not all of the document is written down as it happens -- a camera moves
    under VTK's own trackball, telling nobody -- so ``refresh`` is the chance to
    put that right. It runs at the end of an action and not at the start,
    deliberately: what the document held before is what the last action left
    it holding, and a drag that happened in between is that action's doing to
    undo, not this one's to be told about too late.

    While nothing is watching, no comparison is made, nothing is refreshed and
    no state block is opened: an action runs exactly as it did before there was
    a journal at all.
    """

    def __init__(
        self,
        state,
        keys: ty.Callable[[], ty.Iterable[str]],
        refresh: ty.Callable[[], None] | None = None,
    ):
        self._state = state
        self._keys = keys
        self._refresh = refresh or (lambda: None)
        self._listeners: list[ty.Callable[[Change], None]] = []

    @property
    def watched(self) -> bool:
        return bool(self._listeners)

    def watch(self, listener) -> None:
        """Be told about every action, whether or not it changed anything.

        Nothing is refreshed while nothing is watching, so the first thing to
        watch brings the document up to date rather than inheriting whatever
        was last written down.
        """
        self._refresh()
        self._listeners.append(listener)

    def unwatch(self, listener) -> None:
        self._listeners.remove(listener)

    def snapshot(self) -> dict:
        """The document as it stands, copied so that later writes cannot reach it."""
        return {key: copy.deepcopy(self._state[key]) for key in self._keys()}

    @cl.contextmanager
    def record(self, name: str, arguments: dict):
        if not self.watched:
            yield
            return

        before = self.snapshot()
        with self._state:
            yield
            # Inside the block, so that anything the refresh writes is flushed
            # with the rest rather than left pending for a later action.
            self._refresh()

        after = self.snapshot()

        moved = [key for key in after if not same(before.get(key), after[key])]
        change = Change(
            action=name,
            arguments=arguments,
            before={key: before[key] for key in moved},
            after={key: after[key] for key in moved},
        )
        for listener in list(self._listeners):
            listener(change)


def declared_actions(cls) -> list[tuple[str, str]]:
    """The ``(attribute, action name)`` pairs ``cls`` and its bases declare.

    Walks the classes rather than the instance: an instance's attributes
    include its properties, and reading those is not free and not always
    possible before the state they describe exists.
    """
    found = {}
    for klass in reversed(cls.__mro__):
        for attribute, value in vars(klass).items():
            name = getattr(value, "_action_name", None)
            if name is not None:
                found[attribute] = name
    return sorted(found.items())


class Registry:
    """Every action the app has, and the one way any of them is called."""

    def __init__(self, journal: Journal | None = None):
        self._actions: dict[str, Action] = {}
        self._observers: list[ty.Callable[[str, dict], None]] = []
        self.journal = journal

    def add(self, owner) -> None:
        """Register every action ``owner``'s class declares."""
        for attribute, name in declared_actions(type(owner)):
            if name in self._actions:
                raise ValueError(f"Two actions are named {name!r}")
            method = getattr(owner, attribute)
            self._actions[name] = Action(name, method, parameter_model(name, method))

    def bind(self, controller) -> None:
        """Publish the actions on trame's controller, which is what the UI calls.

        What is published is the same entry point a script reaches, so that a
        button press is recorded like anything else rather than going round the
        back of the journal.
        """
        for name in self._actions:
            setattr(controller, name, ft.partial(self.run, name))

    def observe(self, observer) -> None:
        """Watch every call: a context manager taking the name and arguments.

        Not the journal, which answers what an action *moved* and pays a copy of
        the whole document for the answer. This one answers only what was asked
        for, costs nothing, and so can stay armed for as long as a session lasts
        -- which a log of everything the user did has to be.

        Entered before the call, so a command that goes on to raise is still a
        command that was given, and exited after it, so a watcher can tell what
        an action did from what was done to the app while none was running.
        """
        self._observers.append(observer)

    def unobserve(self, observer) -> None:
        self._observers.remove(observer)

    @cl.contextmanager
    def _watching(self, name: str, arguments: dict):
        """Every observer, wrapped around the call in the order they asked."""
        with cl.ExitStack() as stack:
            for observer in list(self._observers):
                stack.enter_context(observer(name, arguments))
            yield

    def run(self, name: str, *positional, **keyword):
        """Do the named thing, with its arguments checked against its signature."""
        if name not in self._actions:
            raise KeyError(f"No such action: {name!r}. Known: {', '.join(self.names)}")

        entry = self._actions[name]
        arguments = entry.bind(positional, keyword)

        with self._watching(name, arguments):
            if self.journal is None:
                return entry.call(**arguments)

            with self.journal.record(name, arguments):
                return entry.call(**arguments)

    @property
    def names(self) -> list[str]:
        return sorted(self._actions)

    def __getitem__(self, name: str) -> Action:
        return self._actions[name]
