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
import dataclasses as dc
import functools as ft
import inspect
import typing as ty

# Third Party
import pydantic as pc
from trame.app import asynchronous


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

    def __call__(self, **arguments):
        validated = self.arguments(**arguments)
        return self.call(
            **{
                field: getattr(validated, field)
                for field in self.arguments.model_fields
            }
        )


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

    def __init__(self):
        self._actions: dict[str, Action] = {}

    def add(self, owner) -> None:
        """Register every action ``owner``'s class declares."""
        for attribute, name in declared_actions(type(owner)):
            if name in self._actions:
                raise ValueError(f"Two actions are named {name!r}")
            method = getattr(owner, attribute)
            self._actions[name] = Action(name, method, parameter_model(name, method))

    def bind(self, controller) -> None:
        """Publish the actions on trame's controller, which is what the UI calls."""
        for name, entry in self._actions.items():
            setattr(controller, name, entry.call)

    def dispatch(self, name: str, **arguments):
        """Do the named thing, with its arguments checked against its signature."""
        if name not in self._actions:
            raise KeyError(f"No such action: {name!r}. Known: {', '.join(self.names)}")
        return self._actions[name](**arguments)

    @property
    def names(self) -> list[str]:
        return sorted(self._actions)

    def __contains__(self, name: str) -> bool:
        return name in self._actions

    def __getitem__(self, name: str) -> Action:
        return self._actions[name]

    def __len__(self) -> int:
        return len(self._actions)
