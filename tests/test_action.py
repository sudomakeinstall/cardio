"""The action registry and the journal that watches it.

The registry is covered end to end against a real server in test_app_smoke.py;
what is here is the parts that are easier to be exact about on their own -- how
a call's arguments are resolved, and what the journal does and does not do.
"""

# System
import contextlib as cl
import typing as ty

# Third Party
import numpy as np
import pydantic as pc
import pytest

# Internal
from cardio.action import Journal, Registry, action, background, same

# ---------------------------------------------------------------- doubles ----


class Recorder:
    """A few actions and a plain method, to register over."""

    def __init__(self, state=None):
        self.calls = []
        self.state = state

    @action("takes_two")
    def takes_two(self, first: int, second: str = "b"):
        self.calls.append((first, second))

    @action("takes_none")
    def takes_none(self, **kwargs):
        self.calls.append(())

    @action("assign")
    def assign(self, key: str, value: ty.Any):
        self.state[key] = value

    @action("append")
    def append(self, key: str, value: ty.Any):
        """Changes a value in place, which the journal must not follow."""
        self.state[key].append(value)

    def not_an_action(self):
        self.calls.append("no")


class Inheriting(Recorder):
    @action("also_mine")
    def also_mine(self):
        self.calls.append("mine")


class FakeState(dict):
    """Enough of trame's state for the journal: item access and a state block."""

    def __init__(self, **initial):
        super().__init__(initial)
        self.blocks = 0

    def __enter__(self):
        self.blocks += 1
        return self

    def __exit__(self, *exc_info):
        return False


# ------------------------------------------------------------- registry ------


def test_only_marked_methods_are_registered():
    registry = Registry()
    registry.add(Recorder())

    assert registry.names == ["append", "assign", "takes_none", "takes_two"]


def test_actions_are_inherited():
    registry = Registry()
    registry.add(Inheriting())

    assert "also_mine" in registry.names


def test_two_actions_may_not_share_a_name():
    registry = Registry()
    registry.add(Recorder())

    with pytest.raises(ValueError, match="Two actions are named"):
        registry.add(Recorder())


def test_an_unknown_action_says_what_it_knows():
    registry = Registry()
    registry.add(Recorder())

    with pytest.raises(KeyError, match="takes_two"):
        registry.run("nonsense")


@pytest.mark.parametrize(
    "positional,keyword",
    [
        ((1, "z"), {}),
        ((1,), {"second": "z"}),
        ((), {"first": 1, "second": "z"}),
    ],
)
def test_arguments_arrive_the_same_however_they_were_passed(positional, keyword):
    recorder = Recorder()
    registry = Registry()
    registry.add(recorder)

    registry.run("takes_two", *positional, **keyword)

    assert recorder.calls == [(1, "z")]


def test_a_default_is_filled_in():
    recorder = Recorder()
    registry = Registry()
    registry.add(recorder)

    registry.run("takes_two", 1)

    assert recorder.calls == [(1, "b")]


def test_a_missing_argument_is_refused():
    registry = Registry()
    registry.add(Recorder())

    with pytest.raises(pc.ValidationError):
        registry.run("takes_two")


def test_a_misspelled_argument_is_refused():
    registry = Registry()
    registry.add(Recorder())

    with pytest.raises(pc.ValidationError):
        registry.run("takes_two", first=1, secnod="z")


def test_too_many_positional_arguments_are_refused():
    registry = Registry()
    registry.add(Recorder())

    with pytest.raises(TypeError, match="takes_two"):
        registry.run("takes_two", 1, "z", "extra")


def test_kwargs_are_not_parameters():
    """Several actions are change listeners too, which trame hands the state."""
    registry = Registry()
    registry.add(Recorder())

    assert list(registry["takes_none"].arguments.model_fields) == []


def test_binding_publishes_a_call_that_goes_through_the_registry():
    recorder = Recorder()
    registry = Registry()
    registry.add(recorder)

    controller = type("Controller", (), {})()
    registry.bind(controller)
    controller.takes_two(1, "z")

    assert recorder.calls == [(1, "z")]


def test_background_keeps_the_signature_it_wraps():
    """trame's own task decorator does not, which would empty the model."""

    class Slow:
        @action("slow")
        @background
        async def slow(self, count: int):
            pass

    registry = Registry()
    registry.add(Slow())

    assert list(registry["slow"].arguments.model_fields) == ["count"]


# ------------------------------------------------------------ observers ------


def journalled(**state):
    """A registry over one recorder, journalling the keys of ``state``."""
    fake = FakeState(**state)
    journal = Journal(fake, lambda: list(fake))
    registry = Registry(journal)
    registry.add(Recorder(fake))
    return registry, journal, fake


def watcher(into, label=""):
    """An observer that writes down when it was entered and when it left."""

    @cl.contextmanager
    def observe(name, arguments):
        into.append((f"{label}enter", name, arguments))
        try:
            yield
        finally:
            into.append((f"{label}exit", name, arguments))

    return observe


def test_an_observer_is_told_the_name_and_the_arguments():
    registry = Registry()
    registry.add(Recorder())
    seen = []
    registry.observe(watcher(seen))

    registry.run("takes_two", 1)

    assert seen == [
        ("enter", "takes_two", {"first": 1, "second": "b"}),
        ("exit", "takes_two", {"first": 1, "second": "b"}),
    ]


def test_an_observer_is_entered_before_the_call_and_left_after_it():
    """The two halves are what tell an action's doing from anybody else's."""
    order = []

    class Slow:
        @action("slow")
        def slow(self):
            order.append("called")

    registry = Registry()
    registry.add(Slow())
    registry.observe(watcher(order))

    registry.run("slow")

    assert [step[0] if isinstance(step, tuple) else step for step in order] == [
        "enter",
        "called",
        "exit",
    ]


def test_an_observer_is_left_even_when_the_action_raises():
    """A command that goes on to raise is still a command that was given."""
    order = []

    class Failing:
        @action("boom")
        def boom(self):
            raise RuntimeError("no")

    registry = Registry()
    registry.add(Failing())
    registry.observe(watcher(order))

    with pytest.raises(RuntimeError):
        registry.run("boom")

    assert [step[0] for step in order] == ["enter", "exit"]


def test_an_unknown_action_is_never_observed():
    registry = Registry()
    registry.add(Recorder())
    seen = []
    registry.observe(watcher(seen))

    with pytest.raises(KeyError):
        registry.run("nonsense")

    assert seen == []


def test_observers_nest_in_the_order_they_asked_to_watch():
    registry = Registry()
    registry.add(Recorder())
    order = []
    registry.observe(watcher(order, "first "))
    registry.observe(watcher(order, "second "))

    registry.run("takes_none")

    assert [step[0] for step in order] == [
        "first enter",
        "second enter",
        "second exit",
        "first exit",
    ]


def test_observing_does_not_arm_the_journal():
    """The console watches every action all session; the journal cannot.

    Watching the journal copies the whole document twice per action, and a
    drag dispatches one per mouse move. An observer costs nothing, and this is
    what says the two hooks stayed separate.
    """
    registry, journal, state = journalled(a=1)
    registry.observe(watcher([]))

    registry.run("assign", "a", 10)

    assert not journal.watched
    assert state.blocks == 0


def test_an_observer_can_stop_observing():
    registry = Registry()
    registry.add(Recorder())
    seen = []
    observer = watcher(seen)

    registry.observe(observer)
    registry.run("takes_none")

    registry.unobserve(observer)
    registry.run("takes_none")

    assert [step[0] for step in seen] == ["enter", "exit"]


# -------------------------------------------------------------- journal ------


def test_an_unwatched_journal_does_nothing_at_all():
    """Not even a state block: an action runs as it did before there was one."""
    registry, journal, state = journalled(a=1)

    registry.run("takes_none")

    assert not journal.watched
    assert state.blocks == 0


def test_a_watched_action_runs_inside_a_state_block():
    """So the listeners it sets off have settled before the second look."""
    registry, journal, state = journalled(a=1)
    journal.watch(lambda change: None)

    registry.run("takes_none")

    assert state.blocks == 1


def test_the_change_carries_only_what_moved():
    registry, journal, _ = journalled(a=1, b=2, c=3)
    changes = []
    journal.watch(changes.append)

    registry.run("assign", "a", 10)

    (change,) = changes
    assert change.before == {"a": 1}
    assert change.after == {"a": 10}
    assert change.changed


def test_an_action_that_changes_nothing_still_reports():
    """The journal reports; deciding a no-op is not worth keeping is the
    listener's business."""
    registry, journal, _ = journalled(a=1)
    changes = []
    journal.watch(changes.append)

    registry.run("takes_none")

    (change,) = changes
    assert change.action == "takes_none"
    assert not change.changed


def test_the_arguments_are_recorded_by_name():
    registry, journal, _ = journalled(a=1)
    changes = []
    journal.watch(changes.append)

    registry.run("takes_two", 1)

    assert changes[0].arguments == {"first": 1, "second": "b"}


def test_the_snapshot_is_copied_from_the_state():
    """A value the action mutates in place must not move the record with it."""
    registry, journal, _ = journalled(a=[1, 2])
    changes = []
    journal.watch(changes.append)

    registry.run("append", "a", 3)

    assert changes[0].before == {"a": [1, 2]}
    assert changes[0].after == {"a": [1, 2, 3]}


def test_a_listener_can_stop_watching():
    _, journal, _ = journalled(a=1)

    def listener(change):
        pass

    journal.watch(listener)
    assert journal.watched

    journal.unwatch(listener)
    assert not journal.watched


@pytest.mark.parametrize(
    "one,other,expected",
    [
        (1, 1, True),
        (1, 2, False),
        ([1, 2], [1, 2], True),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), True),
        (np.array([1.0, 2.0]), np.array([1.0, 3.0]), False),
        (np.array([1.0]), 1.0, True),
        (None, 0, False),
    ],
)
def test_values_compare_without_an_array_ruining_it(one, other, expected):
    assert same(one, other) is expected
