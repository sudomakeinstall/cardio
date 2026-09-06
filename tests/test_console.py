"""The action log, and the syntax it is written in and read from.

The claim worth being exact about is that ``format_call`` and ``parse_call``
are inverses: a line the console printed is a line the prompt takes and a line
a script may contain. Everything else here is the log's bookkeeping -- what
coalesces, what is bounded, and what comes back out as something to re-run.
"""

# System
import datetime as dt
import itertools

# Third Party
import pytest

# Internal
import cardio.console as console

# ------------------------------------------------------------- the syntax ----

# One call per shape: none, one, several, a list argument, and an enum, which
# is the case a bare repr would spell as its member rather than its value.
CALLS = [
    ("place_camera", {}),
    ("toggle_crosshairs", {}),
    ("add_rotation", {"axis": "Z"}),
    ("set_window_level_preset", {"preset": 3}),
    ("adjust_window_level", {"window_delta": 10.0, "level_delta": -5.0}),
    ("pan_view", {"view_name": "axial", "dx": 3.0, "dy": -2.0}),
    ("rotate_view", {"view_name": "axial", "start": [1.0, 2.0], "end": [3.0, 4.0]}),
    ("toggle_maximized", {"view": ""}),
    ("set_snap_mode", {"enabled": True, "label": None}),
]


@pytest.mark.parametrize("name,arguments", CALLS, ids=[name for name, _ in CALLS])
def test_a_printed_call_is_a_call_that_can_be_typed(name, arguments):
    """The whole feature in one line: the log and the prompt are one language."""
    assert console.parse_call(console.format_call(name, arguments)) == (
        name,
        [],
        arguments,
    )


def test_an_enum_argument_is_spelled_as_its_value():
    """A bare repr would print the member, which is not something to type back."""
    from cardio.view import Layout

    text = console.format_call("toggle_maximized", {"view": Layout.AXIAL})

    assert text == "toggle_maximized(view='axial')"
    assert console.parse_call(text) == ("toggle_maximized", [], {"view": "axial"})


def test_a_bare_name_is_a_call_with_no_arguments():
    assert console.parse_call("place_camera") == ("place_camera", [], {})


def test_positional_arguments_are_allowed():
    """``Action.bind`` resolves them, so the syntax need not forbid them."""
    assert console.parse_call("toggle_maximized('axial')") == (
        "toggle_maximized",
        ["axial"],
        {},
    )


def test_surrounding_space_is_not_part_of_the_command():
    assert console.parse_call("  place_camera  ") == ("place_camera", [], {})


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", "Nothing to run"),
        ("   ", "Nothing to run"),
        ("1 + 1", "not a call"),
        ("add_rotation(axis='Z') ; place_camera", "not a call"),
        ("add_rotation(axis=", "not a call"),
        ("os.system('rm -rf /')", "not an action name"),
        ("add_rotation(axis=Z)", "not a plain value"),
        ("add_rotation(axis=open('x'))", "not a plain value"),
        ("add_rotation(*axes)", "not a plain value"),
        ("add_rotation(**arguments)", r"\*\* arguments"),
    ],
)
def test_what_is_not_a_command_says_why(text, expected):
    with pytest.raises(ValueError, match=expected):
        console.parse_call(text)


def test_parsing_evaluates_nothing():
    """A name that happens to be a builtin is a name, and nothing more.

    Parsing does not decide whether an action exists -- the registry does, and
    says what it knows. What matters here is that reaching that refusal costs
    nothing: no call was made on the way.
    """
    assert console.parse_call("open('/etc/passwd')") == ("open", ["/etc/passwd"], {})


# ---------------------------------------------------------------- the log ----


def clock(step=1.0, start="12:00:00"):
    """A clock that ticks ``step`` seconds per look.

    The default is far enough apart that nothing coalesces; a gesture is
    spelled with a step small enough that everything does.
    """
    base = dt.datetime.strptime(start, "%H:%M:%S").astimezone()
    ticks = itertools.count()
    return lambda: base + dt.timedelta(seconds=step * next(ticks))


GESTURE = 1 / 60


def test_an_action_becomes_one_line():
    log = console.Log(now=clock())

    assert log.record("add_rotation", {"axis": "Z"}) is True

    (entry,) = log.entries
    assert entry["text"] == "add_rotation(axis='Z')"
    assert entry["at"] == "12:00:00"
    assert entry["kind"] == "action"
    assert entry["count"] == 1


def test_a_gesture_counts_rather_than_repeats():
    """A drag dispatches per mouse move; forty-seven identical lines is not a log."""
    log = console.Log(now=clock(GESTURE))

    assert log.record("adjust_window_level", {"window_delta": 1.0}) is True
    assert log.record("adjust_window_level", {"window_delta": 2.0}) is False
    assert log.record("adjust_window_level", {"window_delta": 3.0}) is False

    (entry,) = log.entries
    assert entry["count"] == 3
    assert entry["text"] == "adjust_window_level(window_delta=3.0)", "the latest"


def test_the_same_action_asked_for_again_later_is_a_new_line():
    """Two button presses are two things done, however alike they look."""
    log = console.Log(now=clock())

    log.record("add_rotation", {"axis": "Z"})
    log.record("add_rotation", {"axis": "X"})

    assert [entry["text"] for entry in log.entries] == [
        "add_rotation(axis='X')",
        "add_rotation(axis='Z')",
    ]


def test_a_different_action_starts_a_new_line():
    log = console.Log(now=clock())

    log.record("add_rotation", {"axis": "Z"})
    log.record("place_camera", {})
    log.record("add_rotation", {"axis": "X"})

    assert [entry["count"] for entry in log.entries] == [1, 1, 1]


def test_the_newest_line_comes_first():
    """Which is the order the panel stacks them in."""
    log = console.Log(now=clock())

    log.record("add_rotation", {"axis": "Z"})
    log.record("place_camera", {})

    assert [entry["text"] for entry in log.entries] == [
        "place_camera()",
        "add_rotation(axis='Z')",
    ]


def test_every_line_has_an_identity_of_its_own():
    """The panel keys its rows by it, and a bounded log reuses no number."""
    log = console.Log(limit=2, now=clock())

    for axis in "XYZW":
        log.record("add_rotation", {"axis": axis})
        log.record("place_camera", {})

    numbers = [entry["n"] for entry in log.entries]
    assert len(set(numbers)) == len(numbers)


def test_the_log_is_bounded():
    log = console.Log(limit=3, now=clock())

    for i in range(10):
        log.record(f"action_{i}", {})

    assert len(log) == 3
    assert [entry["text"] for entry in log.entries] == [
        "action_9()",
        "action_8()",
        "action_7()",
    ]


def test_an_error_is_a_line_that_did_nothing():
    log = console.Log(now=clock())

    log.record("add_rotation", {"axis": "Z"})
    log.error("No such action: 'nonsense'")

    assert [entry["kind"] for entry in log.entries] == ["error", "action"]
    assert log.script == [("add_rotation", {"axis": "Z"})]


def test_two_errors_do_not_coalesce():
    """They are not one thing asked for twice; they are two refusals."""
    log = console.Log(now=clock())

    log.error("first")
    log.error("second")

    assert len(log.entries) == 2


def test_clearing_leaves_nothing_behind():
    log = console.Log(now=clock())
    log.record("add_rotation", {"axis": "Z"})

    log.clear()

    assert log.entries == []
    assert log.script == []


# ------------------------------------------------------------- the script ----


def test_the_script_is_what_was_asked_for_in_the_order_it_was_asked():
    log = console.Log(now=clock())

    log.record("add_rotation", {"axis": "Z"})
    log.record("place_camera", {})

    assert log.script == [
        ("add_rotation", {"axis": "Z"}),
        ("place_camera", {}),
    ]


def test_a_collapsed_drag_keeps_every_call_in_the_script():
    """What collapsed was the reading of it, not the record.

    Dropping the rest would leave a script that does not put the app back
    where the drag left it, the deltas being cumulative.
    """
    log = console.Log(now=clock(GESTURE))

    for delta in (1.0, 2.0, 3.0):
        log.record("adjust_window_level", {"window_delta": delta})

    assert len(log.entries) == 1, "one line"
    assert log.script == [
        ("adjust_window_level", {"window_delta": 1.0}),
        ("adjust_window_level", {"window_delta": 2.0}),
        ("adjust_window_level", {"window_delta": 3.0}),
    ]


def test_the_script_does_not_share_its_arguments_with_the_log():
    """A caller editing what it was handed must not rewrite the record."""
    log = console.Log(now=clock())
    log.record("add_rotation", {"axis": "Z"})

    log.script[0][1]["axis"] = "X"

    assert log.script == [("add_rotation", {"axis": "Z"})]
