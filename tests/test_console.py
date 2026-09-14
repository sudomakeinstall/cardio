"""The action log, and the syntax it is written in and read from.

The claim worth being exact about is that ``format_call`` and ``parse_call``
are inverses: a line the console printed is a line the prompt takes and a line
a script may contain. Everything else here is the log's bookkeeping -- what
coalesces, what is bounded, and what comes back out as something to re-run.
"""

# System
import copy
import datetime as dt
import itertools

# Third Party
import pytest

# Internal
import cardio.console as console
from cardio.ui.common import DRAWER_WIDTH, STATIC, STYLESHEET, drawer_styles
from tests.test_app_smoke import build_app, build_scene, connect
from tests.test_session import session_on

# ------------------------------------------------------------- the syntax ----

# One call per shape: none, one, several, a list argument, and an enum, which
# is the case a bare repr would spell as its member rather than its value.
CALLS = [
    ("place_camera", {}),
    ("toggle_crosshairs", {}),
    ("add_rotation", {"axis": "Z"}),
    ("set_window_level_preset", {"preset": 3}),
    ("adjust_window_level", {"window_delta": 10.0, "level_delta": -5.0}),
    ("pan_view", {"view_name": "ul", "dx": 3.0, "dy": -2.0}),
    ("rotate_view", {"view_name": "ul", "start": [1.0, 2.0], "end": [3.0, 4.0]}),
    ("toggle_maximized", {"view": ""}),
    ("set_snap_mode", {"enabled": True, "label": None}),
]


@pytest.mark.parametrize("name,arguments", CALLS, ids=[name for name, _ in CALLS])
def test_a_printed_call_is_a_call_that_can_be_typed(name, arguments):
    """The whole feature in one line: the log and the prompt are one language.

    Printed as the console prints it, prefix and all, so that a line lifted off
    the panel runs as it was copied rather than after being edited.
    """
    printed = console.format_call(name, arguments, prefix=console.PREFIX)

    assert console.parse_call(printed) == (name, [], arguments)


@pytest.mark.parametrize("name,arguments", CALLS, ids=[name for name, _ in CALLS])
def test_the_prefix_is_optional_at_the_prompt(name, arguments):
    """Typing is not copying: what the prompt asks of is not in doubt."""
    assert console.parse_call(console.format_call(name, arguments)) == (
        name,
        [],
        arguments,
    )


def test_an_enum_argument_is_spelled_as_its_value():
    """A bare repr would print the member, which is not something to type back."""
    from cardio.view import Layout

    text = console.format_call("toggle_maximized", {"view": Layout.UL})

    assert text == "toggle_maximized(view='ul')"
    assert console.parse_call(text) == ("toggle_maximized", [], {"view": "ul"})


def test_a_bare_name_is_a_call_with_no_arguments():
    assert console.parse_call("place_camera") == ("place_camera", [], {})
    assert console.parse_call("do.place_camera") == ("place_camera", [], {})


def test_positional_arguments_are_allowed():
    """``Action.bind`` resolves them, so the syntax need not forbid them."""
    assert console.parse_call("toggle_maximized('ul')") == (
        "toggle_maximized",
        ["ul"],
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
    assert entry["text"] == "do.add_rotation(axis='Z')"
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
    assert entry["text"] == "do.adjust_window_level(window_delta=3.0)", "the latest"


def test_the_same_action_asked_for_again_later_is_a_new_line():
    """Two button presses are two things done, however alike they look."""
    log = console.Log(now=clock())

    log.record("add_rotation", {"axis": "Z"})
    log.record("add_rotation", {"axis": "X"})

    assert [entry["text"] for entry in log.entries] == [
        "do.add_rotation(axis='Z')",
        "do.add_rotation(axis='X')",
    ]


def test_a_different_action_starts_a_new_line():
    log = console.Log(now=clock())

    log.record("add_rotation", {"axis": "Z"})
    log.record("place_camera", {})
    log.record("add_rotation", {"axis": "X"})

    assert [entry["count"] for entry in log.entries] == [1, 1, 1]


def test_the_newest_line_comes_last():
    """The order things happened in, which is the order they are read in.

    The panel puts the newest at the bottom by reversing its own box rather
    than the lines, so that a selection dragged down the log runs the way the
    lines do.
    """
    log = console.Log(now=clock())

    log.record("add_rotation", {"axis": "Z"})
    log.record("place_camera", {})

    assert [entry["text"] for entry in log.entries] == [
        "do.add_rotation(axis='Z')",
        "do.place_camera()",
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
        "do.action_7()",
        "do.action_8()",
        "do.action_9()",
    ]


def test_an_error_is_a_line_that_did_nothing():
    log = console.Log(now=clock())

    log.record("add_rotation", {"axis": "Z"})
    log.error("No such action: 'nonsense'")

    assert [entry["kind"] for entry in log.entries] == ["action", "error"]
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


# --------------------------------------------------------- the walk back ----
#
# The prompt's history: the whole log, narrowed to what is already typed, the
# way a terminal narrows it.

WALKED = [
    "do.add_rotation(axis='Z')",
    "do.toggle_console()",
    "do.set_state(key='tile_cols', value=4)",
    "do.toggle_maximized(view='ul')",
]


def test_a_walk_with_nothing_typed_is_every_line():
    assert console.matching(WALKED, "") == WALKED


def test_a_walk_is_narrowed_by_what_is_typed():
    """The whole feature: what is typed is the start of what is wanted."""
    assert console.matching(WALKED, "do.t") == [
        "do.toggle_console()",
        "do.toggle_maximized(view='ul')",
    ]


def test_a_bare_name_finds_the_lines_the_console_printed():
    """The prompt takes a name without the prefix, so a walk has to as well."""
    assert console.matching(WALKED, "t") == console.matching(WALKED, "do.t")


def test_a_walk_that_matches_nothing_is_empty():
    assert console.matching(WALKED, "do.nonsense") == []


def test_a_repeated_line_is_walked_once():
    """A log is mostly repetition, and stopping eleven times at one text is not
    a walk anybody finishes.
    """
    lines = ["do.place_camera()", "do.add_rotation(axis='Z')", "do.place_camera()"]

    assert console.matching(lines, "") == [
        "do.add_rotation(axis='Z')",
        "do.place_camera()",
    ], "the most recent of them, in its own place"


def walked(steps, lines=None, typed=""):
    """Where the prompt lands after each of ``steps``, starting from ``typed``.

    Each step is handed what the last one put in the box, which is what the
    field hands back when the arrow is pressed again.
    """
    history = console.History()
    lines = WALKED if lines is None else lines

    landed = []
    for step in steps:
        typed = history.walk(lines, typed, step)
        landed.append(typed)
    return landed


def test_a_step_back_takes_the_newest_line():
    assert walked([-1]) == ["do.toggle_maximized(view='ul')"]


def test_stepping_back_walks_the_log_backwards():
    assert walked([-1, -1, -1]) == [
        "do.toggle_maximized(view='ul')",
        "do.set_state(key='tile_cols', value=4)",
        "do.toggle_console()",
    ]


def test_the_far_end_of_the_log_is_where_a_walk_stops():
    """Rather than wrapping around to the newest, which loses the place."""
    assert walked([-1] * 6)[-2:] == ["do.add_rotation(axis='Z')"] * 2


def test_stepping_forward_comes_back():
    assert walked([-1, -1, 1]) == [
        "do.toggle_maximized(view='ul')",
        "do.set_state(key='tile_cols', value=4)",
        "do.toggle_maximized(view='ul')",
    ]


def test_stepping_past_the_newest_gives_back_what_was_being_typed():
    """The box is not left holding the last recall, which is not what was meant."""
    history = console.History()
    recalled = history.walk(WALKED, "do.t", -1)

    assert history.walk(WALKED, recalled, 1) == "do.t"


def test_a_walk_with_nothing_to_show_leaves_the_prompt_alone():
    assert walked([-1], typed="do.nonsense") == ["do.nonsense"]


def test_editing_a_recalled_line_searches_for_the_edit():
    """What says something was typed is that the box no longer holds what the
    walk put there -- which needs no listening to keystrokes.
    """
    history = console.History()
    history.walk(WALKED, "", -1)

    assert history.walk(WALKED, "do.t", -1) == "do.toggle_maximized(view='ul')"
    assert history.walk(WALKED, "do.toggle_maximized(view='ul')", -1) == (
        "do.toggle_console()"
    ), "and goes on walking what it found"


def test_the_log_moving_on_ends_the_walk():
    """The log grows while the prompt sits there, which a shell's history does
    not: everything the drawer does is a line.

    A walk that kept the matches it started with would go on walking a log the
    app has moved on from, and an arrow would never reach what was just done.
    """
    history = console.History()
    recalled = history.walk(WALKED[:2], "", -1)

    assert history.walk(WALKED, recalled, -1) == WALKED[-1], "the newest of them"


def test_a_recall_left_in_the_box_is_not_searched_for():
    """It is the walk's own text rather than anybody's, so the next walk is not
    a search for lines beginning with it.
    """
    history = console.History()
    recalled = history.walk(["do.toggle_console()"], "", -1)

    grown = ["do.toggle_console()", "do.set_state(key='frame', value=1)"]

    assert history.walk(grown, recalled, -1) == "do.set_state(key='frame', value=1)"


def test_what_was_typed_survives_the_log_moving_on():
    """What the walk left is not anybody's; what was typed over it is."""
    history = console.History()
    history.walk(WALKED[:2], "", -1)

    assert history.walk(WALKED, "do.t", -1) == "do.toggle_maximized(view='ul')"


def test_forgetting_a_walk_starts_the_next_one_over():
    """A command that was run ends the walk, even when it left its text behind.

    The next arrow searches for what is in the box rather than going on with
    the walk the run interrupted.
    """
    kept, forgotten = console.History(), console.History()
    for history in (kept, forgotten):
        recalled = history.walk(WALKED, "", -1)
    forgotten.forget()

    assert kept.walk(WALKED, recalled, -1) == "do.set_state(key='tile_cols', value=4)"
    assert forgotten.walk(WALKED, recalled, -1) == recalled, "the newest match again"


def test_the_whole_log_is_what_is_walked():
    """Not the typed part of it: a drawer control is a line like any other."""
    log = console.Log(now=clock())
    log.record("set_state", {"key": "tile_cols", "value": 4})
    log.record("add_rotation", {"axis": "Z"})

    assert log.recallable == [
        "do.set_state(key='tile_cols', value=4)",
        "do.add_rotation(axis='Z')",
    ]


def test_a_refusal_is_walked_as_what_was_asked():
    """A typo is the thing a history is most used for, and the message is not it."""
    log = console.Log(now=clock())
    log.error("No such action: 'nonsens'", asked="nonsens()")

    assert log.recallable == ["nonsens()"]


def test_a_refusal_nobody_typed_is_not_walked():
    log = console.Log(now=clock())
    log.error("something went wrong")

    assert log.recallable == []


def test_a_call_that_raised_is_walked_as_it_was_printed():
    """What it did not do is a reason to run it again, not a reason not to."""
    log = console.Log(now=clock())
    log.record("add_rotation", {"axis": "W"}, ok=False)

    assert log.recallable == ["do.add_rotation(axis='W')"]


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


# ------------------------------------------------------- against a session ----


@pytest.fixture
def session(tmp_path):
    return session_on(tmp_path)


def log_of(session):
    return session.logic.console.log


def test_an_action_asked_for_is_an_action_logged(session):
    session.do("add_rotation", axis="Z")

    (entry,) = log_of(session).entries
    assert console.parse_call(entry["text"]) == ("add_rotation", [], {"axis": "Z"})


def test_a_typed_call_logs_what_a_button_would_have(session):
    """One line, not two: the wrapper is not a thing the app was asked to do."""
    session.do("run_command", text="add_rotation(axis='Z')")

    (entry,) = log_of(session).entries
    assert entry["text"] == "do.add_rotation(axis='Z')"


def test_a_typed_call_does_what_dispatching_it_would(tmp_path_factory):
    typed = session_on(tmp_path_factory.mktemp("typed"))
    dispatched = session_on(tmp_path_factory.mktemp("dispatched"))

    typed.do("run_command", text="add_rotation(axis='Z')")
    dispatched.do("add_rotation", axis="Z")

    assert [step.axis for step in typed.scene.mpr_rotation_sequence.angles_list] == [
        step.axis for step in dispatched.scene.mpr_rotation_sequence.angles_list
    ]
    assert typed.server.state.console_input == ""


@pytest.mark.parametrize(
    "text,expected",
    [
        ("nonsense()", "No such action"),
        ("add_rotation(ax='Z')", "ax"),
        ("add_rotation()", "axis"),
        ("add_rotation(axis=", "not a call"),
        ("os.system('rm -rf /')", "not an action name"),
    ],
)
def test_a_command_that_cannot_be_run_says_so_and_does_nothing(session, text, expected):
    session.do("run_command", text=text)

    (entry,) = log_of(session).entries
    assert entry["kind"] == "error"
    assert expected in entry["text"]
    assert session.scene.mpr_rotation_sequence.angles_list == []


def test_a_refused_command_stays_in_the_box_to_be_corrected(session):
    session.server.state.console_input = "add_rotation(ax='Z')"

    session.do("run_command", text="add_rotation(ax='Z')")

    assert session.server.state.console_input == "add_rotation(ax='Z')"


def recall(session, step=-1):
    """Press an arrow, and say what the prompt is left holding."""
    session.do("recall_command", step=step)
    return session.server.state.console_input


def test_an_arrow_puts_the_newest_line_in_the_prompt(session):
    session.do("add_rotation", axis="Z")

    assert recall(session) == "do.add_rotation(axis='Z')"


def test_a_walk_steps_through_what_nobody_typed(session):
    """The whole log: a drawer control is as good a thing to arrow back to."""
    session.ready()
    moved(session, tile_cols=4)
    session.do("add_rotation", axis="Z")

    assert recall(session) == "do.add_rotation(axis='Z')"
    assert recall(session) == "do.set_state(key='tile_cols', value=4)"


def test_a_walk_is_narrowed_by_what_is_already_typed(session):
    session.do("add_rotation", axis="Z")
    session.do("toggle_crosshairs")
    session.server.state.console_input = "do.a"

    assert recall(session) == "do.add_rotation(axis='Z')"


def test_a_command_that_was_refused_is_walked_back_to(session):
    """What the message says is not what there is anything to fix in."""
    session.do("run_command", text="nonsens()")

    assert recall(session) == "nonsens()"


def test_running_a_command_starts_the_next_walk_over(session):
    """A run ends the walk, wherever it had got to, as a shell's does."""
    session.do("add_rotation", axis="Z")
    session.do("toggle_crosshairs")
    recall(session)

    session.do("run_command", text="add_rotation(axis='X')")

    assert recall(session) == "do.add_rotation(axis='X')", (
        "the newest, not where it was"
    )


def test_an_arrow_after_doing_something_reaches_what_was_just_done(session):
    """The log grows while the prompt sits there, holding an earlier recall.

    That text is the walk's own and not anybody's, so the arrow is asking for
    the whole log again rather than for lines that begin with what it is left
    holding.
    """
    session.ready()
    session.do("toggle_console")
    assert recall(session) == "do.toggle_console()"

    moved(session, frame=1)
    moved(session, tile_cols=4)

    assert recall(session) == "do.set_state(key='tile_cols', value=4)"
    assert recall(session) == "do.set_state(key='frame', value=1)"
    assert recall(session) == "do.toggle_console()"


def test_walking_the_log_is_not_itself_a_line_of_it(session):
    """It is about the console rather than about the app, as the prompt is."""
    session.do("add_rotation", axis="Z")

    recall(session)

    assert texts(session) == ["do.add_rotation(axis='Z')"]


def test_clearing_the_console_leaves_no_trace_of_itself(session):
    session.do("add_rotation", axis="Z")

    session.do("clear_console")

    assert log_of(session).entries == []


def test_the_log_is_published_only_while_the_console_is_showing(session):
    session.do("add_rotation", axis="Z")
    assert session.server.state.console_entries == [], "nothing is looking"

    session.do("toggle_console")

    published = session.server.state.console_entries
    assert [entry["text"] for entry in published] == [
        "do.add_rotation(axis='Z')",
        "do.toggle_console()",
    ], "opening it shows what it missed"


def test_a_configured_console_opens_showing(tmp_path):
    session = session_on(tmp_path, view={"console_visible": True})

    assert session.server.state.console_visible is True

    session.do("add_rotation", axis="Z")
    assert session.server.state.console_entries


def test_the_console_does_not_arm_the_journal(session):
    """The assertion that keeps a log of everything off the deepcopy path.

    Watching the journal copies every document key twice per action, and a
    drag dispatches one per mouse move. The console is armed all session, so
    it must not be watching that.
    """
    session.do("add_rotation", axis="Z")

    assert not session.logic.journal.watched


def test_the_log_is_a_script_that_reproduces_the_session(tmp_path_factory):
    """What makes capture and replay a file format away rather than a design away."""
    steps = [
        ("add_rotation", {"axis": "Z"}),
        ("add_rotation", {"axis": "X"}),
        ("toggle_crosshairs", None),
        ("set_window_level_preset", {"preset": 3}),
    ]

    first = session_on(tmp_path_factory.mktemp("first"))
    first.run(steps)

    second = session_on(tmp_path_factory.mktemp("second"))
    second.run(log_of(first).script)

    assert [step.axis for step in second.scene.mpr_rotation_sequence.angles_list] == [
        step.axis for step in first.scene.mpr_rotation_sequence.angles_list
    ]
    for key in ("mpr_crosshairs_enabled", "mpr_window", "mpr_level"):
        assert second.server.state[key] == first.server.state[key], key


# --------------------------------------------- what nobody asked for by name ----
#
# Most of the drawer binds a document key straight to a widget, so most of what
# a person does reaches no action at all. What is checked here is that those
# still reach the log, as the action that would do them again.


def moved(session, **keys):
    """Write document keys the way a bound widget does: no action anywhere."""
    with session.server.state as state:
        for key, value in keys.items():
            state[key] = value


def texts(session):
    return [entry["text"] for entry in log_of(session).entries]


def test_a_key_bound_straight_to_a_widget_still_reaches_the_log(session):
    session.ready()

    moved(session, tile_cols=4)

    assert texts(session) == ["do.set_state(key='tile_cols', value=4)"]


def test_dragging_the_depth_range_is_written_down(session):
    """It moves only when a person drags it, which is the whole of why it is
    document state: nothing else in the app ever writes it.
    """
    session.ready()

    moved(session, clip_depth=[12.0, 345.0])

    assert texts(session) == ["do.set_state(key='clip_depth', value=[12.0, 345.0])"]


def test_what_the_log_says_is_what_would_do_it_again(session):
    session.ready()
    moved(session, theme_mode="light")

    (line,) = texts(session)

    assert console.parse_call(line) == (
        "set_state",
        [],
        {"key": "theme_mode", "value": "light"},
    )


def test_several_keys_moving_at_once_are_several_lines(session):
    """A grid is two keys, and a script has to set both."""
    session.ready()

    moved(session, tile_rows=2, tile_cols=4)

    assert sorted(texts(session)) == [
        "do.set_state(key='tile_cols', value=4)",
        "do.set_state(key='tile_rows', value=2)",
    ]


def test_an_action_is_not_written_down_twice(session):
    """It moves document keys too, and the log already names it."""
    session.ready()

    session.do("toggle_crosshairs")

    assert texts(session) == ["do.toggle_crosshairs()"], (
        "the keys it moved must not come back as set_state as well"
    )


def test_bringing_the_app_up_is_not_something_anybody_did(session):
    """Seeding writes the whole document, and the log starts after it."""
    session.ready()

    assert log_of(session).entries == []


def test_two_keys_dragged_in_turn_do_not_collapse_into_one_line(session):
    """One action name now stands for every key, so the name cannot be enough."""
    session.ready()

    moved(session, tile_cols=4)
    moved(session, tile_rows=2)

    assert len(texts(session)) == 2


def test_the_same_key_dragged_is_one_line(session):
    session.ready()

    for cols in (2, 3, 4):
        moved(session, tile_cols=cols)

    (entry,) = log_of(session).entries
    assert entry["count"] == 3
    assert entry["text"] == "do.set_state(key='tile_cols', value=4)"


# ------------------------------------------- a place inside a document key ----
#
# One key holds a document of its own. The rotation panel binds paths within it
# under DeepReactive, which syncs the whole variable -- so the app is handed the
# entire sequence to report that one field of one step moved. What is checked
# here is that the log says the field.


def rotations(session, steps=2, clear=True):
    """A session holding ``steps`` rotations.

    The log is wiped by default, so that what a test does next is the only
    thing in it. A replay wants the rotations its edits are made to, and asks
    to keep them.
    """
    session.ready()
    for axis in "ZXY"[:steps]:
        session.do("add_rotation", axis=axis)
    if clear:
        session.do("clear_console")
    return session


def edited(session, index, **fields):
    """Move one step's fields the way a DeepReactive widget does.

    The whole variable comes back with one thing different in it, which is all
    the client is able to say.
    """
    data = copy.deepcopy(session.server.state.mpr_rotation_data)
    data["angles_list"][index].update(fields)
    moved(session, mpr_rotation_data=data)


def test_a_step_within_the_sequence_is_named_rather_than_repeated(session):
    """The whole feature: the line says the field, not the document holding it."""
    rotations(session)

    edited(session, 1, visible=False)

    assert texts(session) == [
        "do.set_state(key='mpr_rotation_data.angles_list.1.visible', value=False)"
    ]


def test_the_line_about_a_step_is_a_line_that_can_be_run(session):
    rotations(session)
    edited(session, 1, visible=False)

    (line,) = texts(session)

    assert console.parse_call(line) == (
        "set_state",
        [],
        {"key": "mpr_rotation_data.angles_list.1.visible", "value": False},
    )


def test_running_that_line_moves_the_step_it_names(session):
    rotations(session)

    session.do("set_state", key="mpr_rotation_data.angles_list.1.visible", value=False)

    steps = session.server.state.mpr_rotation_data["angles_list"]
    assert [step["visible"] for step in steps] == [True, False]


def test_a_path_that_leads_nowhere_says_so(session):
    rotations(session)

    session.do("run_command", text="set_state(key='mpr_rotation_data.nope', value=1)")

    assert (
        "mpr_rotation_data has no 'nope'; it holds angles_list, metadata, mpr_origin"
        in texts(session)
    )


def test_a_path_into_session_state_is_refused_like_a_key(session):
    """The head of the path is the key, and the same rule applies to it."""
    session.ready()

    session.do("run_command", text="set_state(key='playing.0', value=1)")

    assert "'playing' is not a document key" in texts(session)


def test_two_fields_of_one_step_are_two_lines(session):
    rotations(session)

    edited(session, 0, angle=45.0, visible=False)

    assert sorted(texts(session)) == [
        "do.set_state(key='mpr_rotation_data.angles_list.0.angle', value=45.0)",
        "do.set_state(key='mpr_rotation_data.angles_list.0.visible', value=False)",
    ]


def test_dragging_one_step_is_one_line(session):
    """The group is the path, so a drag coalesces the way every gesture does."""
    rotations(session)

    for angle in (10.0, 20.0, 30.0):
        edited(session, 0, angle=angle)

    (entry,) = log_of(session).entries
    assert entry["count"] == 3
    assert (
        entry["text"]
        == "do.set_state(key='mpr_rotation_data.angles_list.0.angle', value=30.0)"
    )


def test_two_steps_dragged_in_turn_stay_two_lines(session):
    rotations(session)

    edited(session, 0, angle=10.0)
    edited(session, 1, angle=20.0)

    assert len(texts(session)) == 2


def test_a_step_added_by_an_action_rebases_what_the_next_edit_compares_with(session):
    """The cache is kept up to date even while nothing is being written down.

    An action's own writes are hidden from the log, but they still move the
    sequence -- and an edit compared against what was there before the action
    would report the action as well.
    """
    rotations(session)

    session.do("add_rotation", axis="Y")
    session.do("clear_console")
    edited(session, 2, visible=False)

    assert texts(session) == [
        "do.set_state(key='mpr_rotation_data.angles_list.2.visible', value=False)"
    ]


def test_a_sequence_that_changed_length_is_reported_whole(session):
    """There is no naming which entry a shorter list is missing."""
    rotations(session)

    data = copy.deepcopy(session.server.state.mpr_rotation_data)
    data["angles_list"].pop()
    moved(session, mpr_rotation_data=data)

    (line,) = texts(session)
    assert line.startswith("do.set_state(key='mpr_rotation_data.angles_list', value=[")


def test_switching_units_is_the_switch_and_not_what_it_re_expresses(session):
    """Every angle is rewritten by the listener; the switch is what was asked for.

    Replaying the switch does the conversion, which is a truer account of it
    than a script asserting the numbers it came out with.
    """
    rotations(session)

    moved(session, angle_units="degrees")

    assert texts(session) == ["do.set_state(key='angle_units', value='degrees')"]


def test_switching_the_index_order_is_the_switch_alone(session):
    rotations(session)

    moved(session, index_order="roma")

    assert texts(session) == ["do.set_state(key='index_order', value='roma')"]


def test_a_step_edited_after_a_switch_is_still_written_down(session):
    """A consequence covers the flush that follows it, and not the app's future."""
    rotations(session)
    moved(session, angle_units="degrees")
    session.do("clear_console")

    edited(session, 1, visible=False)

    assert texts(session) == [
        "do.set_state(key='mpr_rotation_data.angles_list.1.visible', value=False)"
    ]


def test_a_session_edited_through_the_rotation_panel_replays(tmp_path_factory):
    """The point of the smaller line: it is still the whole of what was done."""
    first = session_on(tmp_path_factory.mktemp("first"))
    rotations(first, clear=False)
    edited(first, 0, angle=45.0)
    edited(first, 1, visible=False)

    second = session_on(tmp_path_factory.mktemp("second"))
    second.ready()
    second.run(log_of(first).script)

    assert (
        second.server.state.mpr_rotation_data["angles_list"]
        == first.server.state.mpr_rotation_data["angles_list"]
    )


def test_the_cameras_moving_is_not_something_anybody_asked_for(session):
    """They are refitted by a layout change and turned by a trackball drag.

    The state is only ever a note of where VTK's ended up, and the one time a
    person means to move one, place_camera writes that down as the action.
    """
    session.ready()

    moved(session, maximized_view="tile")

    assert texts(session) == ["do.set_state(key='maximized_view', value='tile')"]


def test_the_frame_is_not_logged_while_a_loop_is_stepping_it(session):
    """A cine writes it thirty times a second; what was asked for was to play."""
    session.ready()
    moved(session, playing=True)

    moved(session, frame=1)
    moved(session, frame=2)

    assert texts(session) == []


def test_the_frame_is_logged_when_a_person_scrubs_it(session):
    session.ready()

    moved(session, frame=1)

    assert texts(session) == ["do.set_state(key='frame', value=1)"]


def test_set_state_refuses_a_key_that_is_not_the_document(session):
    """Session state is not a thing a script has business reaching into."""
    session.ready()

    session.do("run_command", text="set_state(key='capture_running', value=True)")

    tried, said = log_of(session).entries
    assert "not a document key" in said["text"]
    assert tried["text"] == "do.set_state(key='capture_running', value=True)"
    assert tried["kind"] == "error", "the call is shown as one that did not happen"
    assert log_of(session).script == [], "and is not in the script"


def test_a_session_driven_from_the_drawer_alone_replays(tmp_path_factory):
    """The whole point: a log of a session nobody dispatched an action in."""
    first = session_on(tmp_path_factory.mktemp("first"))
    first.ready()

    moved(first, maximized_view="tile")
    moved(first, tile_rows=2, tile_cols=4)
    moved(first, theme_mode="light")
    moved(first, mpr_segmentation_opacity=0.25)

    second = session_on(tmp_path_factory.mktemp("second"))
    second.run(log_of(first).script)

    for key in ("maximized_view", "tile_rows", "tile_cols", "theme_mode"):
        assert second.server.state[key] == first.server.state[key], key
    assert second.server.state.mpr_segmentation_opacity == pytest.approx(0.25)


# ---------------------------------------------------------- against a page ----


@pytest.fixture
def page(tmp_path):
    """The whole app, built as ``CardioApp`` builds it, with the dock open."""
    scene = build_scene(
        tmp_path, active_volume_label="vol", view={"console_visible": True}
    )
    server, scene, logic, ui = build_app(scene)
    connect(server)
    return server, logic, ui.layout.html


def test_the_dock_reads_the_log_out_of_state(page):
    """The first thing here built from a list that grows rather than a scene."""
    _, _, html = page

    assert 'v-for="entry in console_entries"' in html
    assert ':key="entry.n"' in html


def test_the_lines_sit_in_one_child_of_the_reversed_box(page):
    """So the reversal pins the scroll without reversing the lines themselves.

    A selection follows the document, and lines laid out backwards are lines
    that have to be dragged across backwards.
    """
    _, _, html = page
    body = html[html.find("cardio-console-body") :]
    wrapper, rows = body.find("cardio-console-lines"), body.find('v-for="entry in')

    assert wrapper != -1 and rows != -1
    assert wrapper < rows


def test_what_is_written_beside_a_line_is_not_part_of_it(page):
    """The time and the count are notes about the line, and would not paste."""
    _, _, html = page

    assert "cardio-console-aside" in html
    assert "user-select: none" in (STATIC / STYLESHEET).read_text()


def test_the_prompt_hands_what_was_typed_to_the_action(page):
    """Rather than the action reading it off state, so a replay carries it."""
    _, _, html = page

    assert "@keyup.enter" in html
    assert "[console_input]" in html


def test_the_arrows_walk_the_log_from_the_prompt(page):
    """On keydown, which is the event that can stop the caret jumping with them."""
    _, _, html = page
    field = html[html.find("cardio-console-prompt") :].split("/>")[0]

    assert "@keydown.up.prevent=" in field
    assert "@keydown.down.prevent=" in field


def test_the_prompt_does_not_let_a_keystroke_reach_the_interactor(page):
    """Or typing `a` in the box would maximize the upper-left view.

    Not on ``keyup``, which is the event the submit is on: two handlers for
    one event, one of them stopping immediate propagation, is a question about
    ordering with no reason to be asked.
    """
    _, _, html = page
    field = html[html.find("cardio-console-prompt") :].split("/>")[0]

    assert "@keydown=" in field and "stopImmediatePropagation" in field
    assert "@keypress=" in field
    assert "@keyup=" not in field


def test_a_button_press_and_a_typed_call_write_the_same_line(page):
    """The claim the whole syntax exists for."""
    server, logic, _ = page

    with server.state:
        logic.dispatch("add_rotation", axis="Z")
    clicked = logic.console.log.entries[-1]["text"]

    with server.state:
        logic.dispatch("clear_console")
        logic.dispatch("run_command", text="add_rotation(axis='Z')")
    typed = logic.console.log.entries[-1]["text"]

    assert typed == clicked == "do.add_rotation(axis='Z')"


def test_the_dock_starts_where_the_drawer_stops(page):
    """Written on the element, not in the served stylesheet.

    The rules arrive as a cached asset and the drawer's width is a python
    constant, so the one thing that decides whether the log is readable is
    kept where both are already known.
    """
    _, _, html = page

    assert f"left: var(--v-layout-left, {DRAWER_WIDTH}px)" in html


def test_the_stylesheet_is_named_by_what_is_in_it():
    """Or an edit to it shows up whenever the browser's cache decides to."""
    served = {}

    class Server:
        def enable_module(self, module):
            served.update(module)

    drawer_styles(Server())

    (style,) = served["styles"]
    assert style.startswith("__cardio/drawer.css?v=")

    digest = style.split("=")[-1]
    assert len(digest) == 12 and digest.isalnum()
