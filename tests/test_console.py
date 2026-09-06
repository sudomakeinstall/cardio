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
from cardio.ui.common import DRAWER_WIDTH, drawer_styles
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
    assert entry["text"] == "add_rotation(axis='Z')"


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
        "toggle_console()",
        "add_rotation(axis='Z')",
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

    assert texts(session) == ["set_state(key='tile_cols', value=4)"]


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
        "set_state(key='tile_cols', value=4)",
        "set_state(key='tile_rows', value=2)",
    ]


def test_an_action_is_not_written_down_twice(session):
    """It moves document keys too, and the log already names it."""
    session.ready()

    session.do("toggle_crosshairs")

    assert texts(session) == ["toggle_crosshairs()"], (
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
    assert entry["text"] == "set_state(key='tile_cols', value=4)"


def test_the_cameras_moving_is_not_something_anybody_asked_for(session):
    """They are refitted by a layout change and turned by a trackball drag.

    The state is only ever a note of where VTK's ended up, and the one time a
    person means to move one, place_camera writes that down as the action.
    """
    session.ready()

    moved(session, maximized_view="tile")

    assert texts(session) == ["set_state(key='maximized_view', value='tile')"]


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

    assert texts(session) == ["set_state(key='frame', value=1)"]


def test_set_state_refuses_a_key_that_is_not_the_document(session):
    """Session state is not a thing a script has business reaching into."""
    session.ready()

    session.do("run_command", text="set_state(key='capture_running', value=True)")

    said, tried = log_of(session).entries
    assert "not a document key" in said["text"]
    assert tried["text"] == "set_state(key='capture_running', value=True)"
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


def test_the_prompt_hands_what_was_typed_to_the_action(page):
    """Rather than the action reading it off state, so a replay carries it."""
    _, _, html = page

    assert "@keyup.enter" in html
    assert "[console_input]" in html


def test_the_prompt_does_not_let_a_keystroke_reach_the_interactor(page):
    """Or typing `a` in the box would maximize the axial view.

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
    clicked = logic.console.log.entries[0]["text"]

    with server.state:
        logic.dispatch("clear_console")
        logic.dispatch("run_command", text="add_rotation(axis='Z')")
    typed = logic.console.log.entries[0]["text"]

    assert typed == clicked == "add_rotation(axis='Z')"


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
