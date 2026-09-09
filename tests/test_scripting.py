"""The log as a file that runs it again.

The claim worth being exact about is that a script line is the console line
with a prefix and nothing else -- that a log is *copied* into a script rather
than translated into one. Everything else here is what has to be true for the
copy to run: it names a scene, it names the one the session opened with, and
running it puts a fresh app where the first one ended.
"""

# System
import runpy

# Third Party
import pytest
import tomlkit as tk

# Internal
import cardio.console as console
import cardio.scripting as scripting
from tests.test_console import CALLS
from tests.test_session import session_on

STEPS = [
    ("add_rotation", {"axis": "Z"}),
    ("add_rotation", {"axis": "X"}),
    ("toggle_crosshairs", {}),
    ("set_window_level_preset", {"preset": 3}),
]


@pytest.fixture
def session(tmp_path):
    """A session writing its output under the test's own directory.

    ``serialization_directory`` defaults to ``./data``, which every test would
    otherwise share -- and a test that counts the files it wrote would count
    the ones another test left behind.
    """
    return session_on(tmp_path, serialization_directory=tmp_path / "out")


def log_of(session):
    return session.logic.console.log


# ------------------------------------------------------------- the syntax ----


@pytest.mark.parametrize("name,arguments", CALLS, ids=[name for name, _ in CALLS])
def test_a_script_line_is_the_console_line_with_a_prefix(name, arguments):
    """The whole feature in one line: the log and the script are one language."""
    assert console.format_call(name, arguments, prefix="do.") == "do." + (
        console.format_call(name, arguments)
    )


def test_a_script_line_is_a_call_the_prompt_would_take_back():
    """Strip the prefix and it is the console's line again, parse and all."""
    line = console.format_call("add_rotation", {"axis": "Z"}, prefix="do.")

    assert console.parse_call(line.removeprefix("do.")) == (
        "add_rotation",
        [],
        {"axis": "Z"},
    )


# ------------------------------------------------------------ the rendering ----


def test_a_script_names_the_config_it_is_asked_against():
    text = scripting.render(STEPS, "session.toml")

    assert "import cardio" in text
    assert "cardio.script(pl.Path(__file__).parent / 'session.toml')" in text


def test_every_call_is_a_line_in_the_order_it_happened():
    body = [
        line
        for line in scripting.render(STEPS, "session.toml").splitlines()
        if line.startswith("do.")
    ]

    assert body == [
        "do.add_rotation(axis='Z')",
        "do.add_rotation(axis='X')",
        "do.toggle_crosshairs()",
        "do.set_window_level_preset(preset=3)",
    ]


def test_closing_the_application_is_not_a_line_of_a_script():
    """A script ends by ending; stopping the server is not a step in it."""
    text = scripting.render([*STEPS, ("close_application", {})], "session.toml")

    assert "close_application" not in text


def test_a_log_that_lost_its_beginning_says_so():
    text = scripting.render(STEPS, "session.toml", dropped=7)

    assert "first 7 lines" in text


def test_a_log_that_lost_nothing_says_nothing():
    body = scripting.render(STEPS, "session.toml").split("cardio.script")[1]

    assert "#" not in body


# ---------------------------------------------------------------- the log ----


def test_the_log_counts_what_falls_off_the_far_end():
    log = console.Log(limit=3)

    for i in range(10):
        log.record(f"action_{i}", {})

    assert log.dropped == 7


def test_clearing_the_log_is_losing_it_too():
    """A script written after a clear does not start where the session did."""
    log = console.Log()
    log.record("add_rotation", {"axis": "Z"})

    log.clear()

    assert log.dropped == 1


# ------------------------------------------------------- against a session ----


def test_a_session_says_what_config_would_open_it_again(session):
    assert "[[volumes]]" in session.opened_as


def test_the_scene_a_session_opened_with_is_not_the_one_it_ends_showing(session):
    """An action moves the scene under it; a script needs where it began."""
    before = session.opened_as

    session.do("add_rotation", axis="Z")

    assert session.scene.mpr_rotation_sequence.angles_list, "the scene moved"
    assert session.opened_as == before


def test_saving_writes_the_script_and_the_scene_it_opens(session):
    session.run(STEPS)

    session.do("save_script")

    written = sorted(path.suffix for path in session.scene.scripts_directory.iterdir())
    assert written == [".py", ".toml"]


def test_saving_says_where_it_went(session):
    session.run(STEPS)

    session.do("save_script")

    assert session.server.state.script_saved_at
    assert session.server.state.script_summary.startswith(f"{len(STEPS)} calls to ")


def test_a_script_does_not_record_its_own_saving(session):
    """Which would put a save in the file, and a save in the file after that."""
    session.run(STEPS)

    session.do("save_script")

    assert "save_script" not in _script_text(session)


def test_the_scene_written_beside_a_script_opens_where_the_session_did(session):
    session.do("add_rotation", axis="Z")

    session.do("save_script")

    (config,) = session.scene.scripts_directory.glob("*.toml")
    written = tk.parse(config.read_text())

    assert written["mpr_rotation_sequence"]["angles_list"] == []
    assert session.scene.mpr_rotation_sequence.angles_list, "the session's has one"


def _script_text(session) -> str:
    (path,) = session.scene.scripts_directory.glob("*.py")
    return path.read_text()


# ---------------------------------------------------------------- running ----


def test_an_unknown_action_on_the_proxy_says_what_is_known(session):
    do = scripting.Actions(session)

    with pytest.raises(AttributeError, match="No such action: 'nonsense'"):
        unknown = do.nonsense

    assert callable(do.add_rotation), "and a known one is a call"


def test_the_proxy_asks_for_the_action_a_button_would(session):
    do = scripting.Actions(session)

    do.add_rotation(axis="Z")

    assert [step.axis for step in session.scene.mpr_rotation_sequence.angles_list] == [
        "Z"
    ]


def test_a_saved_script_run_again_reproduces_the_session(session, tmp_path):
    """The feature, end to end: through a file, on a session of its own."""
    session.run(STEPS)
    session.do("save_script")

    (script,) = session.scene.scripts_directory.glob("*.py")
    again = runpy.run_path(str(script))["do"].session

    for key in ("mpr_crosshairs_enabled", "mpr_window", "mpr_level"):
        assert again.server.state[key] == session.server.state[key], key
    assert [step.axis for step in again.scene.mpr_rotation_sequence.angles_list] == [
        step.axis for step in session.scene.mpr_rotation_sequence.angles_list
    ]
