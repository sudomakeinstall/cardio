"""The app driven without a page.

What is being checked is that a session is the whole app and not a cut-down
one: the same actions, reaching the same state, through the same dispatch. The
two things a browser was quietly providing -- something to arm the change
listeners, and somewhere for a background action to run -- are the parts worth
being exact about.
"""

# System
import itertools
import pathlib as pl

# Third Party
import pytest

# Internal
from cardio.scene import Scene
from cardio.session import VIEW_FUNCTIONS, Session
from tests.test_app_smoke import build_scene, write_objects

_names = itertools.count()


def session_on(directory: pl.Path, **overrides) -> Session:
    """A session nobody has connected to, on a server nobody has started."""
    return Session(
        build_scene(directory, active_volume_label="vol", **overrides),
        server=f"session-{next(_names)}",
    )


@pytest.fixture
def session(tmp_path) -> Session:
    return session_on(tmp_path)


def test_a_session_is_built_without_a_ui(session):
    assert session.scene.mpr_views is None, "not until it is armed"
    assert set(session.actions.names), "the actions are there already"


def test_arming_builds_the_windows_the_page_would_have(session):
    session.ready()

    assert session.scene.mpr_views is not None
    assert session.scene.tile_views is not None


def test_arming_is_what_makes_an_action_reach_the_renderer(session):
    """Nothing flushes until the state is ready, listeners included."""
    session.do("add_rotation", axis="Z")

    assert session.server.state.is_ready
    assert session.scene.mpr_rotation_sequence.angles_list[0].axis == "Z"


def test_the_view_functions_may_have_no_implementation(session):
    """With no page, nothing assigns them, and trame raises on those by default."""
    session.ready()

    for name in VIEW_FUNCTIONS:
        assert getattr(session.server.controller, name)() is None


def test_an_action_sets_off_the_listeners_it_would_from_a_button(session):
    """An action mostly writes state; the flush is what comes of it.

    Dragging the window away from a preset is the listener's cue to drop the
    selection. Without a flush the write would land and nothing would follow
    it -- no slice resampled, no preset dropped.
    """
    session.do("adjust_window_level", window_delta=40.0, level_delta=-10.0)

    assert session.server.state.mpr_window_level_preset is None


def test_an_action_reaches_the_state_it_would_from_a_button(session):
    session.do("adjust_window_level", window_delta=10.0, level_delta=-5.0)

    assert session.server.state.mpr_window == pytest.approx(810.0)
    assert session.server.state.mpr_level == pytest.approx(195.0)


def test_a_sequence_runs_in_the_order_it_is_written(session):
    session.run(
        [
            ("add_rotation", {"axis": "Z"}),
            ("add_rotation", {"axis": "X"}),
            ("toggle_crosshairs", None),
        ]
    )

    axes = [step.axis for step in session.scene.mpr_rotation_sequence.angles_list]
    assert axes == ["Z", "X"]
    assert session.server.state.mpr_crosshairs_enabled is False


def test_an_unknown_action_is_refused(session):
    with pytest.raises(KeyError):
        session.do("nonsense")


def test_a_capture_has_written_before_the_next_action_begins(tmp_path):
    """The one action that returns before it has done anything.

    A capture is scheduled on the event loop; from a script there is nobody to
    return to while it runs, so the session waits for it.
    """
    out = tmp_path / "out"
    session = session_on(tmp_path, serialization_directory=out)

    session.do("screenshot")

    written = sorted(path.name for path in out.glob("screenshots/*/*"))
    assert written, "the capture wrote nothing"
    assert session.server.state.capture_ok
    assert not session.server.state.capture_running


def test_a_config_file_is_the_scene(tmp_path):
    directory = write_objects(tmp_path)
    config = tmp_path / "cardio.toml"
    config.write_text(
        "\n".join(
            [
                "mpr_window = 1234.0",
                "mpr_level = 56.0",
                "mpr_segmentation_opacity = 0.25",
                "tile_rows = 2",
                "tile_cols = 4",
                "active_volume_label = 'vol'",
                "[[volumes]]",
                "label = 'vol'",
                f"directory = '{directory}'",
                "file_paths = ['vol0.nii.gz']",
                "[view]",
                "theme = 'light'",
            ]
        )
    )

    session = Session.from_config(config, server=f"session-{next(_names)}")
    session.ready()

    assert session.server.state.mpr_segmentation_opacity == 0.25
    assert session.server.state.mpr_window == 1234.0
    assert session.server.state.mpr_level == 56.0
    assert session.server.state.tile_rows == 2
    assert session.server.state.tile_cols == 4
    assert session.server.state.theme_mode == "light"


def test_a_scene_built_after_a_config_read_is_not_still_reading_it(tmp_path):
    """The sources are handed to pydantic as class attributes and taken back."""
    directory = write_objects(tmp_path)
    config = tmp_path / "cardio.toml"
    config.write_text("tile_rows = 2")

    assert Scene.load(config_file=config).tile_rows == 2

    assert (
        Scene(
            volumes=[
                {"label": "vol", "directory": directory, "file_paths": ["vol0.nii.gz"]}
            ]
        ).tile_rows
        == 3
    )
