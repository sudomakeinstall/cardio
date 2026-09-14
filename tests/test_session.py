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
import numpy as np
import pytest
from PIL import Image

# Internal
from cardio.capture.banner import band_height
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


def test_every_view_function_can_be_called_without_a_page(session):
    """The ones that draw, draw; the one that is only a page's stays empty.

    Trame raises on a controller function with no implementation rather than
    passing quietly, so this is what says a session has answered for all of
    them.
    """
    session.ready()

    for name in VIEW_FUNCTIONS:
        assert getattr(session.server.controller, name)() is None


def test_asking_a_view_to_update_draws_it(session):
    """A page renders as it pushes each view on; with no page, this does.

    Watched rather than measured: what is being checked is that the call
    reaches a render at all, which an empty implementation would not.
    """
    session.ready()
    window = session.window("ul")
    renders = []
    window.AddObserver("StartEvent", lambda *_: renders.append(1))

    session.server.controller.ul_update()

    assert renders, "the update did not render"


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


def captures(root: pl.Path) -> list[np.ndarray]:
    """Every still one capture left under ``root``, by viewport, as greyscale."""
    return [
        np.asarray(Image.open(path).convert("L"))
        for path in sorted(root.glob("out/screenshots/*/*/*"))
    ]


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


def test_a_headless_capture_is_a_picture_of_something(tmp_path):
    """The two halves a page was quietly providing: a size, and a render.

    Without a size the windows are nothing by nothing; without a render the
    frame buffer is one nothing has drawn into. Either way a capture still
    writes a file per viewport, and a test counting names would call it a
    success -- so this one looks at the pixels.
    """
    out = tmp_path / "out"
    session = session_on(tmp_path, serialization_directory=out)

    session.do("screenshot")

    written = sorted(out.glob("screenshots/*/*/*"))
    assert written, "the capture wrote nothing"
    for path in written:
        pixels = np.asarray(Image.open(path).convert("L"))
        width, height = session.scene.headless_size
        assert pixels.shape == (height, width), path
        assert pixels.max(), f"{path.parent.name} is a picture of nothing"


def test_a_banner_is_written_in_a_band_below_the_picture(tmp_path):
    """Taller by the band, and the band is not blank.

    Compared against the same capture without one rather than a known height,
    so what the band does to a picture is what is being checked.
    """
    banner = "NOT FOR CLINICAL USE"
    plain, marked = (tmp_path / "plain"), (tmp_path / "marked")
    plain.mkdir()
    marked.mkdir()

    session_on(plain, serialization_directory=plain / "out").do("screenshot")
    session_on(
        marked, serialization_directory=marked / "out", capture_banner=banner
    ).do("screenshot")

    for before, after in zip(captures(plain), captures(marked), strict=True):
        assert after.shape[1] == before.shape[1]
        assert after.shape[0] == before.shape[0] + band_height(before.shape[1])

        band = after[before.shape[0] :]
        assert band.min() < band.max(), "the band carries no lettering"


def test_the_size_a_session_renders_at_is_configured(tmp_path):
    session = session_on(tmp_path, headless_size=(320, 240))

    session.ready()

    assert session.scene.renderWindow.GetSize() == (320, 240)
    assert session.scene.mpr_views["ul"].GetSize() == (320, 240)


def test_a_config_file_is_the_scene(tmp_path):
    directory = write_objects(tmp_path)
    config = tmp_path / "cardio.toml"
    config.write_text(
        "\n".join(
            [
                "mpr_window = 1234.0",
                "mpr_level = 56.0",
                "mpr_segmentation_opacity = 0.25",
                "active_volume_label = 'vol'",
                "[[volumes]]",
                "label = 'vol'",
                f"directory = '{directory}'",
                "file_paths = ['vol0.nii.gz']",
                "[view]",
                "theme = 'light'",
                "[tile]",
                "rows = 2",
                "cols = 4",
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
    config.write_text("[tile]\nrows = 2")

    assert Scene.load(config_file=config).tile.rows == 2

    assert (
        Scene(
            volumes=[
                {"label": "vol", "directory": directory, "file_paths": ["vol0.nii.gz"]}
            ]
        ).tile.rows
        == 3
    )
