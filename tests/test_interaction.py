"""Test that view events become the right controller calls.

This logic sat inside ui.py's 1050-line setup() and had no coverage; the
arithmetic it used to do now lives on the MPR controller, so the handlers can
be driven against a recording double.
"""

# Third Party
import pytest

# Internal
from cardio.ui.interaction import HANDLED_EVENTS, Interaction


class RecordingMPR:
    def __init__(self):
        self.window_level = []
        self.scrolls = []

    def adjust_window_level(self, window_delta, level_delta):
        self.window_level.append((window_delta, level_delta))

    def scroll_slice(self, view_name, distance):
        self.scrolls.append((view_name, distance))


class FakeLogic:
    def __init__(self):
        self.mpr = RecordingMPR()


class FakeState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def interaction():
    server = type(
        "Server",
        (),
        {
            "state": FakeState(
                mpr_crosshairs_enabled=True,
                help_overlay_visible=False,
                maximized_view="",
                mpr_window_level_preset=None,
            )
        },
    )()
    return Interaction(server, FakeLogic())


def press(interaction, key):
    interaction.on_event({"type": "KeyPress", "key": key})
    # defeat the debounce, so a test can press twice
    interaction.last_keypress_time.clear()


def move(interaction, view, x, y):
    interaction.on_event(
        {"type": "MouseMove", "position": {"x": x, "y": y}}, view_name=view
    )


def test_listeners_cover_every_handled_event(interaction):
    listeners = interaction.listeners_for_view("axial")
    assert set(listeners) == set(HANDLED_EVENTS)


@pytest.mark.parametrize(
    "key,view", [("v", "volume"), ("a", "axial"), ("c", "coronal"), ("s", "sagittal")]
)
def test_maximize_keys_toggle(interaction, key, view):
    press(interaction, key)
    assert interaction.server.state.maximized_view == view

    press(interaction, key)
    assert interaction.server.state.maximized_view == ""


def test_switching_directly_between_maximized_views(interaction):
    press(interaction, "a")
    press(interaction, "c")
    assert interaction.server.state.maximized_view == "coronal"


def test_l_toggles_crosshairs_and_h_toggles_help(interaction):
    press(interaction, "l")
    assert interaction.server.state.mpr_crosshairs_enabled is False

    press(interaction, "h")
    assert interaction.server.state.help_overlay_visible is True


def test_digit_keys_select_a_window_level_preset(interaction):
    press(interaction, "1")
    assert interaction.server.state.mpr_window_level_preset == 1


def test_a_digit_with_no_preset_is_ignored(interaction):
    """Presets are keyed 1-9; 0 names nothing."""
    press(interaction, "0")
    assert interaction.server.state.mpr_window_level_preset is None


def test_repeated_keys_are_debounced(interaction):
    interaction.on_event({"type": "KeyPress", "key": "a"})
    interaction.on_event({"type": "KeyPress", "key": "a"})
    assert interaction.server.state.maximized_view == "axial"


def test_left_drag_adjusts_window_and_level(interaction):
    interaction.on_event(
        {"type": "LeftButtonPress", "position": {"x": 100, "y": 100}},
        view_name="axial",
    )
    move(interaction, "axial", 110, 90)

    assert interaction.logic.mpr.window_level == [
        (-10 * interaction.window_sensitivity, 10 * interaction.level_sensitivity)
    ]


def test_right_drag_scrolls_slices(interaction):
    interaction.on_event(
        {"type": "RightButtonPress", "position": {"x": 100, "y": 100}},
        view_name="coronal",
    )
    move(interaction, "coronal", 100, 120)

    assert interaction.logic.mpr.scrolls == [
        ("coronal", 20 * interaction.slice_sensitivity)
    ]


def test_moving_without_a_button_does_nothing(interaction):
    move(interaction, "axial", 10, 10)
    assert interaction.logic.mpr.window_level == []
    assert interaction.logic.mpr.scrolls == []


def test_releasing_ends_the_drag(interaction):
    interaction.on_event(
        {"type": "LeftButtonPress", "position": {"x": 0, "y": 0}}, view_name="axial"
    )
    interaction.on_event({"type": "LeftButtonRelease"}, view_name="axial")
    move(interaction, "axial", 50, 50)

    assert interaction.logic.mpr.window_level == []


def test_dragging_in_the_volume_view_is_ignored(interaction):
    """The 3D view has its own trackball interactor."""
    interaction.on_event(
        {"type": "LeftButtonPress", "position": {"x": 0, "y": 0}}, view_name="volume"
    )
    move(interaction, "volume", 50, 50)

    assert interaction.logic.mpr.window_level == []


def test_an_empty_event_payload_is_ignored(interaction):
    interaction.on_event()
