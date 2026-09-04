"""Test that view events become the right controller calls.

This logic sat inside ui.py's 1050-line setup() and had no coverage; the
arithmetic it used to do now lives on the MPR controller, so the handlers can
be driven against a recording double.
"""

# System
import math

# Third Party
import pytest

# Internal
from cardio.ui.interaction import HANDLED_EVENTS, Interaction
from tests.fakes import FakeState


class RecordingMPR:
    def __init__(self):
        self.window_level = []
        self.scrolls = []
        self.pans = []
        self.rotations = []
        self.zooms = []

    def pan_view(self, view_name, dx, dy):
        self.pans.append((view_name, dx, dy))

    def rotate_view(self, view_name, start, end):
        self.rotations.append((view_name, list(start), list(end)))

    def zoom_views(self, factor):
        self.zooms.append(factor)

    def adjust_window_level(self, window_delta, level_delta):
        self.window_level.append((window_delta, level_delta))

    def scroll_slice(self, view_name, distance):
        self.scrolls.append((view_name, distance))


class RecordingTiles:
    def __init__(self):
        self.zooms = []

    def zoom_tiles(self, factor):
        self.zooms.append(factor)


class FakeLogic:
    def __init__(self):
        self.mpr = RecordingMPR()
        self.tiles = RecordingTiles()


@pytest.fixture
def interaction():
    server = type(
        "Server",
        (),
        {
            "state": FakeState(
                mpr_crosshairs_enabled=True,
                help_overlay_visible=False,
                metadata_overlay_visible=False,
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


def press_buttons(interaction, view, x, y):
    """Put both buttons down at one point, as a slice scroll starts."""
    for button in ("LeftButtonPress", "RightButtonPress"):
        interaction.on_event(
            {"type": button, "position": {"x": x, "y": y}}, view_name=view
        )


def move(interaction, view, x, y):
    interaction.on_event(
        {"type": "MouseMove", "position": {"x": x, "y": y}}, view_name=view
    )


def wheel(interaction, view, spin_y):
    interaction.on_event(
        {"type": "MouseWheel", "spinY": spin_y, "position": {"x": 50, "y": 50}},
        view_name=view,
    )


def test_listeners_cover_every_handled_event(interaction):
    listeners = interaction.listeners_for_view("axial")
    assert set(listeners) == set(HANDLED_EVENTS)


@pytest.mark.parametrize(
    "key,view",
    [
        ("v", "volume"),
        ("a", "axial"),
        ("c", "coronal"),
        ("s", "sagittal"),
        ("t", "tile"),
    ],
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


def test_i_toggles_the_metadata_sheet(interaction):
    press(interaction, "i")
    assert interaction.server.state.metadata_overlay_visible is True

    press(interaction, "i")
    assert interaction.server.state.metadata_overlay_visible is False


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


def test_both_buttons_drag_scrolls_slices(interaction):
    press_buttons(interaction, "coronal", 100, 100)
    move(interaction, "coronal", 100, 120)

    assert interaction.logic.mpr.scrolls == [
        ("coronal", 20 * interaction.slice_sensitivity)
    ]
    assert interaction.logic.mpr.window_level == []


def test_right_drag_alone_does_nothing(interaction):
    interaction.on_event(
        {"type": "RightButtonPress", "position": {"x": 100, "y": 100}},
        view_name="coronal",
    )
    move(interaction, "coronal", 110, 120)

    assert interaction.logic.mpr.scrolls == []
    assert interaction.logic.mpr.window_level == []


def test_releasing_the_right_button_resumes_window_level(interaction):
    """The remaining drag continues from where it is, without a jump."""
    press_buttons(interaction, "axial", 100, 100)
    move(interaction, "axial", 100, 120)
    interaction.on_event({"type": "RightButtonRelease"}, view_name="axial")
    move(interaction, "axial", 110, 110)

    assert interaction.logic.mpr.window_level == [
        (-10 * interaction.window_sensitivity, 10 * interaction.level_sensitivity)
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


def test_left_drag_over_the_tile_grid_windows_every_tile(interaction):
    interaction.on_event(
        {"type": "LeftButtonPress", "position": {"x": 100, "y": 100}},
        view_name="tile",
    )
    move(interaction, "tile", 110, 90)

    assert interaction.logic.mpr.window_level == [
        (-10 * interaction.window_sensitivity, 10 * interaction.level_sensitivity)
    ]


def test_both_buttons_over_the_tile_grid_do_nothing(interaction):
    """A grid of cuts along a path has no single slice to move."""
    press_buttons(interaction, "tile", 100, 100)
    move(interaction, "tile", 110, 120)

    assert interaction.logic.mpr.scrolls == []
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


def test_wheel_scrolls_slices(interaction):
    """vtk.js normalises a notch to a spin of one."""
    wheel(interaction, "axial", 1.0)

    assert interaction.logic.mpr.scrolls == [
        ("axial", 1.0 * interaction.wheel_sensitivity)
    ]


def test_wheel_reverses_with_the_spin_direction(interaction):
    wheel(interaction, "sagittal", -1.0)

    assert interaction.logic.mpr.scrolls == [
        ("sagittal", -1.0 * interaction.wheel_sensitivity)
    ]


def test_a_trackpad_spin_scrolls_proportionally(interaction):
    wheel(interaction, "coronal", 0.25)

    assert interaction.logic.mpr.scrolls == [
        ("coronal", 0.25 * interaction.wheel_sensitivity)
    ]


def test_the_wheel_travels_the_same_way_as_an_upward_drag(interaction):
    """The two slice-scroll gestures must not fight each other."""
    press_buttons(interaction, "axial", 100, 100)
    move(interaction, "axial", 100, 110)
    wheel(interaction, "axial", 1.0)

    dragged, wheeled = (distance for _, distance in interaction.logic.mpr.scrolls)
    assert dragged > 0 and wheeled > 0


@pytest.mark.parametrize("view", ["tile", "volume"])
def test_wheel_outside_the_mpr_views_is_ignored(interaction, view):
    """The tile grid has no single slice, and the 3D view zooms itself."""
    wheel(interaction, view, -1.0)

    assert interaction.logic.mpr.scrolls == []


def test_a_wheel_event_without_a_spin_is_ignored(interaction):
    interaction.on_event({"type": "MouseWheel"}, view_name="axial")

    assert interaction.logic.mpr.scrolls == []


def test_the_wheel_does_not_disturb_window_level(interaction):
    wheel(interaction, "axial", -1.0)

    assert interaction.logic.mpr.window_level == []


def press_middle(interaction, view, x, y):
    interaction.on_event(
        {"type": "MiddleButtonPress", "position": {"x": x, "y": y}}, view_name=view
    )


def test_middle_drag_pans(interaction):
    press_middle(interaction, "axial", 100, 100)
    move(interaction, "axial", 110, 90)

    assert interaction.logic.mpr.pans == [("axial", 10, -10)]
    assert interaction.logic.mpr.window_level == []
    assert interaction.logic.mpr.scrolls == []


def test_middle_drag_pans_one_to_one_with_the_cursor(interaction):
    """Panning is a grab, so the delta reaches the view unscaled."""
    press_middle(interaction, "coronal", 0, 0)
    move(interaction, "coronal", 37, 11)

    assert interaction.logic.mpr.pans == [("coronal", 37, 11)]


@pytest.mark.parametrize("view", ["tile", "volume"])
def test_middle_drag_outside_the_mpr_views_does_not_pan(interaction, view):
    press_middle(interaction, view, 100, 100)
    move(interaction, view, 110, 90)

    assert interaction.logic.mpr.pans == []


def press_left_middle(interaction, view, x, y):
    for button in ("LeftButtonPress", "MiddleButtonPress"):
        interaction.on_event(
            {"type": button, "position": {"x": x, "y": y}}, view_name=view
        )


def press_right_middle(interaction, view, x, y):
    for button in ("RightButtonPress", "MiddleButtonPress"):
        interaction.on_event(
            {"type": button, "position": {"x": x, "y": y}}, view_name=view
        )


def test_left_and_middle_rotates(interaction):
    """Rotation gets both positions: it is an angle swept, not a distance."""
    press_left_middle(interaction, "axial", 100, 100)
    move(interaction, "axial", 110, 90)

    assert interaction.logic.mpr.rotations == [("axial", [100, 100], [110, 90])]


def test_each_move_rotates_from_where_the_last_one_left_off(interaction):
    press_left_middle(interaction, "axial", 100, 100)
    move(interaction, "axial", 110, 90)
    move(interaction, "axial", 130, 70)

    assert interaction.logic.mpr.rotations == [
        ("axial", [100, 100], [110, 90]),
        ("axial", [110, 90], [130, 70]),
    ]


def test_rotating_is_not_also_a_pan_or_a_window_level(interaction):
    """The middle button is in three gestures; only one may fire."""
    press_left_middle(interaction, "axial", 100, 100)
    move(interaction, "axial", 110, 90)

    assert interaction.logic.mpr.pans == []
    assert interaction.logic.mpr.window_level == []


def test_right_and_middle_zooms(interaction):
    press_right_middle(interaction, "axial", 100, 100)
    move(interaction, "axial", 100, 120)

    assert interaction.logic.mpr.zooms == [
        pytest.approx(math.exp(20 * interaction.zoom_sensitivity))
    ]


def test_dragging_up_zooms_in_and_down_zooms_out(interaction):
    press_right_middle(interaction, "axial", 100, 100)
    move(interaction, "axial", 100, 120)
    move(interaction, "axial", 100, 80)

    zoomed_in, zoomed_out = interaction.logic.mpr.zooms
    assert zoomed_in > 1.0
    assert zoomed_out < 1.0


def test_zoom_ignores_horizontal_movement(interaction):
    press_right_middle(interaction, "axial", 100, 100)
    move(interaction, "axial", 150, 100)

    assert interaction.logic.mpr.zooms == [pytest.approx(1.0)]


def test_rotating_needs_a_single_slice(interaction):
    """There is no one plane to spin over the tile grid or the 3D view."""
    for view in ("tile", "volume"):
        press_left_middle(interaction, view, 100, 100)
        move(interaction, view, 110, 90)

    assert interaction.logic.mpr.rotations == []


def test_zooming_over_the_tile_grid_zooms_the_grid(interaction):
    """Zoom is about a grid of views, so the tiles take it as the MPRs do."""
    press_right_middle(interaction, "tile", 100, 100)
    move(interaction, "tile", 100, 120)

    assert interaction.logic.tiles.zooms == [
        pytest.approx(math.exp(20 * interaction.zoom_sensitivity))
    ]
    assert interaction.logic.mpr.zooms == []


def test_zooming_over_an_mpr_view_leaves_the_tile_grid_alone(interaction):
    press_right_middle(interaction, "axial", 100, 100)
    move(interaction, "axial", 100, 120)

    assert interaction.logic.tiles.zooms == []
    assert interaction.logic.mpr.zooms != []


def test_zooming_over_the_volume_view_zooms_nothing(interaction):
    """The 3D view has its own trackball, and zooms itself."""
    press_right_middle(interaction, "volume", 100, 100)
    move(interaction, "volume", 100, 120)

    assert interaction.logic.mpr.zooms == []
    assert interaction.logic.tiles.zooms == []


def test_releasing_the_middle_button_returns_to_window_level(interaction):
    press_left_middle(interaction, "axial", 100, 100)
    move(interaction, "axial", 110, 90)
    interaction.on_event({"type": "MiddleButtonRelease"}, view_name="axial")
    move(interaction, "axial", 120, 80)

    assert len(interaction.logic.mpr.rotations) == 1
    assert interaction.logic.mpr.window_level == [
        (-10 * interaction.window_sensitivity, 10 * interaction.level_sensitivity)
    ]


def test_releasing_the_middle_button_ends_the_pan(interaction):
    press_middle(interaction, "axial", 100, 100)
    interaction.on_event({"type": "MiddleButtonRelease"}, view_name="axial")
    move(interaction, "axial", 150, 150)

    assert interaction.logic.mpr.pans == []
