"""Test that view events become the right named actions.

This logic sat inside ui.py's 1050-line setup() and had no coverage; the
arithmetic it used to do now lives on the MPR controller, and which controller
that is is no longer Interaction's business either -- it names an action and
dispatches it.
"""

# System
import math
import pathlib as pl
import re
import types

# Third Party
import pytest

# Internal
import cardio.ui.interaction as interaction_module
from cardio.planimetry import CLICK_SLOP
from cardio.ui.interaction import (
    CROSSHAIR_KEY,
    DOUBLE_CLICK_MS,
    FRAME_KEYS,
    HANDLED_EVENTS,
    HELP_KEY,
    MAXIMIZE_KEYS,
    SLICE_KEYS,
    VIEW_LAYOUTS,
    Interaction,
)


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
        self.focused = []

    def zoom_tiles(self, factor):
        self.zooms.append(factor)

    def toggle_tile_focus(self, x, y):
        self.focused.append((x, y))


class FakeServer:
    """The one thing Interaction reads rather than reports: the tracing mode.

    A click means something different while a region is being traced, and
    saying so on every click of a gesture that is not tracing would fill the
    action log with it -- so this is read here rather than dispatched blind.
    """

    def __init__(self, **state):
        self.state = types.SimpleNamespace(
            measuring=False, measurement_editing=None, **state
        )


class FakeLogic:
    """Interaction's whole view of Logic: ``dispatch``, the mode, and no more.

    Every call is recorded by name. A gesture is also played into the recorder
    that owns it, and the recorders keep the real methods' parameter names --
    so dispatching an argument an action does not take fails here rather than
    passing quietly.
    """

    def __init__(self):
        self.mpr = RecordingMPR()
        self.tiles = RecordingTiles()
        self.server = FakeServer()
        self.calls = []
        # What ``grab_measurement_point`` answers. It is the one action whose
        # return the gesture layer reads, so it is the one a fake has to have
        # an opinion about.
        self.grabs = False

    def dispatch(self, name, **arguments):
        self.calls.append((name, arguments))
        for recorder in (self.mpr, self.tiles):
            method = getattr(recorder, name, None)
            if method is not None:
                method(**arguments)
        return self.grabs if name == "grab_measurement_point" else None

    @property
    def names(self):
        return [name for name, _ in self.calls]

    def arguments(self, name):
        """What each call to ``name`` was made with, in order."""
        return [arguments for called, arguments in self.calls if called == name]


@pytest.fixture
def interaction():
    return Interaction(FakeLogic())


def press(interaction, key, view=None):
    interaction.on_event({"type": "KeyPress", "key": key}, view_name=view)
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


def listed_keys() -> set[str]:
    """Every key the shortcut sheet writes down, as it writes it."""
    source = (pl.Path(interaction_module.__file__).parent / "help.py").read_text()
    return set(re.findall(r'html\.Td\("(\S)"\)', source))


def test_listeners_cover_every_handled_event(interaction):
    listeners = interaction.listeners_for_view("ul")
    assert set(listeners) == set(HANDLED_EVENTS)


@pytest.mark.parametrize(
    "key,view",
    [
        ("v", "volume"),
        ("a", "ul"),
        ("c", "ll"),
        ("s", "lr"),
        ("t", "tile"),
        ("y", "volumetry"),
    ],
)
def test_maximize_keys_name_the_view_they_maximize(interaction, key, view):
    """Whether a second press maximizes or restores is the action's business."""
    press(interaction, key)
    assert interaction.logic.arguments("toggle_maximized") == [{"view": view}]


def test_the_help_sheet_lists_every_key_that_maximizes_a_view():
    """The shortcut table is written out by hand, one row at a time.

    So a key added to ``MAXIMIZE_KEYS`` reaches the app and never reaches the
    sheet that is the only place it is written down -- which is not something
    anybody notices, because the key works.
    """
    assert set(MAXIMIZE_KEYS) <= listed_keys()


def test_the_help_sheet_lists_every_key_that_is_bound():
    """Same reason, for the keys that are not about maximizing a view.

    ``?`` and ``+`` are here because hjkl displaced them, and a reference that
    still says ``h`` opens the help is worse than no reference at all.
    """
    bound = set(FRAME_KEYS) | set(SLICE_KEYS) | {HELP_KEY, CROSSHAIR_KEY}

    assert bound <= listed_keys()


def test_plus_toggles_crosshairs_and_question_toggles_help(interaction):
    """Both were letters until hjkl wanted them."""
    press(interaction, CROSSHAIR_KEY)
    press(interaction, HELP_KEY)
    assert interaction.logic.names == ["toggle_crosshairs", "toggle_help"]


@pytest.mark.parametrize(
    "key, action",
    [("h", "decrement_frame"), ("l", "increment_frame")],
)
def test_h_and_l_step_the_frame(interaction, key, action):
    press(interaction, key, view="ul")
    assert interaction.logic.names == [action]


def test_the_frame_keys_do_not_need_a_view(interaction):
    """A frame belongs to the study rather than to any one cut, so a press
    over the volume or the tiles steps it too."""
    press(interaction, "l", view="volume")
    assert interaction.logic.names == ["increment_frame"]


@pytest.mark.parametrize("key, distance", [("k", 1.0), ("j", -1.0)])
def test_j_and_k_scroll_the_slice_under_the_cursor(interaction, key, distance):
    press(interaction, key, view="lr")
    assert interaction.logic.arguments("scroll_slice") == [
        {"view_name": "lr", "distance": distance}
    ]


def test_k_travels_the_way_an_upward_wheel_does(interaction):
    """The key and the wheel are the same gesture, so they are signed alike."""
    press(interaction, "k", view="ul")
    wheel(interaction, "ul", 1)

    distances = [
        call["distance"] for call in interaction.logic.arguments("scroll_slice")
    ]
    assert len(distances) == 2 and distances[0] > 0 and distances[1] > 0


@pytest.mark.parametrize("view", ["volume", "tile", "volumetry", None])
def test_the_slice_keys_are_ignored_where_there_is_no_slice(interaction, view):
    press(interaction, "j", view=view)
    assert interaction.logic.calls == []


def test_i_toggles_the_metadata_sheet(interaction):
    press(interaction, "i")
    assert interaction.logic.arguments("toggle_metadata") == [{}]


def test_digit_keys_select_a_window_level_preset(interaction):
    press(interaction, "1")
    assert interaction.logic.arguments("set_window_level_preset") == [{"preset": 1}]


def test_a_digit_with_no_preset_is_ignored(interaction):
    """Presets are keyed 1-9; 0 names nothing."""
    press(interaction, "0")
    assert interaction.logic.calls == []


def test_repeated_keys_are_debounced(interaction):
    interaction.on_event({"type": "KeyPress", "key": "a"})
    interaction.on_event({"type": "KeyPress", "key": "a"})
    assert interaction.logic.arguments("toggle_maximized") == [{"view": "ul"}]


def test_left_drag_adjusts_window_and_level(interaction):
    interaction.on_event(
        {"type": "LeftButtonPress", "position": {"x": 100, "y": 100}},
        view_name="ul",
    )
    move(interaction, "ul", 110, 90)

    assert interaction.logic.mpr.window_level == [
        (-10 * interaction.window_sensitivity, 10 * interaction.level_sensitivity)
    ]


def test_both_buttons_drag_scrolls_slices(interaction):
    press_buttons(interaction, "ll", 100, 100)
    move(interaction, "ll", 100, 120)

    assert interaction.logic.mpr.scrolls == [("ll", 20 * interaction.slice_sensitivity)]
    assert interaction.logic.mpr.window_level == []


def test_right_drag_alone_does_nothing(interaction):
    interaction.on_event(
        {"type": "RightButtonPress", "position": {"x": 100, "y": 100}},
        view_name="ll",
    )
    move(interaction, "ll", 110, 120)

    assert interaction.logic.mpr.scrolls == []
    assert interaction.logic.mpr.window_level == []


def test_releasing_the_right_button_resumes_window_level(interaction):
    """The remaining drag continues from where it is, without a jump."""
    press_buttons(interaction, "ul", 100, 100)
    move(interaction, "ul", 100, 120)
    interaction.on_event({"type": "RightButtonRelease"}, view_name="ul")
    move(interaction, "ul", 110, 110)

    assert interaction.logic.mpr.window_level == [
        (-10 * interaction.window_sensitivity, 10 * interaction.level_sensitivity)
    ]


def test_moving_without_a_button_does_nothing(interaction):
    move(interaction, "ul", 10, 10)
    assert interaction.logic.mpr.window_level == []
    assert interaction.logic.mpr.scrolls == []


def test_releasing_ends_the_drag(interaction):
    interaction.on_event(
        {"type": "LeftButtonPress", "position": {"x": 0, "y": 0}}, view_name="ul"
    )
    interaction.on_event({"type": "LeftButtonRelease"}, view_name="ul")
    move(interaction, "ul", 50, 50)

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
    wheel(interaction, "ul", 1.0)

    assert interaction.logic.mpr.scrolls == [
        ("ul", 1.0 * interaction.wheel_sensitivity)
    ]


def test_wheel_reverses_with_the_spin_direction(interaction):
    wheel(interaction, "lr", -1.0)

    assert interaction.logic.mpr.scrolls == [
        ("lr", -1.0 * interaction.wheel_sensitivity)
    ]


def test_a_trackpad_spin_scrolls_proportionally(interaction):
    wheel(interaction, "ll", 0.25)

    assert interaction.logic.mpr.scrolls == [
        ("ll", 0.25 * interaction.wheel_sensitivity)
    ]


def test_the_wheel_travels_the_same_way_as_an_upward_drag(interaction):
    """The two slice-scroll gestures must not fight each other."""
    press_buttons(interaction, "ul", 100, 100)
    move(interaction, "ul", 100, 110)
    wheel(interaction, "ul", 1.0)

    dragged, wheeled = (distance for _, distance in interaction.logic.mpr.scrolls)
    assert dragged > 0 and wheeled > 0


@pytest.mark.parametrize("view", ["tile", "volume"])
def test_wheel_outside_the_mpr_views_is_ignored(interaction, view):
    """The tile grid has no single slice, and the 3D view zooms itself."""
    wheel(interaction, view, -1.0)

    assert interaction.logic.mpr.scrolls == []


def test_a_wheel_event_without_a_spin_is_ignored(interaction):
    interaction.on_event({"type": "MouseWheel"}, view_name="ul")

    assert interaction.logic.mpr.scrolls == []


def test_the_wheel_does_not_disturb_window_level(interaction):
    wheel(interaction, "ul", -1.0)

    assert interaction.logic.mpr.window_level == []


def press_middle(interaction, view, x, y):
    interaction.on_event(
        {"type": "MiddleButtonPress", "position": {"x": x, "y": y}}, view_name=view
    )


def test_middle_drag_pans(interaction):
    press_middle(interaction, "ul", 100, 100)
    move(interaction, "ul", 110, 90)

    assert interaction.logic.mpr.pans == [("ul", 10, -10)]
    assert interaction.logic.mpr.window_level == []
    assert interaction.logic.mpr.scrolls == []


def test_middle_drag_pans_one_to_one_with_the_cursor(interaction):
    """Panning is a grab, so the delta reaches the view unscaled."""
    press_middle(interaction, "ll", 0, 0)
    move(interaction, "ll", 37, 11)

    assert interaction.logic.mpr.pans == [("ll", 37, 11)]


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
    press_left_middle(interaction, "ul", 100, 100)
    move(interaction, "ul", 110, 90)

    assert interaction.logic.mpr.rotations == [("ul", [100, 100], [110, 90])]


def test_each_move_rotates_from_where_the_last_one_left_off(interaction):
    press_left_middle(interaction, "ul", 100, 100)
    move(interaction, "ul", 110, 90)
    move(interaction, "ul", 130, 70)

    assert interaction.logic.mpr.rotations == [
        ("ul", [100, 100], [110, 90]),
        ("ul", [110, 90], [130, 70]),
    ]


def test_rotating_is_not_also_a_pan_or_a_window_level(interaction):
    """The middle button is in three gestures; only one may fire."""
    press_left_middle(interaction, "ul", 100, 100)
    move(interaction, "ul", 110, 90)

    assert interaction.logic.mpr.pans == []
    assert interaction.logic.mpr.window_level == []


def test_right_and_middle_zooms(interaction):
    press_right_middle(interaction, "ul", 100, 100)
    move(interaction, "ul", 100, 120)

    assert interaction.logic.mpr.zooms == [
        pytest.approx(math.exp(20 * interaction.zoom_sensitivity))
    ]


def test_dragging_up_zooms_in_and_down_zooms_out(interaction):
    press_right_middle(interaction, "ul", 100, 100)
    move(interaction, "ul", 100, 120)
    move(interaction, "ul", 100, 80)

    zoomed_in, zoomed_out = interaction.logic.mpr.zooms
    assert zoomed_in > 1.0
    assert zoomed_out < 1.0


def test_zoom_ignores_horizontal_movement(interaction):
    press_right_middle(interaction, "ul", 100, 100)
    move(interaction, "ul", 150, 100)

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
    press_right_middle(interaction, "ul", 100, 100)
    move(interaction, "ul", 100, 120)

    assert interaction.logic.tiles.zooms == []
    assert interaction.logic.mpr.zooms != []


def test_zooming_over_the_volume_view_zooms_nothing(interaction):
    """The 3D view has its own trackball, and zooms itself."""
    press_right_middle(interaction, "volume", 100, 100)
    move(interaction, "volume", 100, 120)

    assert interaction.logic.mpr.zooms == []
    assert interaction.logic.tiles.zooms == []


def test_releasing_the_middle_button_returns_to_window_level(interaction):
    press_left_middle(interaction, "ul", 100, 100)
    move(interaction, "ul", 110, 90)
    interaction.on_event({"type": "MiddleButtonRelease"}, view_name="ul")
    move(interaction, "ul", 120, 80)

    assert len(interaction.logic.mpr.rotations) == 1
    assert interaction.logic.mpr.window_level == [
        (-10 * interaction.window_sensitivity, 10 * interaction.level_sensitivity)
    ]


def test_releasing_the_middle_button_ends_the_pan(interaction):
    press_middle(interaction, "ul", 100, 100)
    interaction.on_event({"type": "MiddleButtonRelease"}, view_name="ul")
    move(interaction, "ul", 150, 150)

    assert interaction.logic.mpr.pans == []


# --- tracing a region ---------------------------------------------------------

# A click is the one gesture none of the existing ones is: window/level, the
# zoom, the slice scroll and the rotation all travel. So tracing takes the click
# and leaves every drag alone, and this is where that is held to.


def measuring(interaction) -> Interaction:
    interaction.logic.server.state.measuring = True
    return interaction


def click(interaction, view, x, y, button="Left", travel=0):
    interaction.on_event(
        {"type": f"{button}ButtonPress", "position": {"x": x, "y": y}},
        view_name=view,
    )
    interaction.on_event(
        {"type": f"{button}ButtonRelease", "position": {"x": x + travel, "y": y}},
        view_name=view,
    )


def placed(interaction) -> list:
    return [
        args
        for name, args in interaction.logic.calls
        if name == "place_measurement_point"
    ]


def test_a_click_on_a_cut_places_a_point(interaction):
    click(measuring(interaction), "ul", 120, 90)

    assert placed(interaction) == [{"view_name": "ul", "x": 120, "y": 90}]


def test_a_click_outside_the_mode_places_nothing(interaction):
    click(interaction, "ul", 120, 90)

    assert placed(interaction) == []


def test_a_click_on_the_volume_view_places_nothing(interaction):
    """A region is traced on a cut, and the volume rendering is not one."""
    click(measuring(interaction), "volume", 120, 90)

    assert placed(interaction) == []


def test_a_drag_is_not_a_click(interaction):
    """Which is what leaves window/level alone while the mode is on."""
    click(measuring(interaction), "ul", 120, 90, travel=40)

    assert placed(interaction) == []


def test_a_hand_that_is_not_quite_still_still_clicks(interaction):
    click(measuring(interaction), "ul", 120, 90, travel=2)

    assert len(placed(interaction)) == 1


def test_window_level_still_works_while_measuring(interaction):
    """The mode takes the click; the drags are untouched."""
    measuring(interaction)
    interaction.on_event(
        {"type": "LeftButtonPress", "position": {"x": 100, "y": 100}}, view_name="ul"
    )
    move(interaction, "ul", 130, 100)

    assert interaction.logic.mpr.window_level, "the drag went through"


def test_a_right_click_closes_the_region(interaction):
    click(measuring(interaction), "ul", 120, 90, button="Right")

    assert ("close_measurement", {}) in interaction.logic.calls


def test_a_right_click_outside_the_mode_closes_nothing(interaction):
    """Right alone is otherwise an unused gesture, and stays one."""
    click(interaction, "ul", 120, 90, button="Right")

    assert interaction.logic.calls == []


def test_a_press_in_one_view_and_a_release_in_another_is_not_a_click(interaction):
    """The presses are kept per view, so a stray release cannot fabricate one."""
    measuring(interaction)
    interaction.on_event(
        {"type": "LeftButtonPress", "position": {"x": 10, "y": 10}}, view_name="ul"
    )
    interaction.on_event(
        {"type": "LeftButtonRelease", "position": {"x": 10, "y": 10}}, view_name="ll"
    )

    assert placed(interaction) == []


@pytest.mark.parametrize(
    "key, action",
    [
        ("m", "toggle_measuring"),
        ("u", "undo_measurement_point"),
        ("x", "cancel_measurement"),
        ("e", "toggle_measurement_edit"),
    ],
)
def test_the_measuring_keys_reach_their_actions(interaction, key, action):
    interaction.on_event({"type": "KeyPress", "key": key})

    assert (action, {}) in interaction.logic.calls


def test_the_finish_key_closes_the_region_being_traced(interaction):
    press(interaction, interaction_module.FINISH_KEY)

    assert ("close_measurement", {}) in interaction.logic.calls


def test_the_finish_key_ends_a_correction_instead_while_one_is_open(interaction):
    """The two are never open at once, so one key finishes whichever is."""
    interaction.logic.server.state.measurement_editing = 1

    press(interaction, interaction_module.FINISH_KEY)

    assert interaction.logic.names == ["finish_measurement_edit"]


def test_the_finish_key_never_reverts_a_correction(interaction):
    """Giving up on a correction stays on a button: nothing puts it back."""
    interaction.logic.server.state.measurement_editing = 1

    press(interaction, interaction_module.FINISH_KEY)

    assert "revert_measurement_edit" not in interaction.logic.names


def test_the_measuring_keys_are_ones_a_keypress_fires_for(interaction):
    """The views forward the DOM ``keypress`` event, which does not fire for
    Escape or Backspace. Every printable key fires it; Enter is the one key
    that fires it without being printable, and is spelled as the DOM spells
    it."""
    for key in (
        interaction_module.MEASURE_KEY,
        interaction_module.UNDO_POINT_KEY,
        interaction_module.CANCEL_TRACE_KEY,
        interaction_module.EDIT_KEY,
    ):
        assert len(key) == 1 and key.isprintable()

    assert interaction_module.FINISH_KEY == "Enter"


# --- correcting a region ------------------------------------------------------

# The one gesture taken away from an existing one: a left drag that began on a
# point of the region being corrected is that point's, and every other left drag
# is the window and level it always was. Which of the two a press is cannot be
# decided here -- it is how near the point was -- so it is asked of the action
# and answered by it.


def editing(interaction, grabs=True) -> Interaction:
    interaction.logic.server.state.measurement_editing = 0
    interaction.logic.grabs = grabs
    return interaction


def drag(interaction, view, x, y, dx=30, dy=20):
    interaction.on_event(
        {"type": "LeftButtonPress", "position": {"x": x, "y": y}}, view_name=view
    )
    interaction.on_event(
        {"type": "MouseMove", "position": {"x": x + dx, "y": y + dy}}, view_name=view
    )
    interaction.on_event(
        {"type": "LeftButtonRelease", "position": {"x": x + dx, "y": y + dy}},
        view_name=view,
    )


def test_a_drag_from_a_point_moves_that_point(interaction):
    drag(editing(interaction), "ul", 120, 90)

    assert interaction.logic.names == [
        "grab_measurement_point",
        "drag_measurement_point",
        "drop_measurement_point",
    ]


def test_a_drag_from_a_point_does_not_window_and_level(interaction):
    drag(editing(interaction), "ul", 120, 90)

    assert interaction.logic.mpr.window_level == []


def test_a_drag_that_took_hold_of_nothing_windows_and_levels(interaction):
    drag(editing(interaction, grabs=False), "ul", 120, 90)

    assert interaction.logic.mpr.window_level != []
    assert "drag_measurement_point" not in interaction.logic.names


def test_a_drag_outside_a_correction_never_asks(interaction):
    drag(interaction, "ul", 120, 90)

    assert "grab_measurement_point" not in interaction.logic.names
    assert interaction.logic.mpr.window_level != []


def test_a_press_where_no_region_could_be_drawn_never_asks(interaction):
    """A question put on every press of every view is a question in the log."""
    drag(editing(interaction), "volume", 120, 90)

    assert "grab_measurement_point" not in interaction.logic.names


def test_the_point_travels_with_the_cursor(interaction):
    drag(editing(interaction), "ul", 120, 90, dx=40, dy=0)

    assert interaction.logic.arguments("drag_measurement_point") == [
        {"view_name": "ul", "x": 160, "y": 90}
    ]


def test_a_click_on_the_contour_adds_a_point(interaction):
    click(editing(interaction, grabs=False), "ul", 120, 90)

    assert interaction.logic.arguments("insert_measurement_point") == [
        {"view_name": "ul", "x": 120, "y": 90}
    ]


def test_a_click_while_correcting_does_not_also_trace(interaction):
    """Tracing is suspended for the duration, and this is where that holds."""
    editing(interaction, grabs=False)
    measuring(interaction)

    click(interaction, "ul", 120, 90)

    assert placed(interaction) == []


def test_letting_go_of_a_point_does_not_add_another(interaction):
    click(editing(interaction), "ul", 120, 90)

    assert "insert_measurement_point" not in interaction.logic.names
    assert "drop_measurement_point" in interaction.logic.names


def test_a_right_click_takes_a_point_away(interaction):
    click(editing(interaction), "ul", 120, 90, button="Right")

    assert interaction.logic.arguments("delete_measurement_point") == [
        {"view_name": "ul", "x": 120, "y": 90}
    ]


def test_a_right_click_while_correcting_closes_no_region(interaction):
    editing(interaction)
    measuring(interaction)

    click(interaction, "ul", 120, 90, button="Right")

    assert "close_measurement" not in interaction.logic.names


def test_a_right_drag_while_correcting_takes_no_point_away(interaction):
    click(editing(interaction), "ul", 120, 90, button="Right", travel=40)

    assert "delete_measurement_point" not in interaction.logic.names


# Double clicking a view


@pytest.fixture
def clock(monkeypatch):
    """A clock the test moves, since a double click is two clicks and a gap."""
    now = [1000.0]
    monkeypatch.setattr(
        interaction_module, "time", types.SimpleNamespace(time=lambda: now[0])
    )
    return now


def wait(clock, milliseconds):
    clock[0] += milliseconds / 1000.0


@pytest.mark.parametrize("view, layout", sorted(VIEW_LAYOUTS.items()))
def test_a_double_click_maximizes_the_view_it_landed_in(
    interaction, clock, view, layout
):
    """Whether a second double click restores is the action's business, as it
    is for the key that does the same thing."""
    click(interaction, view, 50, 50)
    click(interaction, view, 50, 50)

    assert interaction.logic.arguments("toggle_maximized") == [{"view": layout}]


def test_the_two_volume_panes_maximize_the_same_layout():
    """The quad view's pane and the maximized view are two widgets on one
    render window, and a double click on either means the same thing."""
    assert VIEW_LAYOUTS["volume_mpr"] == VIEW_LAYOUTS["volume"]


def test_one_click_maximizes_nothing(interaction, clock):
    click(interaction, "ul", 50, 50)

    assert interaction.logic.calls == []


def test_a_slow_second_click_is_a_second_click(interaction, clock):
    click(interaction, "ul", 50, 50)
    wait(clock, DOUBLE_CLICK_MS + 1)
    click(interaction, "ul", 50, 50)

    assert interaction.logic.calls == []


def test_a_second_click_somewhere_else_is_a_second_click(interaction, clock):
    click(interaction, "ul", 50, 50)
    click(interaction, "ul", 50 + 4 * CLICK_SLOP, 50)

    assert interaction.logic.calls == []


def test_a_click_in_each_of_two_views_is_not_a_double_click(interaction, clock):
    """Otherwise a hand crossing the quad view maximizes what it passes over."""
    click(interaction, "ul", 50, 50)
    click(interaction, "ll", 50, 50)

    assert interaction.logic.calls == []


def test_a_third_click_starts_a_new_gesture(interaction, clock):
    """Three clicks are one double click and one click, not two double clicks."""
    for _ in range(3):
        click(interaction, "ul", 50, 50)

    assert interaction.logic.arguments("toggle_maximized") == [{"view": "ul"}]


def test_four_clicks_are_two_double_clicks(interaction, clock):
    for _ in range(4):
        click(interaction, "ul", 50, 50)

    assert interaction.logic.arguments("toggle_maximized") == [{"view": "ul"}] * 2


def test_a_drag_is_not_a_double_click(interaction, clock):
    """Both the window/level drag and the double click are the left button, so
    the one that travels must not be read as the one that does not."""
    for _ in range(2):
        click(interaction, "ul", 50, 50, travel=40)

    assert "toggle_maximized" not in interaction.logic.names


def test_a_double_click_over_a_tile_blows_up_that_tile(interaction, clock):
    """The grid is one view holding many, so the tile is named by where the
    click landed rather than by the view it landed in."""
    click(interaction, "tile", 120, 80)
    click(interaction, "tile", 120, 80)

    assert interaction.logic.arguments("toggle_tile_focus") == [{"x": 120, "y": 80}]
    assert interaction.logic.tiles.focused == [(120, 80)]


@pytest.mark.parametrize("mode", ["measuring", "measurement_editing"])
def test_a_double_click_on_a_cut_is_ignored_while_a_region_is_open(
    interaction, clock, mode
):
    """The clicks belong to the region: tracing takes the click, and a layout
    that moved under a hand placing points would take the region with it."""
    setattr(interaction.logic.server.state, mode, True)

    click(interaction, "ul", 50, 50)
    click(interaction, "ul", 50, 50)

    assert "toggle_maximized" not in interaction.logic.names


def test_a_double_click_off_the_cuts_still_maximizes_while_tracing(interaction, clock):
    """A click on the volume rendering was never going to be a point."""
    interaction.logic.server.state.measuring = True

    click(interaction, "volume_mpr", 50, 50)
    click(interaction, "volume_mpr", 50, 50)

    assert interaction.logic.arguments("toggle_maximized") == [{"view": "volume"}]


def test_the_volumetry_charts_hear_nothing(interaction, clock):
    """They are given no interactor events at all, so `y` is their only way in
    and out; this says so where a reader of the table would look."""
    click(interaction, "volumetry", 50, 50)
    click(interaction, "volumetry", 50, 50)

    assert interaction.logic.calls == []


def test_the_help_sheet_writes_the_double_click_down():
    """The mouse table is written out by hand, like the key table above it."""
    source = (pl.Path(interaction_module.__file__).parent / "help.py").read_text()
    assert "Double Click" in source
