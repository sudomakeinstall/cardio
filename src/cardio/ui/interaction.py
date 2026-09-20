"""Turning mouse and keyboard events into named actions.

This module decides *what the user asked for*, never what it means
geometrically and no longer which controller answers for it: every gesture and
every key comes out as one ``dispatch`` of an action by name.
"""

# System
import functools as ft
import math
import time

# Internal
from ..planimetry import CLICK_SLOP, is_click
from ..window_level import presets

HANDLED_EVENTS = [
    "MouseMove",
    "MouseWheel",
    "LeftButtonPress",
    "LeftButtonRelease",
    "RightButtonPress",
    "RightButtonRelease",
    "MiddleButtonPress",
    "MiddleButtonRelease",
    "KeyPress",
]

MPR_VIEWS = {"ul", "lr", "ll"}

# The one view whose drags VTK handles itself, turning the camera with its own
# trackball. Nothing tells us it moved, so the release is when we go and look.
TRACKBALL_VIEW = "volume"

# Views a drag means something in. The tile grid takes window/level but not the
# slice scroll, which has no single slice to move.
DRAG_VIEWS = MPR_VIEWS | {"tile"}

# Views a click means something in, which is every view showing a cut: a click
# places a point of a region, and a region is traced on a cut.
CUT_VIEWS = MPR_VIEWS | {"tile"}

# The console's key. A backtick because every letter that reads as "console"
# already names a view, and because it is where a console usually is.
CONSOLE_KEY = "`"

# The measuring keys. All three are printable letters, and have to be: the
# views forward the DOM ``keypress`` event, which does not fire for Escape or
# Backspace. The buttons in the drawer are the path that does not have to be
# remembered; these are for a hand already on the mouse.
MEASURE_KEY = "m"
UNDO_POINT_KEY = "u"
CANCEL_TRACE_KEY = "x"

# Finishing, alongside the right click that has always closed a region: a hand
# on the keyboard finishes what it is doing the way it finishes anything else.
# Which thing it finishes is whichever is open -- a trace, or a correction --
# and the two are never open at once. Its opposite stays on a button: a trace
# given up is a trace nobody had finished, and a correction given up is work
# thrown away that nothing in the app can put back.
#
# The one measuring key that is not printable and does not have to be, since
# ``keypress`` does fire for Enter -- it carries a character where Escape and
# Backspace carry none -- and spelled as the DOM event spells it.
FINISH_KEY = "Enter"

# Correcting a region already closed. Deliberately not given up to ``x``, which
# means give up on the trace in progress: a correction discarded to a stray
# keypress is worse than one that has to be given up through a button.
EDIT_KEY = "e"

# Stepping through the study with vim's hjkl: the frame under the fingers that
# move left and right, the slice under the ones that move up and down. Both
# letters were spoken for -- ``h`` opened the help and ``l`` toggled the
# crosshairs -- and those moved to the keys below rather than the other way
# round, being reached for once a session against once a second.
FRAME_KEYS = {"h": "decrement_frame", "l": "increment_frame"}

# Which way one press travels, in the units the wheel turns in: a notch each,
# signed as the wheel is, so ``k`` goes the way an upward drag does.
SLICE_KEYS = {"k": 1.0, "j": -1.0}

# Where hjkl sent the two it displaced. ``?`` is the help key most things
# already use, and ``+`` is a picture of what it draws; both are punctuation,
# so neither can collide with a letter a view or a mode wants later.
HELP_KEY = "?"
CROSSHAIR_KEY = "+"

# Keys that maximize a view, and the view each one names. The cut views kept
# their anatomical letters when they were renamed for where they sit, since
# the keys are what fingers know and no pane letter is free anyway.
MAXIMIZE_KEYS = {
    "v": "volume",
    "a": "ul",
    "c": "ll",
    "s": "lr",
    "t": "tile",
    "y": "volumetry",
}


class Interaction:
    """Drag and keypress handling for the render views."""

    def __init__(self, logic):
        self.logic = logic

        self.left_dragging = False
        self.right_dragging = False
        self.middle_dragging = False
        self.last_mouse_pos = {}
        # Where each button went down, per view, so a release can say whether
        # it was a click. Keyed by view so a press in one pane and a release in
        # another cannot be read as a click in either.
        self.press_pos = {}
        # Whether the left press in flight took hold of a point of a region. A
        # drag that did belongs to the region; every other drag is what it
        # always was.
        self.grabbed = False

        self.window_sensitivity = 5.0
        self.level_sensitivity = 2.0
        self.slice_sensitivity = 1.0
        self.wheel_sensitivity = 1.0
        self.key_slice_sensitivity = 1.0
        self.zoom_sensitivity = 0.005

        self.last_keypress_time = {}
        self.keypress_debounce_ms = 100

    @property
    def handled_events(self):
        return HANDLED_EVENTS

    def listeners_for_view(self, view_name):
        """Interactor event bindings for one named view."""
        callback = ft.partial(self.on_event, view_name=view_name)
        return {
            event: (callback, "[utils.vtk.event($event)]") for event in HANDLED_EVENTS
        }

    def on_event(self, *args, view_name=None, **kwargs):
        if not args:
            return

        event = args[0]

        match event["type"]:
            case "KeyPress":
                self._on_key(event["key"], view_name)

            case "LeftButtonPress":
                self.left_dragging = True
                self._store_mouse_position(view_name, event)
                self._note_press("left", view_name, event)
                # Only where a region could be drawn, so that a press on the
                # volume view is not a question worth putting in the log.
                self.grabbed = bool(
                    self._editing
                    and view_name in CUT_VIEWS
                    and self._at("grab_measurement_point", view_name, event)
                )

            case "LeftButtonRelease":
                self.left_dragging = False
                self._note_camera(view_name)
                clicked = self._was_click("left", view_name, event)
                if self.grabbed:
                    self.grabbed = False
                    self.logic.dispatch("drop_measurement_point")
                elif clicked and self._editing:
                    self._at("insert_measurement_point", view_name, event)
                elif clicked and self._measuring:
                    self._at("place_measurement_point", view_name, event)

            case "RightButtonPress":
                self.right_dragging = True
                self._store_mouse_position(view_name, event)
                self._note_press("right", view_name, event)

            case "RightButtonRelease":
                self.right_dragging = False
                self._note_camera(view_name)
                clicked = self._was_click("right", view_name, event)
                if clicked and self._editing:
                    self._at("delete_measurement_point", view_name, event)
                elif clicked and self._measuring:
                    self.logic.dispatch("close_measurement")

            case "MiddleButtonPress":
                self.middle_dragging = True
                self._store_mouse_position(view_name, event)

            case "MiddleButtonRelease":
                self.middle_dragging = False
                self._note_camera(view_name)

            case "MouseMove" if self.left_dragging and self.grabbed:
                # The one drag that is not a camera or a window/level gesture,
                # and the only one taken away from them -- a press that did not
                # land on a point never gets here.
                self._at("drag_measurement_point", view_name, event)

            case "MouseMove" if (
                self.left_dragging or self.right_dragging or self.middle_dragging
            ):
                motion = self._drag_motion(view_name, event)
                if motion is not None:
                    self._apply_drag(view_name, *motion)

            case "MouseWheel" if view_name in MPR_VIEWS:
                # Signed to travel the same way as an upward both-buttons drag;
                # a system set to natural scrolling inverts spinY before us
                spin = event.get("spinY")
                if spin:
                    self.logic.dispatch(
                        "scroll_slice",
                        view_name=view_name,
                        distance=spin * self.wheel_sensitivity,
                    )

    def _note_camera(self, view_name):
        """Write down where a trackball drag left the camera."""
        if view_name == TRACKBALL_VIEW:
            self.logic.dispatch("place_camera")

    def _apply_drag(self, view_name, previous, position):
        """One gesture per button combination.

        Window/level and zoom are about a grid of views rather than any one of
        them, so the tile grid takes both; the rest need a single slice to act
        on. Rotation is given both positions rather than the delta, being an
        angle swept about a point rather than a distance travelled.
        """
        dx = position[0] - previous[0]
        dy = position[1] - previous[1]

        if self.left_dragging and not (self.right_dragging or self.middle_dragging):
            self.logic.dispatch(
                "adjust_window_level",
                window_delta=-dx * self.window_sensitivity,
                level_delta=-dy * self.level_sensitivity,
            )
            return

        if self.middle_dragging and self.right_dragging:
            self._zoom(view_name, math.exp(dy * self.zoom_sensitivity))
            return

        if view_name not in MPR_VIEWS:
            return

        if self.middle_dragging and self.left_dragging:
            self.logic.dispatch(
                "rotate_view", view_name=view_name, start=previous, end=position
            )
        elif self.middle_dragging:
            self.logic.dispatch("pan_view", view_name=view_name, dx=dx, dy=dy)
        elif self.left_dragging and self.right_dragging:
            self.logic.dispatch(
                "scroll_slice",
                view_name=view_name,
                distance=dy * self.slice_sensitivity,
            )

    def _zoom(self, view_name, factor):
        """Zoom whichever grid of views the drag is over, all of it together."""
        if view_name == "tile":
            self.logic.dispatch("zoom_tiles", factor=factor)
        elif view_name in MPR_VIEWS:
            self.logic.dispatch("zoom_views", factor=factor)

    def _on_key(self, key, view_name=None):
        """Apply a keyboard shortcut, ignoring repeats inside the debounce.

        ``view_name`` is the view the key was pressed over, which only the
        slice keys need: they move one cut, the way the wheel does, and the one
        they move is the one under the cursor.
        """
        now = time.time() * 1000
        if now - self.last_keypress_time.get(key, 0) < self.keypress_debounce_ms:
            return
        self.last_keypress_time[key] = now

        if key.isdigit() and int(key) in presets:
            self.logic.dispatch("set_window_level_preset", preset=int(key))
        elif key in FRAME_KEYS:
            self.logic.dispatch(FRAME_KEYS[key])
        elif key in SLICE_KEYS and view_name in MPR_VIEWS:
            self.logic.dispatch(
                "scroll_slice",
                view_name=view_name,
                distance=SLICE_KEYS[key] * self.key_slice_sensitivity,
            )
        elif key == CROSSHAIR_KEY:
            self.logic.dispatch("toggle_crosshairs")
        elif key == HELP_KEY:
            self.logic.dispatch("toggle_help")
        elif key == "i":
            self.logic.dispatch("toggle_metadata")
        elif key == CONSOLE_KEY:
            self.logic.dispatch("toggle_console")
        elif key == MEASURE_KEY:
            self.logic.dispatch("toggle_measuring")
        elif key == UNDO_POINT_KEY:
            self.logic.dispatch("undo_measurement_point")
        elif key == CANCEL_TRACE_KEY:
            self.logic.dispatch("cancel_measurement")
        elif key == FINISH_KEY:
            self.logic.dispatch(
                "finish_measurement_edit" if self._editing else "close_measurement"
            )
        elif key == EDIT_KEY:
            self.logic.dispatch("toggle_measurement_edit")
        elif key in MAXIMIZE_KEYS:
            self.logic.dispatch("toggle_maximized", view=MAXIMIZE_KEYS[key])

    @property
    def _measuring(self) -> bool:
        """Whether a click is placing a point rather than merely being a click.

        One of the two things this module reads rather than reports, and they
        earn the exception together: every action goes into the log the console
        shows and the script it exports, so dispatching one on every click of a
        gesture nobody meant for a region would fill both with lines nobody
        asked for. What either mode *means* is still the controller's alone.
        """
        return bool(getattr(self.logic.server.state, "measuring", False))

    @property
    def _editing(self) -> bool:
        """Whether the mouse is correcting a region rather than driving a view.

        Read here for the reason above, and independent of ``_measuring``: a
        region is corrected without entering the tracing mode, and the two are
        never open at once because opening either closes the other.
        """
        return getattr(self.logic.server.state, "measurement_editing", None) is not None

    def _at(self, name: str, view_name, event):
        """Ask for a named action where the cursor is, and say what it answered.

        Every action a press turns into takes the same three arguments, and the
        one that decides whether a drag has been claimed answers rather than
        merely acting -- which is the only thing this module ever learns from a
        dispatch, and it learns it from the action it asked rather than from
        state it went looking through.
        """
        if "position" not in event:
            return None

        return self.logic.dispatch(
            name,
            view_name=view_name,
            x=event["position"]["x"],
            y=event["position"]["y"],
        )

    def _note_press(self, button: str, view_name, event):
        """Remember where a button went down, so its release can be judged."""
        if view_name in CUT_VIEWS and "position" in event:
            self.press_pos[(button, view_name)] = [
                event["position"]["x"],
                event["position"]["y"],
            ]

    def _was_click(self, button: str, view_name, event) -> bool:
        """Whether a release ends a click rather than a drag.

        Every gesture that uses a button travels -- window/level, the zoom, the
        slice scroll -- so a press and release in the same place is the one
        thing none of them is, and can be given to the tracing without taking
        anything away from the rest. The slop is for a hand that is not quite
        still, not for a short drag.
        """
        press = self.press_pos.pop((button, view_name), None)
        if press is None or view_name not in CUT_VIEWS or "position" not in event:
            return False

        return is_click(
            press, [event["position"]["x"], event["position"]["y"]], CLICK_SLOP
        )

    def _store_mouse_position(self, view_name, event):
        """Remember where a drag started, so the next move has a delta."""
        if view_name and "position" in event:
            self.last_mouse_pos[view_name] = [
                event["position"]["x"],
                event["position"]["y"],
            ]

    def _drag_motion(self, view_name, event):
        """Where a draggable view was and is since the last event, or None."""
        if view_name not in DRAG_VIEWS:
            return None
        if view_name not in self.last_mouse_pos or "position" not in event:
            return None

        position = [event["position"]["x"], event["position"]["y"]]
        previous = self.last_mouse_pos[view_name]
        self.last_mouse_pos[view_name] = position

        return previous, position
