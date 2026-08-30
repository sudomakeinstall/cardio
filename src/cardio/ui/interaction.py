"""Turning mouse and keyboard events into controller calls.

This module decides *what the user asked for*, never what it means
geometrically: the window/level and slice-scroll arithmetic lives on the MPR
controller, which owns that state.
"""

# System
import functools as ft
import math
import time

# Internal
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

MPR_VIEWS = {"axial", "sagittal", "coronal"}

# Views a drag means something in. The tile grid takes window/level but not the
# slice scroll, which has no single slice to move.
DRAG_VIEWS = MPR_VIEWS | {"tile"}

# Keys that maximize a view, and the view each one names
MAXIMIZE_KEYS = {
    "v": "volume",
    "a": "axial",
    "c": "coronal",
    "s": "sagittal",
    "t": "tile",
}


class Interaction:
    """Drag and keypress handling for the render views."""

    def __init__(self, server, logic):
        self.server = server
        self.logic = logic

        self.left_dragging = False
        self.right_dragging = False
        self.middle_dragging = False
        self.last_mouse_pos = {}

        self.window_sensitivity = 5.0
        self.level_sensitivity = 2.0
        self.slice_sensitivity = 1.0
        self.wheel_sensitivity = 1.0
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
                self._on_key(event["key"])

            case "LeftButtonPress":
                self.left_dragging = True
                self._store_mouse_position(view_name, event)

            case "LeftButtonRelease":
                self.left_dragging = False

            case "RightButtonPress":
                self.right_dragging = True
                self._store_mouse_position(view_name, event)

            case "RightButtonRelease":
                self.right_dragging = False

            case "MiddleButtonPress":
                self.middle_dragging = True
                self._store_mouse_position(view_name, event)

            case "MiddleButtonRelease":
                self.middle_dragging = False

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
                    self.logic.mpr.scroll_slice(
                        view_name, spin * self.wheel_sensitivity
                    )

    def _apply_drag(self, view_name, previous, position):
        """One gesture per button combination.

        Window/level is the only one the tile grid takes; the rest need a single
        slice to act on, and zoom is the only one that is not about one view.
        Rotation is given both positions rather than the delta, being an angle
        swept about a point rather than a distance travelled.
        """
        dx = position[0] - previous[0]
        dy = position[1] - previous[1]

        if self.left_dragging and not (self.right_dragging or self.middle_dragging):
            self.logic.mpr.adjust_window_level(
                -dx * self.window_sensitivity,
                -dy * self.level_sensitivity,
            )
            return

        if view_name not in MPR_VIEWS:
            return

        if self.middle_dragging and self.left_dragging:
            self.logic.mpr.rotate_view(view_name, previous, position)
        elif self.middle_dragging and self.right_dragging:
            self.logic.mpr.zoom_views(math.exp(dy * self.zoom_sensitivity))
        elif self.middle_dragging:
            self.logic.mpr.pan_view(view_name, dx, dy)
        elif self.left_dragging and self.right_dragging:
            self.logic.mpr.scroll_slice(view_name, dy * self.slice_sensitivity)

    def _on_key(self, key):
        """Apply a keyboard shortcut, ignoring repeats inside the debounce."""
        now = time.time() * 1000
        if now - self.last_keypress_time.get(key, 0) < self.keypress_debounce_ms:
            return
        self.last_keypress_time[key] = now

        state = self.server.state

        if key.isdigit() and int(key) in presets:
            state.mpr_window_level_preset = int(key)
        elif key == "l":
            state.mpr_crosshairs_enabled = not state.mpr_crosshairs_enabled
        elif key == "h":
            state.help_overlay_visible = not state.help_overlay_visible
        elif key in MAXIMIZE_KEYS:
            view = MAXIMIZE_KEYS[key]
            state.maximized_view = "" if state.maximized_view == view else view

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
