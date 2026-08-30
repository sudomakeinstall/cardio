"""Turning mouse and keyboard events into controller calls.

This module decides *what the user asked for*, never what it means
geometrically: the window/level and slice-scroll arithmetic lives on the MPR
controller, which owns that state.
"""

# System
import functools as ft
import time

# Internal
from ..window_level import presets

HANDLED_EVENTS = [
    "MouseMove",
    "LeftButtonPress",
    "LeftButtonRelease",
    "RightButtonPress",
    "RightButtonRelease",
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
        self.last_mouse_pos = {}

        self.window_sensitivity = 5.0
        self.level_sensitivity = 2.0
        self.slice_sensitivity = 1.0

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

            case "MouseMove" if self.left_dragging:
                delta = self._drag_delta(view_name, event)
                if delta is not None:
                    dx, dy = delta
                    self.logic.mpr.adjust_window_level(
                        -dx * self.window_sensitivity,
                        -dy * self.level_sensitivity,
                    )

            case "MouseMove" if self.right_dragging and view_name in MPR_VIEWS:
                delta = self._drag_delta(view_name, event)
                if delta is not None:
                    _, dy = delta
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

    def _drag_delta(self, view_name, event):
        """Movement since the last event in a draggable view, or None."""
        if view_name not in DRAG_VIEWS:
            return None
        if view_name not in self.last_mouse_pos or "position" not in event:
            return None

        position = [event["position"]["x"], event["position"]["y"]]
        previous = self.last_mouse_pos[view_name]
        self.last_mouse_pos[view_name] = position

        return position[0] - previous[0], position[1] - previous[1]
