"""The page itself: which layout is up, and which sheets are open."""

# Internal
from .. import __version__
from ..action import action
from ..metadata import describe_scene
from ..view import QUAD_LAYOUT
from .base import Controller


class ViewController(Controller):
    """Layout, the two reference sheets and the drawer's open sections.

    None of this drives the renderer, so there is nothing here to listen for:
    the layout is read by whoever needs to know what is on screen, and the
    sheets are opened and closed in the browser. What the controller is for is
    that these are configured values like any other, and that the keys which
    toggle them are actions like any other.
    """

    def seed(self):
        state = self.server.state
        view = self.scene.view

        state.trame__title = f"cardio v{__version__}"
        state.maximized_view = view.layout.state_value
        state.help_overlay_visible = view.help_visible
        state.metadata_overlay_visible = view.metadata_visible
        state.drawer_sections = view.open_sections

        entries = describe_scene(self.scene)
        state.metadata_pages = [
            {"title": entry.title, "value": entry.key} for entry in entries
        ]
        state.metadata_object = entries[0].key if entries else ""

    @action("toggle_maximized")
    def toggle_maximized(self, view: str):
        """Show ``view`` alone, or go back to the quad view if it already is.

        The toggle lives here rather than in the key handler because pressing
        `a` twice is one action asked for twice, and only the state says what
        the second press means.
        """
        state = self.server.state
        state.maximized_view = QUAD_LAYOUT if state.maximized_view == view else view

    @action("toggle_help")
    def toggle_help(self):
        """Open or close the keyboard reference."""
        state = self.server.state
        state.help_overlay_visible = not state.help_overlay_visible

    @action("toggle_metadata")
    def toggle_metadata(self):
        """Open or close the sheet saying what each object in the scene is."""
        state = self.server.state
        state.metadata_overlay_visible = not state.metadata_overlay_visible
