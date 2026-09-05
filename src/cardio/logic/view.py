"""The page itself: which layout is up, and which sheets are open."""

# Internal
from .. import __version__
from ..metadata import describe_scene
from .base import Controller


class ViewController(Controller):
    """Layout, the two reference sheets and the drawer's open sections.

    None of this drives the renderer, so there is nothing here to listen for:
    the layout is read by whoever needs to know what is on screen, and the
    sheets are opened and closed in the browser. What the controller is for is
    that these are configured values like any other, and so belong in the one
    pass that writes configured values.
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
