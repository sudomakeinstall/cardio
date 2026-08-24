"""The trame layout, composed from one module per region of the page."""

# Third Party
from trame.ui.vuetify3 import SinglePageWithDrawerLayout

# Internal
from .. import __version__
from ..scene import Scene
from .help import help_dialog
from .interaction import Interaction
from .layout import toolbar, viewports
from .panels import (
    appearance_panel,
    capture_panel,
    clip_depth_panel,
    overlays_panel,
    playback_panel,
    rotations_panel,
    snap_panel,
)

__all__ = ["UI", "Interaction"]


class UI:
    """Builds the page and routes view events to the logic controllers."""

    def __init__(self, server, scene: Scene, logic):
        self.server = server
        self.scene = scene
        self.interaction = Interaction(server, logic)

        self.server.state.help_overlay_visible = False
        self.server.state.maximized_view = ""
        self.server.state.rotations_saved_at = None
        self.server.state.rotations_stale = False

        self.setup()

    @property
    def handled_events(self):
        return self.interaction.handled_events

    def event_listeners_for_view(self, view_name):
        return self.interaction.listeners_for_view(view_name)

    def setup(self):
        self.server.state.trame__title = f"cardio v{__version__}"

        with SinglePageWithDrawerLayout(
            self.server, theme=("theme_mode", "dark")
        ) as layout:
            layout.icon.click = self.server.controller.view_reset_camera
            layout.title.set_text(f"cardio v{__version__}")

            toolbar(self.server, self.scene, layout)

            with layout.content:
                viewports(
                    self.server,
                    self.scene,
                    self.event_listeners_for_view,
                    self.handled_events,
                    self._update_all_mpr_views,
                )
                help_dialog()

            with layout.drawer:
                snap_panel(self.server, self.scene)
                rotations_panel(self.server, self.scene)
                overlays_panel(self.server, self.scene)
                playback_panel(self.server, self.scene)
                capture_panel(self.server, self.scene)
                clip_depth_panel(self.server, self.scene)
                appearance_panel(self.server, self.scene)

    def _update_all_mpr_views(self, **kwargs):
        """Push a new frame to whichever views the layout created."""
        controller = self.server.controller
        for name in (
            "axial_update",
            "coronal_update",
            "sagittal_update",
            "volume_update",
        ):
            if hasattr(controller, name):
                getattr(controller, name)()
