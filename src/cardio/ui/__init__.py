"""The trame layout, composed from one module per region of the page."""

# Third Party
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as vuetify

# Internal
from .. import __version__
from ..scene import Scene
from ..state import DEFAULT_THEME_MODE
from .common import DEFAULT_OPEN_SECTIONS, TILE_ACTIVE, drawer_styles, section
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
    tiles_panel,
    volume_panel,
)

DRAWER_WIDTH = 340

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
        drawer_styles(self.server)

        with SinglePageWithDrawerLayout(
            self.server, theme=("theme_mode", DEFAULT_THEME_MODE)
        ) as layout:
            self.layout = layout
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

            with layout.drawer as drawer:
                drawer.width = DRAWER_WIDTH
                self.drawer()

    def drawer(self):
        """The active volume, then one collapsible section per concern."""
        volume_panel(self.server, self.scene)

        with vuetify.VExpansionPanels(
            v_model=("drawer_sections", DEFAULT_OPEN_SECTIONS),
            multiple=True,
            variant="accordion",
            flat=True,
        ):
            with section("playback", "Playback", "mdi-play-circle-outline"):
                playback_panel(self.server, self.scene)

            with section("appearance", "Appearance", "mdi-palette-outline"):
                clip_depth_panel(self.server, self.scene)
                appearance_panel(self.server, self.scene)
                overlays_panel(self.server, self.scene)

            if self.scene.volumes:
                # The tile grid is posed by the same controls, so they stay
                # reachable while it is on screen.
                with section(
                    "orientation",
                    "Orientation",
                    "mdi-axis-arrow",
                    v_if=f"!maximized_view || {TILE_ACTIVE}",
                ):
                    snap_panel(self.server, self.scene)
                    rotations_panel(self.server, self.scene)

                if self.scene.segmentations:
                    with section("tiles", "Tile View", "mdi-view-grid-outline"):
                        tiles_panel(self.server, self.scene)

            with section("export", "Export", "mdi-video-outline"):
                capture_panel(self.server, self.scene)

    def _update_all_mpr_views(self, **kwargs):
        """Push a new frame to whichever views the layout created."""
        controller = self.server.controller
        for name in (
            "axial_update",
            "coronal_update",
            "sagittal_update",
            "volume_update",
            "tile_update",
        ):
            if hasattr(controller, name):
                getattr(controller, name)()
