"""The trame layout, composed from one module per region of the page."""

# Third Party
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as vuetify

# Internal
from .. import __version__
from ..scene import Scene
from ..view import RENDER_VIEWS
from .common import DRAWER_WIDTH, drawer_styles, section
from .help import help_dialog
from .interaction import Interaction
from .layout import toolbar, viewports
from .metadata import metadata_dialog
from .panels import (
    capture_panel,
    console_panel,
    playback_panel,
    rotations_panel,
    slice_views_panel,
    snap_panel,
    tiles_panel,
    volume_panel,
    volume_rendering_panel,
    zoom_panel,
)

__all__ = ["UI", "Interaction"]


class UI:
    """Builds the page and routes view events to the logic controllers."""

    def __init__(self, server, scene: Scene, logic):
        self.server = server
        self.scene = scene
        self.interaction = Interaction(logic)

        self.setup()

    @property
    def handled_events(self):
        return self.interaction.handled_events

    def event_listeners_for_view(self, view_name):
        return self.interaction.listeners_for_view(view_name)

    def setup(self):
        drawer_styles(self.server)

        with SinglePageWithDrawerLayout(self.server, theme=("theme_mode",)) as layout:
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
                console_panel(self.server, self.scene)
                help_dialog()
                metadata_dialog(self.scene)

            with layout.drawer as drawer:
                drawer.width = DRAWER_WIDTH
                self.drawer()

    def drawer(self):
        """The active volume, then one collapsible section per concern."""
        volume_panel(self.server, self.scene)

        with vuetify.VExpansionPanels(
            v_model=("drawer_sections",),
            multiple=True,
            variant="accordion",
            flat=True,
        ):
            with section("playback", "Playback", "mdi-play-circle-outline"):
                playback_panel(self.server, self.scene)

            with section("appearance", "Appearance", "mdi-palette-outline"):
                # One group per view the controls act on. Each says which view
                # that is, and goes away with it.
                volume_rendering_panel(self.server, self.scene)
                slice_views_panel(self.server, self.scene)

            if self.scene.volumes:
                # Up in every layout: the tile grid is posed by these controls,
                # and so is a volume camera locked to one of the slices, which
                # follows the pose while the slices are off screen.
                with section("orientation", "Orientation", "mdi-axis-arrow"):
                    snap_panel(self.server, self.scene)
                    rotations_panel(self.server, self.scene)

                if self.scene.segmentations:
                    with section("tiles", "Tile View", "mdi-view-grid-outline"):
                        tiles_panel(self.server, self.scene)

                    with section("zoom", "Zoom", "mdi-fit-to-screen-outline"):
                        zoom_panel(self.server, self.scene)

            with section("export", "Export", "mdi-video-outline"):
                capture_panel(self.server, self.scene)

    def _update_all_mpr_views(self, **kwargs):
        """Push a new frame to every view.

        The ones this layout did not build have no implementation and do
        nothing, which Logic is what says is allowed.
        """
        controller = self.server.controller
        for name in RENDER_VIEWS:
            getattr(controller, name)()
