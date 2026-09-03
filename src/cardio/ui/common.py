"""Shared vue expressions and layout helpers for the drawer panels."""

# System
import contextlib as cl
import pathlib as pl

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# The drawer's MPR controls only make sense in the quad view with a volume
# selected. Written out nineteen times before this constant existed.
MPR_ACTIVE = "!maximized_view && active_volume_label"

# The layouts that resample the volume: the quad MPR grid, and the tile grid.
# Controls over the slice pose and its overlays belong to both.
TILE_ACTIVE = "maximized_view === 'tile'"
# Spelled out rather than negating TILE_ACTIVE: "!" binds tighter than "===",
# so f"!{TILE_ACTIVE}" reads as (!maximized_view) === 'tile' and is never true.
NOT_TILE_ACTIVE = "maximized_view !== 'tile'"
RESLICE_ACTIVE = f"(!maximized_view || {TILE_ACTIVE}) && active_volume_label"


@cl.contextmanager
def section(value, title, icon, **kwargs):
    """One collapsible group of the drawer accordion.

    ``value`` is the key the accordion's v-model tracks open sections by.
    """
    with vuetify.VExpansionPanel(value=value, **kwargs):
        with vuetify.VExpansionPanelTitle(classes="text-subtitle-2 px-4"):
            vuetify.VIcon(icon, size="small", classes="mr-3")
            html.Span(title)
        with vuetify.VExpansionPanelText():
            yield


SLIDER_CLASS = "cardio-slider"

SUBPANEL_CLASS = "cardio-subpanel"

STATIC = pl.Path(__file__).parent / "static"


def drawer_styles(server):
    """Serve the drawer stylesheet.

    A ``<style>`` tag written into the layout does not survive vue's template
    compiler, so the rules have to arrive as a served asset instead.
    """
    server.enable_module(
        {
            "serve": {"__cardio": str(STATIC)},
            "styles": ["__cardio/drawer.css"],
        }
    )
