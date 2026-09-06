"""Shared vue expressions and layout helpers for the drawer panels."""

# System
import contextlib as cl
import hashlib
import pathlib as pl

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ..view import Layout

# How wide the drawer is. Read by the layout that sets it and by the console,
# which has to start where the drawer stops.
DRAWER_WIDTH = 340

# The drawer's MPR controls only make sense in the quad view with a volume
# selected. Written out nineteen times before this constant existed.
MPR_ACTIVE = "!maximized_view && active_volume_label"

TILE_ACTIVE = "maximized_view === 'tile'"
# Spelled out rather than negating TILE_ACTIVE: "!" binds tighter than "===",
# so f"!{TILE_ACTIVE}" reads as (!maximized_view) === 'tile' and is never true.
NOT_TILE_ACTIVE = "maximized_view !== 'tile'"

# The layouts with a cut on screen, which is every one that is not the volume
# rendering. Listed from ``Layout`` rather than written out, because the two
# had drifted: this used to name the quad view and the tile grid, and so took
# the overlay controls away from a maximized axial view that was drawing the
# overlays it controls.
_RESLICE_LAYOUTS = ", ".join(
    f"'{layout.state_value}'" for layout in Layout if layout.shows_reslice
)
RESLICE_ACTIVE = f"[{_RESLICE_LAYOUTS}].includes(maximized_view) && active_volume_label"


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

SHEET_CLASS = "cardio-sheet"

SHEET_BODY_CLASS = "cardio-sheet-body"


@cl.contextmanager
def sheet_dialog(visible_key: str, title: str):
    """A full-page reference sheet, opened by a key and closed by a button.

    The help reference and the metadata sheet are the same object with
    different contents, so the chrome is spelled once here: whichever of them
    grows a scrollbar or changes its card, both do.
    """
    with vuetify.VDialog(
        v_model=(visible_key,),
        max_width="700px",
        scrim="rgba(0, 0, 0, 0.7)",
    ):
        with vuetify.VCard(classes=f"pa-6 {SHEET_CLASS}"):
            vuetify.VCardTitle(title, classes="text-h5 mb-4")

            with vuetify.VCardText(classes=SHEET_BODY_CLASS):
                yield

            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn(
                    "Close",
                    click=f"{visible_key} = false",
                    variant="text",
                )


STATIC = pl.Path(__file__).parent / "static"


STYLESHEET = "drawer.css"


def drawer_styles(server):
    """Serve the drawer stylesheet.

    A ``<style>`` tag written into the layout does not survive vue's template
    compiler, so the rules have to arrive as a served asset instead -- and a
    served asset is one the browser is entitled to keep. Naming it by a digest
    of its own contents is what makes an edit to it an edit the next reload
    actually sees, rather than one that shows up whenever the cache decides.
    """
    digest = hashlib.sha256((STATIC / STYLESHEET).read_bytes()).hexdigest()[:12]
    server.enable_module(
        {
            "serve": {"__cardio": str(STATIC)},
            "styles": [f"__cardio/{STYLESHEET}?v={digest}"],
        }
    )
