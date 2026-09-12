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

# A volume to orient. The pose of the cuts is not a property of whatever is on
# screen: a volume camera locked to a slice follows the pose while the slices
# themselves are off screen -- which logic/mpr.py's update_mpr_rotation runs a
# branch of its own for -- and the tile grid is posed by the same controls. So
# the layout does not come into it; having a volume at all does.
VOLUME_ACTIVE = "active_volume_label"

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

# The layouts with the volume rendering on screen. Everything under Appearance
# but the MPR overlays acts on that renderer and no other: what is drawn in it,
# in what colours, cropped to what box, between which two depths.
_RENDERING_LAYOUTS = ", ".join(
    f"'{layout.state_value}'" for layout in Layout if Layout.VOLUME in layout.on_screen
)
RENDERING_ACTIVE = f"[{_RENDERING_LAYOUTS}].includes(maximized_view)"


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


# What kind of thing a row is, in the only place a row has room to say it: two
# kinds may carry the same label, and the eye at the end of either is the same
# eye.
KIND_ICONS = {
    "mesh": "mdi-triangle-outline",
    "volume": "mdi-cube-outline",
    "segmentation": "mdi-shape-outline",
}

EYE_ICONS = ("mdi-eye", "mdi-eye-off")


@cl.contextmanager
def object_row(label: str, icon: str):
    """One object on one line: what it is on the left, its toggles on the right.

    The toggles are the caller's, each in a cell of its own, so that a row with
    nothing to crop and nothing to open still ends where every other row does.
    """
    with vuetify.VRow(no_gutters=True, classes="align-center flex-nowrap"):
        with vuetify.VCol(classes="text-body-2 text-truncate"):
            vuetify.VIcon(icon, size="x-small", classes="mr-2")
            html.Span(label)
        yield


@cl.contextmanager
def row_cell():
    """One trailing control of an object row, sized to itself."""
    with vuetify.VCol(cols="auto"):
        yield


GROUP_CLASS = "cardio-group"


@cl.contextmanager
def target_group(title: str, icon: str, subtitle: str, **kwargs):
    """A run of controls that all act on the same view, named by that view.

    Nothing about a control says which view it changes -- the eye beside a
    segmentation means the rendering in one of these groups and the overlays
    on the cuts in the other -- so the grouping is what says it, and the
    heading names the view rather than the controls under it.
    """
    with html.Div(classes=GROUP_CLASS, **kwargs):
        with html.Div(classes=f"{GROUP_CLASS}-title", title=subtitle):
            vuetify.VIcon(icon, size="x-small", classes="mr-2")
            html.Span(title)
        yield


# A text field swallows its own key events, or they reach the render view's
# interactor and typing `a` maximizes the axial view. Here rather than beside
# one panel because several want it; the console prompt and the rotation names
# guard `keyup` as well, for reasons of their own that are written down there.
SWALLOW_KEYS = {
    "__events": ["keydown", "keypress"],
    "keydown": "$event.stopPropagation(); $event.stopImmediatePropagation();",
    "keypress": "$event.stopPropagation(); $event.stopImmediatePropagation();",
}


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
