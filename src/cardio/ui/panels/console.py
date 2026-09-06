"""The action console: what the app was asked to do, most recent first."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ..common import DRAWER_WIDTH

CONSOLE_CLASS = "cardio-console"

# Where the dock starts and stops. Vuetify publishes the layout's own offsets
# as custom properties on the main region, which track the drawer if it is ever
# made to close; the drawer's width is the answer while it cannot, and is what
# the fallback says rather than a nought that would put the log under it.
INSET = (
    f"left: var(--v-layout-left, {DRAWER_WIDTH}px);"
    " right: var(--v-layout-right, 0px);"
    " bottom: var(--v-layout-bottom, 0px);"
)

CONSOLE_BODY_CLASS = "cardio-console-body"

# A refusal is a line like any other, in the colour that says it did nothing.
ENTRY_CLASS = "'text-pre-wrap ' + (entry.kind === 'error' ? 'text-error' : '')"


def console_panel(server, scene):
    """The dock along the bottom of the viewports.

    The log is the first thing here built from a state list rather than from a
    python loop over the scene: what it holds is not known when the page is,
    and grows as the session does.
    """
    with html.Div(v_if="console_visible", classes=CONSOLE_CLASS, style=INSET):
        with vuetify.VToolbar(density="compact", flat=True, color="transparent"):
            vuetify.VIcon("mdi-console", size="small", classes="mx-3")
            vuetify.VToolbarTitle("Console", classes="text-subtitle-2 flex-grow-0 pa-0")
            html.Span(
                "{{ console_entries.length }} lines",
                v_if="console_entries.length",
                classes="text-caption text-disabled ml-3",
            )
            vuetify.VSpacer()
            vuetify.VBtn(
                icon="mdi-delete-outline",
                variant="text",
                density="compact",
                title="Clear the log",
                click=server.controller.clear_console,
                disabled=("!console_entries.length",),
            )
            vuetify.VBtn(
                icon="mdi-close",
                variant="text",
                density="compact",
                title="Close the console (`)",
                click="console_visible = false",
                classes="mr-2",
            )

        with html.Div(classes=CONSOLE_BODY_CLASS):
            html.Div(
                "Nothing has been done yet.",
                v_if="!console_entries.length",
                classes="text-caption text-disabled px-4 py-2",
            )
            with html.Div(
                v_for="entry in console_entries",
                key="entry.n",
                classes="px-4 py-1 d-flex align-start",
            ):
                html.Span(
                    "{{ entry.at }}",
                    classes="text-disabled mr-3 flex-shrink-0",
                )
                html.Span(
                    "{{ entry.text }}",
                    classes=(ENTRY_CLASS,),
                )
                html.Span(
                    "×{{ entry.count }}",
                    v_if="entry.count > 1",
                    classes="text-disabled ml-3 flex-shrink-0",
                )

