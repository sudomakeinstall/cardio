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

PROMPT_CLASS = "cardio-console-prompt"


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

        _prompt(server)


def _prompt(server):
    """Where a call is typed, in the syntax the log above is written in.

    The field swallows its own key events. Without that they reach the render
    view's interactor, where typing `a` maximizes the axial view -- the same
    guard the rotation name field carries, for the same reason.

    It guards ``keydown`` and ``keypress`` and not ``keyup``, which is the one
    the submit is on: two handlers for the same event, one of them calling
    ``stopImmediatePropagation``, is a question about which of them runs first
    that there is no reason to be asking. ``keydown`` is what the interactor
    reads anyway.
    """
    with vuetify.VRow(no_gutters=True, classes="align-center px-4 py-2"):
        vuetify.VTextField(
            v_model=("console_input",),
            placeholder="add_rotation(axis='Z')",
            prefix=">",
            density="compact",
            variant="plain",
            hide_details=True,
            autofocus=True,
            classes=PROMPT_CLASS,
            __events=[("keyup_enter", "keyup.enter"), "keydown", "keypress"],
            keydown="$event.stopPropagation(); $event.stopImmediatePropagation();",
            keypress="$event.stopPropagation(); $event.stopImmediatePropagation();",
            keyup_enter=(server.controller.run_command, "[console_input]"),
        )
        vuetify.VBtn(
            icon="mdi-play",
            variant="text",
            density="compact",
            title="Run this call",
            disabled=("!console_input",),
            click=(server.controller.run_command, "[console_input]"),
        )
