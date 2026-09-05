"""Choosing what a cine capture writes, and in what format."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ...logic.capture import VIEWPORTS
from ...state import screenshot_viewport

VIEWPORT_LABELS = {
    "vr": "3D",
    "axial": "Axial",
    "coronal": "Coronal",
    "sagittal": "Sagittal",
    "tile": "Tiles",
}

# A viewport can only be captured while the layout is drawing it, so a capture
# needs one that is both ticked and on screen. A ticked viewport that is not
# showing stays ticked and greys out, so switching layout and back keeps the
# selection.
ANY_AVAILABLE = " || ".join(
    f"({screenshot_viewport(name)} && capture_available.includes('{name}'))"
    for name in VIEWPORTS
)

OFF_SCREEN = "This viewport is captured only while the layout is showing it"


def capture_panel(server, scene):
    """The format, the viewport checkboxes and the capture button."""
    vuetify.VListSubheader("Format")
    with vuetify.VRow(classes="mx-1 mb-1"):
        vuetify.VSelect(
            v_model=("capture_format",),
            items=("capture_formats",),
            density="compact",
            hide_details=True,
            classes="mx-1",
        )

    vuetify.VListSubheader("Viewports")
    with vuetify.VRow(classes="mx-1 mb-1"):
        for key in VIEWPORTS:
            vuetify.VCheckbox(
                v_model=(screenshot_viewport(key),),
                label=VIEWPORT_LABELS[key],
                hide_details=True,
                classes="mx-1",
                disabled=(f"!capture_available.includes('{key}')", False),
                title=OFF_SCREEN,
            )

    with vuetify.VRow(justify="center", classes="my-3"):
        vuetify.VBtn(
            "Capture Cine",
            color="info",
            block=True,
            click=server.controller.screenshot,
            title=f"Capture cine to {scene.screenshot_directory}",
            prepend_icon="mdi-video",
            disabled=(f"!({ANY_AVAILABLE}) || capture_running", False),
        )

    vuetify.VProgressLinear(
        v_if="capture_running",
        model_value=("capture_progress",),
        color="info",
        height="6",
        rounded=True,
        classes="mb-2",
    )

    # What the last capture did, in the shape the rotations save reports its own.
    with vuetify.VRow(
        v_if="capture_saved_at",
        no_gutters=True,
        classes="align-center mb-2",
    ):
        vuetify.VIcon(
            icon=("capture_ok ? 'mdi-check-circle' : 'mdi-alert-circle'",),
            color=("capture_ok ? 'success' : 'warning'",),
            size="small",
            classes="mr-1",
        )
        html.Span(
            "{{ capture_summary }} at {{ capture_saved_at }}",
            classes=(
                "'text-caption ' + (capture_ok ? 'text-success' : 'text-warning')",
            ),
        )
