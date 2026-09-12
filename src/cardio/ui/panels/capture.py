"""Choosing what a cine capture writes, and in what format."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ...capture import CaptureFormat, writes_series
from ...state import (
    VIEWPORTS,
    capture_series_description,
    capture_series_number,
    screenshot_viewport,
)
from ..common import SWALLOW_KEYS

VIEWPORT_LABELS = {
    "vr": "3D",
    "axial": "Axial",
    "coronal": "Coronal",
    "sagittal": "Sagittal",
    "tile": "Tiles",
    "volumetry": "Volumetry",
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

# Only the DICOM formats write a series, and only a series has a number and a
# description. Read off the enum rather than tested as a name prefix, so a
# format that starts writing one is offered these without anything else moving.
NAMES_SERIES = " || ".join(
    f"capture_format === '{fmt.value}'" for fmt in CaptureFormat if writes_series(fmt)
)

UNNAMED = "Left empty, the series is named after the viewport and what it holds"

UNBANNERED = "Left empty, nothing is added to the margin"


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

    vuetify.VListSubheader("Banner")
    with vuetify.VRow(classes="mx-1 mb-1"):
        vuetify.VTextField(
            v_model=("capture_banner",),
            placeholder="NOT FOR CLINICAL USE",
            density="compact",
            hide_details=True,
            classes="mx-1",
            title="Text written in a band below every captured picture",
            **SWALLOW_KEYS,
        )
    html.Span(UNBANNERED, classes="text-caption text-medium-emphasis mx-2")

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

    with html.Div(v_if=NAMES_SERIES):
        vuetify.VListSubheader("Series")
        for key in VIEWPORTS:
            available = f"capture_available.includes('{key}')"
            # The row says why the pair is grey, and says nothing while it is
            # not: a disabled input passes the pointer through to the row, so
            # the reason is what shows on the fields that have one.
            with vuetify.VRow(
                no_gutters=True,
                classes="mx-1 mb-2 align-center",
                title=(f"{available} ? '' : '{OFF_SCREEN}'",),
            ):
                with vuetify.VCol(cols="4"):
                    vuetify.VTextField(
                        v_model=(capture_series_number(key),),
                        label=VIEWPORT_LABELS[key],
                        type="number",
                        min=0,
                        density="compact",
                        hide_details=True,
                        classes="mr-2",
                        disabled=(f"!{available}", False),
                        title=f"SeriesNumber the {VIEWPORT_LABELS[key]} capture is written with",
                        **SWALLOW_KEYS,
                    )
                with vuetify.VCol(cols="8"):
                    vuetify.VTextField(
                        v_model=(capture_series_description(key),),
                        placeholder="Description",
                        density="compact",
                        hide_details=True,
                        disabled=(f"!{available}", False),
                        title=UNNAMED,
                        **SWALLOW_KEYS,
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
