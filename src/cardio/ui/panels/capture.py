"""Choosing what a cine capture writes, and in what format."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ...capture import CaptureFormat
from ...logic.capture import VIEWPORTS

VIEWPORT_LABELS = {
    "vr": "3D",
    "axial": "Axial",
    "coronal": "Coronal",
    "sagittal": "Sagittal",
    "tile": "Tiles",
}

# What each format is, in the terms that decide between them: a still per
# frame, one animation, or a series a DICOM viewer can open.
FORMAT_LABELS = {
    CaptureFormat.PNG: "PNG stills",
    CaptureFormat.JPEG: "JPEG stills",
    CaptureFormat.GIF: "GIF animation",
    CaptureFormat.MP4: "MP4 animation",
    CaptureFormat.DICOM_RENDERED: "DICOM (as shown)",
    CaptureFormat.DICOM_DATA: "DICOM (image data)",
}

FORMAT_ITEMS = [
    {"title": FORMAT_LABELS[fmt], "value": fmt.value} for fmt in CaptureFormat
]

# A viewport can only be captured while the layout is drawing it, so a capture
# needs one that is both ticked and on screen. A ticked viewport that is not
# showing stays ticked and greys out, so switching layout and back keeps the
# selection.
ANY_AVAILABLE = " || ".join(
    f"(screenshot_viewport_{name} && capture_available.includes('{name}'))"
    for name in VIEWPORTS
)

OFF_SCREEN = "This viewport is captured only while the layout is showing it"


def capture_panel(server, scene):
    """The format, the viewport checkboxes and the capture button."""
    vuetify.VListSubheader("Format")
    with vuetify.VRow(classes="mx-1 mb-1"):
        vuetify.VSelect(
            v_model=("capture_format", scene.capture_format.value),
            items=("capture_formats", FORMAT_ITEMS),
            density="compact",
            hide_details=True,
            classes="mx-1",
        )

    vuetify.VListSubheader("Viewports")
    with vuetify.VRow(classes="mx-1 mb-1"):
        for key in VIEWPORTS:
            vuetify.VCheckbox(
                v_model=(
                    f"screenshot_viewport_{key}",
                    key in scene.screenshot_viewports,
                ),
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
        model_value=("capture_progress", 0),
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
