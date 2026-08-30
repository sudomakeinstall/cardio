"""Choosing which viewports a cine capture writes."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ...logic.capture import VIEWPORTS

VIEWPORT_LABELS = {
    "vr": "3D",
    "axial": "Axial",
    "coronal": "Coronal",
    "sagittal": "Sagittal",
    "tile": "Tiles",
}

# At least one viewport has to be ticked for a capture to write anything.
ANY_VIEWPORT = " || ".join(f"screenshot_viewport_{name}" for name in VIEWPORTS)


def capture_panel(server, scene):
    """Viewport checkboxes and the capture button."""
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
            )

    with vuetify.VRow(justify="center", classes="my-3"):
        vuetify.VBtn(
            "Capture Cine",
            color="info",
            block=True,
            click=server.controller.screenshot,
            title=f"Capture cine to {scene.screenshot_directory}",
            prepend_icon="mdi-video",
            disabled=(f"!({ANY_VIEWPORT})", False),
        )
