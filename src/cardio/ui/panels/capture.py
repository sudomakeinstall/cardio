"""Choosing which viewports a cine capture writes."""

# Third Party
from trame.widgets import vuetify3 as vuetify


def capture_panel(server, scene):
    """Viewport checkboxes and the capture button."""
    vuetify.VListSubheader("Screenshot Viewports")
    with vuetify.VRow(classes="mx-1 mb-1"):
        for key, label in (
            ("vr", "3D"),
            ("axial", "Axial"),
            ("coronal", "Coronal"),
            ("sagittal", "Sagittal"),
        ):
            vuetify.VCheckbox(
                v_model=(
                    f"screenshot_viewport_{key}",
                    key in scene.screenshot_viewports,
                ),
                label=label,
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
            disabled=(
                "!screenshot_viewport_vr && !screenshot_viewport_axial && !screenshot_viewport_coronal && !screenshot_viewport_sagittal",
                False,
            ),
        )

    vuetify.VDivider(classes="my-2")
