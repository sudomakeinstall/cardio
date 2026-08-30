"""The keyboard and mouse reference dialog."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ..window_level import presets


def help_dialog():
    """The shortcut reference, toggled with the `h` key."""
    with vuetify.VDialog(
        v_model=("help_overlay_visible", False),
        max_width="700px",
        scrim="rgba(0, 0, 0, 0.7)",
    ):
        with vuetify.VCard(
            classes="pa-6",
            style="background: rgba(33, 33, 33, 0.95);",
        ):
            vuetify.VCardTitle("Keyboard Shortcuts & Controls", classes="text-h5 mb-4")

            with vuetify.VCardText():
                html.H3("Keyboard Shortcuts", classes="text-h6 mb-3")
                with vuetify.VTable(density="compact", classes="mb-4"):
                    with html.Thead():
                        with html.Tr():
                            html.Th("Key")
                            html.Th("Action")
                    with html.Tbody():
                        with html.Tr():
                            html.Td("h")
                            html.Td("Toggle this help window")
                        with html.Tr():
                            html.Td("v")
                            html.Td("Toggle 3D volume view")
                        with html.Tr():
                            html.Td("a")
                            html.Td("Toggle axial view")
                        with html.Tr():
                            html.Td("c")
                            html.Td("Toggle coronal view")
                        with html.Tr():
                            html.Td("s")
                            html.Td("Toggle sagittal view")
                        with html.Tr():
                            html.Td("t")
                            html.Td("Toggle tile view")
                        with html.Tr():
                            html.Td("l")
                            html.Td("Toggle crosshairs")

                html.H3("Window/Level Presets", classes="text-h6 mb-3")
                with vuetify.VTable(density="compact", classes="mb-4"):
                    with html.Thead():
                        with html.Tr():
                            html.Th("Key")
                            html.Th("Preset")
                            html.Th("Window")
                            html.Th("Level")
                    with html.Tbody():
                        for key, preset in presets.items():
                            with html.Tr():
                                html.Td(str(key))
                                html.Td(preset.name)
                                html.Td(str(preset.window))
                                html.Td(str(preset.level))

                html.H3("Mouse Controls (MPR Mode)", classes="text-h6 mb-3")
                with vuetify.VTable(density="compact"):
                    with html.Thead():
                        with html.Tr():
                            html.Th("Action")
                            html.Th("Effect")
                    with html.Tbody():
                        with html.Tr():
                            html.Td("Left Drag ←/→")
                            html.Td("Narrow/widen window")
                        with html.Tr():
                            html.Td("Left Drag ↑/↓")
                            html.Td("Increase/decrease level")
                        with html.Tr():
                            html.Td("Right Drag ↑/↓")
                            html.Td("Scroll through slices")

            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn(
                    "Close",
                    click="help_overlay_visible = false",
                    variant="text",
                )
