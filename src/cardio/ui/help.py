"""The keyboard and mouse reference dialog."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ..window_level import presets
from .common import sheet_dialog


def help_dialog(scene):
    """The shortcut reference, toggled with the `h` key."""
    with sheet_dialog(
        "help_overlay_visible",
        "Keyboard Shortcuts & Controls",
        scene.view.help_visible,
    ):
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
                    html.Td("i")
                    html.Td("Toggle the scene metadata window")
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
                    html.Td("Widen/narrow window")
                with html.Tr():
                    html.Td("Left Drag ↑/↓")
                    html.Td("Decrease/increase level")
                with html.Tr():
                    html.Td("Left + Right Drag ↑/↓")
                    html.Td("Scroll through slices")
                with html.Tr():
                    html.Td("Scroll Wheel")
                    html.Td("Scroll through slices")
                with html.Tr():
                    html.Td("Middle Drag")
                    html.Td("Pan; the other views follow")
                with html.Tr():
                    html.Td("Left + Middle Drag ↻")
                    html.Td("Rotate; drag around the crosshair")
                with html.Tr():
                    html.Td("Right + Middle Drag ↑/↓")
                    html.Td("Zoom every view in/out together")

        html.P(
            "A snap lock holds what it owns: locking the position "
            "suspends pan and slice scrolling, locking the orientation "
            "suspends rotation. Centring and aligning are one-off, and "
            "leave every gesture available. In traverse mode scrolling "
            "travels the path instead, so a lock does not suspend it.",
            classes="text-caption mt-3",
        )
