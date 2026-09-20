"""The keyboard and mouse reference dialog."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ..window_level import presets
from .common import sheet_dialog


def help_dialog():
    """The shortcut reference, toggled with the `?` key."""
    with sheet_dialog("help_overlay_visible", "Keyboard Shortcuts & Controls"):
        html.H3("Keyboard Shortcuts", classes="text-h6 mb-3")
        with vuetify.VTable(density="compact", classes="mb-4"):
            with html.Thead():
                with html.Tr():
                    html.Th("Key")
                    html.Th("Action")
            with html.Tbody():
                with html.Tr():
                    html.Td("?")
                    html.Td("Toggle this help window")
                with html.Tr():
                    html.Td("i")
                    html.Td("Toggle the scene metadata window")
                with html.Tr():
                    html.Td("`")
                    html.Td("Toggle the action console")
                with html.Tr():
                    html.Td("v")
                    html.Td("Toggle 3D volume view")
                with html.Tr():
                    html.Td("a")
                    html.Td("Toggle upper-left view")
                with html.Tr():
                    html.Td("c")
                    html.Td("Toggle lower-left view")
                with html.Tr():
                    html.Td("s")
                    html.Td("Toggle lower-right view")
                with html.Tr():
                    html.Td("t")
                    html.Td("Toggle tile view")
                with html.Tr():
                    html.Td("y")
                    html.Td("Toggle volumetry charts")
                with html.Tr():
                    html.Td("+")
                    html.Td("Toggle crosshairs")
                with html.Tr():
                    html.Td("h")
                    html.Td("Step back one frame")
                with html.Tr():
                    html.Td("l")
                    html.Td("Step forward one frame")
                with html.Tr():
                    html.Td("j")
                    html.Td("Scroll back one slice")
                with html.Tr():
                    html.Td("k")
                    html.Td("Scroll forward one slice")
                with html.Tr():
                    html.Td("m")
                    html.Td("Trace a region on a cut")
                with html.Tr():
                    html.Td("Enter")
                    html.Td("Close the region, or finish a correction")
                with html.Tr():
                    html.Td("u")
                    html.Td("Take back the last point placed")
                with html.Tr():
                    html.Td("x")
                    html.Td("Give up on the region being traced")
                with html.Tr():
                    html.Td("e")
                    html.Td("Correct the region the drawer has highlighted")

        html.P(
            "The slice keys move the cut under the cursor, as the wheel does, "
            "and in traverse mode they travel the path instead. The frame "
            "keys are for a study standing still: neither one moves while "
            "playback is running.",
            classes="text-caption mb-4",
        )

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
                with html.Tr():
                    html.Td("Left Click (tracing)")
                    html.Td("Place a point of the region")
                with html.Tr():
                    html.Td("Right Click (tracing)")
                    html.Td("Close the region and measure it")
                with html.Tr():
                    html.Td("Left Drag (correcting)")
                    html.Td("Move the point the drag started on")
                with html.Tr():
                    html.Td("Left Click (correcting)")
                    html.Td("Add a point where the contour was clicked")
                with html.Tr():
                    html.Td("Right Click (correcting)")
                    html.Td("Take away the point that was clicked")

        html.P(
            "A region is drawn only while the cuts are on the plane it was "
            "traced in, and at the frame it was traced at; move off either and "
            "it is hidden rather than drawn somewhere it does not belong. "
            "Recall puts the cuts back where it was measured. Tracing takes "
            "over the click alone, so every drag above goes on working while a "
            "region is being traced.",
            classes="text-caption mt-3",
        )

        html.P(
            "A region already closed is corrected one at a time, from the "
            "pencil on its row and only while its cut is showing. A left drag "
            "then belongs to the region only if it began on one of its points; "
            "every other drag is the window and level it always was.",
            classes="text-caption mt-3",
        )

        html.P(
            "A snap lock holds what it owns: locking the position "
            "suspends pan and slice scrolling, locking the orientation "
            "suspends rotation. Centring and aligning are one-off, and "
            "leave every gesture available. In traverse mode scrolling "
            "travels the path instead, so a lock does not suspend it.",
            classes="text-caption mt-3",
        )
