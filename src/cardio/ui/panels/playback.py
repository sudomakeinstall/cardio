"""Cine playback: frame stepping, speed and rotation."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ...image_quality import (
    DEFAULT_PLAYBACK_QUALITY,
    DEFAULT_PLAYBACK_RESOLUTION,
)


def playback_panel(server, scene):
    """Transport controls and the phase, speed and cycles sliders."""
    vuetify.VListSubheader("Playback Controls")

    with vuetify.VToolbar(flat=True):
        # NOTE: Previous/Next controls should be VBtn components, but we use
        # VCheckbox for consistent sizing/spacing with the other controls.
        # This may be easier to fix in Vuetify 3.
        vuetify.VCheckbox(
            value=False,
            true_icon="mdi-skip-previous-circle",
            false_icon="mdi-skip-previous-circle",
            hide_details=True,
            title="Previous",
            click=server.controller.decrement_frame,
            readonly=True,
        )

        vuetify.VSpacer()

        vuetify.VCheckbox(
            value=False,
            true_icon="mdi-skip-next-circle",
            false_icon="mdi-skip-next-circle",
            hide_details=True,
            title="Next",
            click=server.controller.increment_frame,
            readonly=True,
        )

        vuetify.VSpacer()

        vuetify.VCheckbox(
            v_model=("playing", False),
            true_icon="mdi-pause-circle",
            false_icon="mdi-play-circle",
            title="Play/Pause",
            hide_details=True,
        )

        vuetify.VSpacer()

        vuetify.VCheckbox(
            v_model=("incrementing", True),
            true_icon="mdi-movie-open-outline",
            false_icon="mdi-movie-open-off-outline",
            hide_details=True,
            title="Incrementing",
        )

        vuetify.VSpacer()

        vuetify.VCheckbox(
            v_model=("rotating", False),
            true_icon="mdi-autorenew",
            false_icon="mdi-autorenew-off",
            hide_details=True,
            title="Rotating",
        )

    vuetify.VSlider(
        v_model=("frame", scene.current_frame),
        hint="Phase",
        persistent_hint=True,
        min=0,
        max=scene.nframes - 1,
        step=1,
        hide_details=False,
        style="max-width: 300px",
        ticks=True,
        thumb_label=True,
    )

    vuetify.VSlider(
        v_model=("bpm", 60),
        hint="Speed",
        persistent_hint=True,
        min=20,
        max=120,
        step=1,
        hide_details=False,
        style="max-width: 300px",
        ticks=True,
        thumb_label=True,
    )

    vuetify.VSlider(
        v_model=("bpr", 3),
        hint="Cycles/Rotation",
        persistent_hint=True,
        min=1,
        max=360,
        step=1,
        hide_details=False,
        style="max-width: 300px",
        ticks=True,
        thumb_label=True,
    )

    vuetify.VSlider(
        v_model=("playback_quality", DEFAULT_PLAYBACK_QUALITY),
        hint="Playback Quality",
        persistent_hint=True,
        min=10,
        max=100,
        step=5,
        hide_details=False,
        style="max-width: 300px",
        ticks=True,
        thumb_label=True,
    )

    vuetify.VSlider(
        v_model=("playback_resolution", DEFAULT_PLAYBACK_RESOLUTION),
        hint="Playback Resolution",
        persistent_hint=True,
        min=25,
        max=100,
        step=5,
        hide_details=False,
        style="max-width: 300px",
        ticks=True,
        thumb_label=True,
    )
