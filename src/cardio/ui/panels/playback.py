"""Cine playback: frame stepping, speed, rotation and rendering cost."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ..common import SLIDER_CLASS


def playback_panel(server, scene):
    """Transport controls, then the phase, speed, cycles and cost sliders."""
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
            v_model=("incrementing", scene.playback.incrementing),
            true_icon="mdi-movie-open-outline",
            false_icon="mdi-movie-open-off-outline",
            hide_details=True,
            title="Advance frames while playing",
        )

        vuetify.VSpacer()

        vuetify.VCheckbox(
            v_model=("rotating", scene.playback.rotating),
            true_icon="mdi-autorenew",
            false_icon="mdi-autorenew-off",
            hide_details=True,
            title="Rotate the camera while playing",
        )

    vuetify.VSlider(
        v_model=("frame", scene.current_frame),
        label="Phase",
        title="Frame index within the cardiac cycle",
        classes=SLIDER_CLASS,
        min=0,
        max=scene.nframes - 1,
        step=1,
        hide_details=True,
        ticks=True,
        thumb_label=True,
    )

    vuetify.VSlider(
        v_model=("bpm", scene.playback.bpm),
        label="BPM",
        title="Playback speed, in beats per minute",
        classes=SLIDER_CLASS,
        min=20,
        max=120,
        step=1,
        hide_details=True,
        ticks=True,
        thumb_label=True,
    )

    vuetify.VSlider(
        v_model=("bpr", scene.playback.bpr),
        label="Beats/rot",
        title="Cardiac cycles per full rotation of the camera",
        classes=SLIDER_CLASS,
        min=1,
        max=360,
        step=1,
        hide_details=True,
        ticks=True,
        thumb_label=True,
    )

    vuetify.VSlider(
        v_model=("playback_quality", scene.playback.quality),
        label="Quality",
        title="JPEG encode quality while playing; 100 is full quality",
        classes=SLIDER_CLASS,
        min=10,
        max=100,
        step=5,
        hide_details=True,
        ticks=True,
        thumb_label=True,
    )

    vuetify.VSlider(
        v_model=("playback_resolution", scene.playback.resolution),
        label="Res",
        title="Render resolution while playing, as a percent of full",
        classes=SLIDER_CLASS,
        min=25,
        max=100,
        step=5,
        hide_details=True,
        ticks=True,
        thumb_label=True,
    )
