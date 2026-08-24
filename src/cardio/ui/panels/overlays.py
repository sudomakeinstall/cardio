"""Segmentation overlays drawn on the MPR views."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ...state import ObjectState
from ..common import MPR_ACTIVE


def overlays_panel(server, scene):
    """Opacity and per-segmentation overlay toggles."""
    if scene.segmentations:
        vuetify.VDivider(classes="my-2", v_if=MPR_ACTIVE)

        vuetify.VListSubheader("MPR Overlays", v_if=MPR_ACTIVE)

        vuetify.VSlider(
            v_if=MPR_ACTIVE,
            v_model=("mpr_segmentation_opacity", 0.7),
            label="Overlay Opacity",
            min=0.0,
            max=1.0,
            step=0.05,
            hide_details=True,
            thumb_label=True,
        )

        for seg in scene.segmentations:
            vuetify.VCheckbox(
                v_if=MPR_ACTIVE,
                v_model=(ObjectState.of(seg).mpr_overlay, False),
                label=f"{seg.label}",
                hide_details=True,
            )

        vuetify.VDivider(classes="my-2")
