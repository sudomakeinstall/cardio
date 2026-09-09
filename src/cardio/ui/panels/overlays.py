"""Segmentation overlays drawn on the cuts: the MPR views and the tile grid."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ...state import ObjectState
from ..common import RESLICE_ACTIVE, SLIDER_CLASS, target_group


def slice_views_panel(server, scene):
    """Opacity and per-segmentation overlay toggles.

    Up wherever a cut is on screen, which is every layout but the volume
    rendering -- so the group is the other half of what Appearance offers, and
    the two are never both irrelevant.
    """
    if not scene.segmentations:
        return

    with target_group(
        "Slice Views",
        "mdi-layers-outline",
        "Controls the MPR views and the tile grid only",
        v_if=RESLICE_ACTIVE,
    ):
        vuetify.VSlider(
            v_model=("mpr_segmentation_opacity",),
            label="Opacity",
            title="Opacity of the segmentation overlays on the cuts",
            classes=SLIDER_CLASS,
            min=0.0,
            max=1.0,
            step=0.05,
            hide_details=True,
            thumb_label=True,
        )

        for seg in scene.segmentations:
            vuetify.VCheckbox(
                v_model=(ObjectState.of(seg).mpr_overlay,),
                label=f"{seg.label}",
                hide_details=True,
            )
