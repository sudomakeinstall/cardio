"""Segmentation overlays drawn on the cuts: the MPR views and the tile grid."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ...state import ObjectState
from ..common import (
    EYE_ICONS,
    KIND_ICONS,
    RESLICE_ACTIVE,
    SLIDER_CLASS,
    object_row,
    row_cell,
    target_group,
)


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
            title=("One opacity, shared by every overlay below and by the tiles"),
            classes=SLIDER_CLASS,
            min=0.0,
            max=1.0,
            step=0.05,
            hide_details=True,
            thumb_label=True,
        )

        for seg in scene.segmentations:
            overlay_row(seg)


def overlay_row(seg):
    """One segmentation on one line, drawn as its row in the other group is.

    Whether an overlay is on is the same question the rendering's eye asks, so
    it is asked the same way and in the same place -- the group above the row
    is what says which view the answer applies to.
    """
    eye_on, eye_off = EYE_ICONS

    with object_row(seg.label, KIND_ICONS[seg.kind]):
        with row_cell():
            vuetify.VCheckbox(
                v_model=(ObjectState.of(seg).mpr_overlay,),
                true_icon=eye_on,
                false_icon=eye_off,
                title=f"Draw {seg.label} on the cuts",
                density="compact",
                hide_details=True,
            )
