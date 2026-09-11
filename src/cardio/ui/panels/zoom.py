"""Fitting the MPR views to a chosen set of labels."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ..common import SLIDER_CLASS, VOLUME_ACTIVE
from .snap import ACTION_CLASS

# There is nothing to fit until at least one label is chosen.
LABELS_CHOSEN = "zoom_labels.length > 0"


def zoom_panel(server, scene):
    """The label set the views are fitted to, the plane, and how much it fills."""
    if not (scene.volumes and scene.segmentations):
        return

    vuetify.VSelect(
        v_if=f"{VOLUME_ACTIVE} && segmentation_items.length >= 2",
        v_model=("zoom_seg_label",),
        items=("segmentation_items",),
        item_title="title",
        item_value="value",
        label="Segmentation",
        hide_details=True,
        classes="mb-2",
    )
    vuetify.VSelect(
        v_if=VOLUME_ACTIVE,
        v_model=("zoom_labels",),
        items=("zoom_available_labels",),
        item_title="title",
        item_value="value",
        label="Labels",
        title="The labels the views are fitted to",
        multiple=True,
        chips=True,
        hide_details=True,
        density="compact",
        classes="mb-2",
    )
    vuetify.VSelect(
        v_if=VOLUME_ACTIVE,
        v_model=("zoom_plane",),
        items=("zoom_plane_items",),
        item_title="title",
        item_value="value",
        label="Plane",
        title="The plane the labels are projected onto to measure the fit",
        hide_details=True,
        density="compact",
        classes="mb-2",
    )
    vuetify.VSlider(
        v_if=VOLUME_ACTIVE,
        v_model=("zoom_fill",),
        label="Fill",
        title="How much of the viewport the labels should take up",
        classes=f"{SLIDER_CLASS} mb-2",
        min=10,
        max=100,
        step=5,
        hide_details=True,
        thumb_label=True,
        disabled=(f"!({LABELS_CHOSEN})",),
    )
    # Stepped fine enough to reach the values that matter: a few mislabelled
    # voxels are a thousandth of a cloud, so the trim that clears them is small.
    vuetify.VSlider(
        v_if=VOLUME_ACTIVE,
        v_model=("label_percentile",),
        label="Cover",
        title=(
            "How much of the labels a measurement has to cover -- this fit, and"
            " the tile stack. Below 100 the outermost voxels are ignored, so a"
            " stray one cannot set the extent"
        ),
        classes=f"{SLIDER_CLASS} mb-2",
        min=99,
        max=100,
        step=0.01,
        hide_details=True,
        thumb_label=True,
    )
    with vuetify.VRow(
        v_if=VOLUME_ACTIVE,
        no_gutters=True,
        classes="mb-2 align-center",
    ):
        with vuetify.VCol():
            vuetify.VBtn(
                "Zoom to Labels",
                click=server.controller.zoom_to_labels,
                title="Zoom every view until the labels fill the viewport",
                disabled=(f"!({LABELS_CHOSEN})",),
                block=True,
                classes=ACTION_CLASS,
                prepend_icon="mdi-fit-to-screen-outline",
            )
        with vuetify.VCol(cols="auto", classes="ps-1"):
            vuetify.VCheckbox(
                v_model=("zoom_locked",),
                true_icon="mdi-lock",
                false_icon="mdi-lock-open-variant",
                color="primary",
                disabled=(f"!({LABELS_CHOSEN})",),
                title=(
                    "Hold the fit as the views move, and stop the zoom gesture"
                    " from fighting it"
                ),
                density="compact",
                hide_details=True,
            )
