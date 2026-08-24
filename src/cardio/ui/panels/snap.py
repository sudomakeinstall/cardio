"""Volume selection and snapping the MPR origin to a segmentation feature."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ..common import MPR_ACTIVE

# Above this, the fitted plane is a poor description of the interface.
NONPLANAR_FLATNESS = 0.25


def snap_panel(server, scene):
    """The volume picker, plus the snap and align controls it enables."""
    if scene.volumes:
        vuetify.VSelect(
            v_if="(!maximized_view || maximized_view === 'volume') && volume_items.length >= 2",
            v_model=("active_volume_label", ""),
            items=("volume_items", []),
            item_title="text",
            item_value="value",
            title="Select which volume to use for MPR",
            hide_details=True,
        )

        if scene.segmentations:
            vuetify.VDivider(
                classes="my-2",
                v_if=MPR_ACTIVE,
            )
            vuetify.VListSubheader(
                "Snap to Centroid",
                v_if=MPR_ACTIVE,
            )
            vuetify.VSelect(
                v_if="!maximized_view && active_volume_label && snap_seg_items.length >= 2",
                v_model=("snap_seg_label", ""),
                items=("snap_seg_items", []),
                item_title="title",
                item_value="value",
                label="Segmentation",
                hide_details=True,
                classes="mb-2",
            )
            with vuetify.VBtnToggle(
                v_if=MPR_ACTIVE,
                v_model=("snap_mode", "label"),
                mandatory=True,
                classes="mb-2",
            ):
                vuetify.VBtn(value="label", text="Label")
                vuetify.VBtn(value="interface", text="Interface")
                vuetify.VBtn(value="reset", text="Reset")
            vuetify.VSelect(
                v_if="!maximized_view && active_volume_label && snap_mode !== 'reset'",
                v_model=("snap_labels_a", []),
                items=("snap_available_labels", []),
                item_title="title",
                item_value="value",
                label=("snap_mode === 'interface' ? 'Group A' : 'Labels'",),
                multiple=True,
                chips=True,
                hide_details=True,
                classes="mb-2",
            )
            vuetify.VSelect(
                v_if="!maximized_view && active_volume_label && snap_mode === 'interface'",
                v_model=("snap_labels_b", []),
                items=("snap_available_labels", []),
                item_title="title",
                item_value="value",
                label="Group B",
                multiple=True,
                chips=True,
                hide_details=True,
                classes="mb-2",
            )
            vuetify.VAlert(
                "No interface found between selected groups.",
                v_if="snap_no_interface",
                type="warning",
                classes="mb-2",
                variant="tonal",
            )
            with vuetify.VRow(
                v_if=MPR_ACTIVE,
                no_gutters=True,
                classes="mb-2 align-center",
            ):
                with vuetify.VCol():
                    vuetify.VBtn(
                        "Snap",
                        click=server.controller.snap_to_centroid,
                        disabled=(
                            "snap_mode === 'reset' ? false : snap_mode === 'label' ? snap_labels_a.length === 0 : snap_labels_a.length === 0 || snap_labels_b.length === 0",
                        ),
                        block=True,
                        prepend_icon="mdi-target",
                    )
                with vuetify.VCol(cols="auto", classes="ps-1"):
                    vuetify.VCheckbox(
                        v_if="snap_mode !== 'reset'",
                        v_model=("snap_locked", False),
                        true_icon="mdi-lock",
                        false_icon="mdi-lock-open-variant",
                        color="primary",
                        disabled=(
                            "snap_mode === 'label' ? snap_labels_a.length === 0 : snap_labels_a.length === 0 || snap_labels_b.length === 0",
                        ),
                        title="Re-snap automatically whenever the frame changes",
                        density="compact",
                        hide_details=True,
                    )
            with vuetify.VRow(
                v_if="!maximized_view && active_volume_label && snap_mode === 'interface'",
                no_gutters=True,
                classes="mb-2 align-center",
            ):
                with vuetify.VCol():
                    vuetify.VBtn(
                        "Align to Interface",
                        click=server.controller.align_to_interface,
                        disabled=(
                            "snap_labels_a.length === 0 || snap_labels_b.length === 0",
                        ),
                        title="Rotate the MPR views into the plane of the interface",
                        block=True,
                        prepend_icon="mdi-axis-arrow",
                    )
                with vuetify.VCol(cols="auto", classes="ps-1"):
                    vuetify.VCheckbox(
                        v_model=("snap_orientation_locked", False),
                        true_icon="mdi-lock",
                        false_icon="mdi-lock-open-variant",
                        color="primary",
                        disabled=(
                            "snap_labels_a.length === 0 || snap_labels_b.length === 0",
                        ),
                        title="Re-align to the interface plane whenever the frame changes",
                        density="compact",
                        hide_details=True,
                    )
            vuetify.VAlert(
                "Interface is not clearly planar; the fitted plane may "
                "not be meaningful.",
                v_if=f"interface_flatness > {NONPLANAR_FLATNESS}",
                type="warning",
                classes="mb-2",
                variant="tonal",
            )
