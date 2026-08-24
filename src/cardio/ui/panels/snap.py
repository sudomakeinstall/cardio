"""Snapping the MPR origin and orientation to a segmentation feature."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ..common import MPR_ACTIVE, SLIDER_CLASS

# Modes that fit a plane, and so offer Align and the orientation lock.
PLANAR_MODE = "(snap_mode === 'interface' || snap_mode === 'traverse')"

# Every group the current mode needs has at least one label.
GROUPS_CHOSEN = (
    "snap_labels_a.length > 0"
    f" && (!{PLANAR_MODE} || snap_labels_b.length > 0)"
    " && (snap_mode !== 'traverse' || snap_labels_c.length > 0)"
)

# Above this, the fitted plane is a poor description of the interface.
NONPLANAR_FLATNESS = 0.25


def volume_panel(server, scene):
    """The volume the MPR views resample; every other panel depends on it."""
    if scene.volumes:
        vuetify.VSelect(
            v_if="(!maximized_view || maximized_view === 'volume') && volume_items.length >= 2",
            v_model=("active_volume_label", ""),
            items=("volume_items", []),
            item_title="text",
            item_value="value",
            label="Active Volume",
            title="The volume the MPR views resample",
            hide_details=True,
            density="compact",
        )


def snap_panel(server, scene):
    """Snapping the MPR origin and orientation to a segmentation feature."""
    if scene.volumes and scene.segmentations:
        vuetify.VListSubheader(
            "Snap & Align",
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
            vuetify.VBtn(value="traverse", text="Traverse")
            vuetify.VBtn(value="reset", text="Reset")
        with vuetify.VRow(
            v_if=f"{MPR_ACTIVE} && snap_mode !== 'reset' && snap_mode !== 'traverse'",
            no_gutters=True,
            classes="mb-2 align-center",
        ):
            with vuetify.VCol():
                vuetify.VSelect(
                    v_model=("snap_labels_a", []),
                    items=("snap_available_labels", []),
                    item_title="title",
                    item_value="value",
                    label=("snap_mode === 'interface' ? 'Group A' : 'Labels'",),
                    multiple=True,
                    chips=True,
                    hide_details=True,
                    density="compact",
                )
            with vuetify.VCol(
                v_if="snap_mode === 'interface'", cols="auto", classes="px-1"
            ):
                vuetify.VBtn(
                    click=server.controller.swap_snap_groups,
                    icon="mdi-swap-horizontal",
                    disabled=(
                        "snap_labels_a.length === 0 || snap_labels_b.length === 0",
                    ),
                    title="Swap Group A and Group B, viewing the interface from the other side",
                    variant="text",
                    density="compact",
                )
            with vuetify.VCol(v_if="snap_mode === 'interface'"):
                vuetify.VSelect(
                    v_model=("snap_labels_b", []),
                    items=("snap_available_labels", []),
                    item_title="title",
                    item_value="value",
                    label="Group B",
                    multiple=True,
                    chips=True,
                    hide_details=True,
                    density="compact",
                )
        # Three groups will not fit across the drawer, so traverse mode stacks
        # its pickers rather than reusing the interface row.
        with vuetify.VRow(
            v_if=f"{MPR_ACTIVE} && snap_mode === 'traverse'",
            no_gutters=True,
        ):
            for variable, label in (
                ("snap_labels_a", "Group A (start)"),
                ("snap_labels_b", "Group B (middle)"),
                ("snap_labels_c", "Group C (end)"),
            ):
                with vuetify.VCol(cols="12", classes="mb-2"):
                    vuetify.VSelect(
                        v_model=(variable, []),
                        items=("snap_available_labels", []),
                        item_title="title",
                        item_value="value",
                        label=label,
                        multiple=True,
                        chips=True,
                        hide_details=True,
                        density="compact",
                    )
        with vuetify.VRow(
            v_if=f"{MPR_ACTIVE} && snap_mode === 'traverse'",
            no_gutters=True,
            classes="mb-2 align-center",
        ):
            with vuetify.VCol():
                vuetify.VSlider(
                    v_model=("snap_traverse", 0),
                    label="Traverse",
                    title="Travel from the A|B interface to the B|C interface",
                    classes=SLIDER_CLASS,
                    min=0,
                    max=100,
                    step=1,
                    hide_details=True,
                    thumb_label=True,
                    disabled=(f"!({GROUPS_CHOSEN})",),
                )
            with vuetify.VCol(cols="auto", classes="ps-1"):
                vuetify.VBtn(
                    click=server.controller.swap_snap_groups,
                    icon="mdi-swap-horizontal",
                    disabled=(f"!({GROUPS_CHOSEN})",),
                    title="Reverse the direction of travel, swapping Group A and Group C",
                    variant="text",
                    density="compact",
                )
        vuetify.VAlert(
            "No interface found between the selected groups.",
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
                    "Center",
                    click=server.controller.snap_to_centroid,
                    disabled=(f"snap_mode !== 'reset' && !({GROUPS_CHOSEN})",),
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
                    disabled=(f"!({GROUPS_CHOSEN})",),
                    title="Re-snap automatically whenever the frame changes",
                    density="compact",
                    hide_details=True,
                )
        with vuetify.VRow(
            v_if=f"{MPR_ACTIVE} && {PLANAR_MODE}",
            no_gutters=True,
            classes="mb-2 align-center",
        ):
            with vuetify.VCol():
                vuetify.VBtn(
                    "Align",
                    click=server.controller.align_to_interface,
                    disabled=(f"!({GROUPS_CHOSEN})",),
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
                    disabled=(f"!({GROUPS_CHOSEN})",),
                    title="Re-align to the interface plane whenever the frame changes",
                    density="compact",
                    hide_details=True,
                )
        vuetify.VAlert(
            "Interface is not clearly planar; the fitted plane may not be meaningful.",
            v_if=f"interface_flatness > {NONPLANAR_FLATNESS}",
            type="warning",
            classes="mb-2",
            variant="tonal",
        )
