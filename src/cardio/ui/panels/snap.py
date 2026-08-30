"""Snapping the MPR origin and orientation to a segmentation feature."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ..common import NOT_TILE_ACTIVE, RESLICE_ACTIVE, SLIDER_CLASS, TILE_ACTIVE

# Modes that fit a plane, and so offer Align and the orientation lock.
PLANAR_MODE = "(snap_mode === 'interface' || snap_mode === 'traverse')"

# Every group the current mode needs has at least one label.
GROUPS_CHOSEN = (
    "snap_labels_a.length > 0"
    f" && (!{PLANAR_MODE} || snap_labels_b.length > 0)"
    " && (snap_mode !== 'traverse' || snap_labels_c.length > 0)"
)

# Left-aligned labels put every prepend icon in one column down the panel,
# which a centred block button cannot do -- its icon moves with its label.
ACTION_CLASS = "justify-start"

# A path for the tiles to sample: traverse mode, with all three groups chosen.
TRAVERSE_READY = f"snap_mode === 'traverse' && {GROUPS_CHOSEN}"

# Above this, the fitted plane is a poor description of the interface.
NONPLANAR_FLATNESS = 0.25


def volume_panel(server, scene):
    """The volume the MPR views resample; every other panel depends on it."""
    if scene.volumes:
        vuetify.VSelect(
            v_if=(
                "(!maximized_view || maximized_view === 'volume'"
                f" || {TILE_ACTIVE}) && volume_items.length >= 2"
            ),
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
            v_if=RESLICE_ACTIVE,
        )
        vuetify.VSelect(
            v_if=f"{RESLICE_ACTIVE} && snap_seg_items.length >= 2",
            v_model=("snap_seg_label", ""),
            items=("snap_seg_items", []),
            item_title="title",
            item_value="value",
            label="Segmentation",
            hide_details=True,
            classes="mb-2",
        )
        with vuetify.VBtnToggle(
            v_if=RESLICE_ACTIVE,
            v_model=("snap_mode", "label"),
            mandatory=True,
            classes="mb-2",
        ):
            vuetify.VBtn(value="label", text="Label")
            vuetify.VBtn(value="interface", text="Interface")
            vuetify.VBtn(value="traverse", text="Traverse")
        # Every mode stacks its group pickers down the drawer, so the panel
        # reads the same whether the mode needs one group or three.
        for variable, label, shown in (
            (
                "snap_labels_a",
                ("snap_mode === 'label' ? 'Labels' : 'Group A'",),
                RESLICE_ACTIVE,
            ),
            (
                "snap_labels_b",
                "Group B",
                f"{RESLICE_ACTIVE} && {PLANAR_MODE}",
            ),
            (
                "snap_labels_c",
                "Group C",
                f"{RESLICE_ACTIVE} && snap_mode === 'traverse'",
            ),
        ):
            vuetify.VSelect(
                v_if=shown,
                v_model=(variable, []),
                items=("snap_available_labels", []),
                item_title="title",
                item_value="value",
                label=label,
                multiple=True,
                chips=True,
                hide_details=True,
                density="compact",
                classes="mb-2",
            )
        # The slider drives a single pose, which the tile grid has no use for:
        # its tiles already span the whole path.
        vuetify.VSlider(
            v_if=f"{RESLICE_ACTIVE} && snap_mode === 'traverse' && {NOT_TILE_ACTIVE}",
            v_model=("snap_traverse", 0),
            label="Traverse",
            title="Travel from the A|B interface to the B|C interface",
            classes=f"{SLIDER_CLASS} mb-2",
            min=0,
            max=100,
            step=1,
            hide_details=True,
            thumb_label=True,
            disabled=(f"!({GROUPS_CHOSEN})",),
        )
        vuetify.VAlert(
            "No interface found between the selected groups.",
            v_if="snap_no_interface",
            type="warning",
            classes="mb-2",
            variant="tonal",
        )
        with vuetify.VRow(
            v_if=RESLICE_ACTIVE,
            no_gutters=True,
            classes="mb-2 align-center",
        ):
            with vuetify.VCol():
                vuetify.VBtn(
                    "Center",
                    click=server.controller.snap_to_centroid,
                    disabled=(f"!({GROUPS_CHOSEN})",),
                    block=True,
                    classes=ACTION_CLASS,
                    prepend_icon="mdi-target",
                )
            with vuetify.VCol(cols="auto", classes="ps-1"):
                vuetify.VCheckbox(
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
            v_if=f"{RESLICE_ACTIVE} && {PLANAR_MODE}",
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
                    classes=ACTION_CLASS,
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
        # Both reverse the selection, which is why they share a controller
        # function, but they mean different things and only one mode shows at a
        # time. ``GROUPS_CHOSEN`` already demands A and B here, and C as well in
        # traverse mode, so it serves as the guard for both.
        vuetify.VBtn(
            "Reverse",
            v_if=f"{RESLICE_ACTIVE} && snap_mode === 'interface'",
            click=server.controller.swap_snap_groups,
            title="Swap Group A and Group B, viewing the interface from the other side",
            disabled=(f"!({GROUPS_CHOSEN})",),
            block=True,
            classes=f"mb-2 {ACTION_CLASS}",
            prepend_icon="mdi-swap-horizontal",
        )
        vuetify.VBtn(
            "Reverse",
            v_if=f"{RESLICE_ACTIVE} && snap_mode === 'traverse'",
            click=server.controller.swap_snap_groups,
            title="Reverse the direction of travel, swapping Group A and Group C",
            disabled=(f"!({GROUPS_CHOSEN})",),
            block=True,
            classes=f"mb-2 {ACTION_CLASS}",
            prepend_icon="mdi-swap-horizontal",
        )
        # Undoing a snap is not a mode of snapping, so it sits below the modes
        # rather than among them, and needs no selection to be usable.
        vuetify.VBtn(
            "Reset",
            v_if=RESLICE_ACTIVE,
            click=server.controller.reset_snap,
            title=(
                "Clear the groups, release the locks, drop the interface"
                " alignment and recentre the views"
            ),
            block=True,
            classes=ACTION_CLASS,
            prepend_icon="mdi-restore",
            variant="text",
        )
