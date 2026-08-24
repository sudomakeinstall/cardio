"""The MPR rotation stack: add, name, angle, visibility, delete."""

# System
import functools as ft

# Third Party
from trame.widgets import client, html
from trame.widgets import vuetify3 as vuetify

# Internal
from ..common import MPR_ACTIVE


def rotations_panel(server, scene):
    """Rotation sliders, units, index order, camera lock and save/reset."""
    if not scene.volumes:
        return

    # MPR Rotation controls
    vuetify.VListSubheader("Rotations", v_if=MPR_ACTIVE)

    # Rotation buttons
    with vuetify.VRow(
        v_if=MPR_ACTIVE,
        no_gutters=True,
        classes="mb-2",
    ):
        with vuetify.VCol(cols="4"):
            vuetify.VBtn(
                "X",
                click=server.controller.add_x_rotation,
                color="primary",
            )
        with vuetify.VCol(cols="4"):
            vuetify.VBtn(
                "Y",
                click=server.controller.add_y_rotation,
                color="primary",
            )
        with vuetify.VCol(cols="4"):
            vuetify.VBtn(
                "Z",
                click=server.controller.add_z_rotation,
                color="primary",
            )

    # Individual rotation sliders with DeepReactive
    with client.DeepReactive("mpr_rotation_data"):
        for i in range(scene.max_mpr_rotations):
            with vuetify.VContainer(
                v_if=f"{MPR_ACTIVE} && mpr_rotation_data.angles_list && mpr_rotation_data.angles_list.length > {i}",
                fluid=True,
                classes="pa-0 mb-2",
            ):
                with vuetify.VRow(no_gutters=True):
                    with vuetify.VCol(cols="12"):
                        vuetify.VTextField(
                            v_model=(f"mpr_rotation_data.angles_list[{i}].name",),
                            placeholder="Name",
                            hide_details=True,
                            readonly=(
                                f"!mpr_rotation_data.angles_list[{i}].name_editable",
                            ),
                            __events=["keydown", "keyup", "keypress"],
                            keydown="$event.stopPropagation(); $event.stopImmediatePropagation();",
                            keyup="$event.stopPropagation(); $event.stopImmediatePropagation();",
                            keypress="$event.stopPropagation(); $event.stopImmediatePropagation();",
                        )
                with vuetify.VRow(
                    no_gutters=True,
                    v_if=f"!mpr_rotation_data.angles_list[{i}].quaternion",
                ):
                    with vuetify.VCol(cols="12"):
                        vuetify.VSlider(
                            v_model=(f"mpr_rotation_data.angles_list[{i}].angle",),
                            min=("angle_units === 'radians' ? -Math.PI : -180",),
                            max=("angle_units === 'radians' ? Math.PI : 180",),
                            step=("angle_units === 'radians' ? 0.01 : 1",),
                            hide_details=True,
                            thumb_label=True,
                        )
                with vuetify.VRow(no_gutters=True, classes="align-center"):
                    vuetify.VSpacer()
                    with vuetify.VCol(cols="4"):
                        vuetify.VSelect(
                            v_if=f"!mpr_rotation_data.angles_list[{i}].quaternion",
                            v_model=(f"mpr_rotation_data.angles_list[{i}].axis",),
                            items=(["X", "Y", "Z"],),
                            hide_details=True,
                            label="Axis",
                        )
                    vuetify.VSpacer()
                    with vuetify.VCol(cols="auto"):
                        vuetify.VCheckbox(
                            v_model=(f"mpr_rotation_data.angles_list[{i}].visible",),
                            true_icon="mdi-eye",
                            false_icon="mdi-eye-off",
                            hide_details=True,
                            title="Toggle this rotation",
                        )
                    vuetify.VSpacer()
                    with vuetify.VCol(cols="auto"):
                        vuetify.VBtn(
                            icon="mdi-restore",
                            v_if=f"!mpr_rotation_data.angles_list[{i}].quaternion",
                            click=ft.partial(
                                server.controller.reset_rotation_angle,
                                i,
                            ),
                            title="Reset angle to zero",
                        )
                    vuetify.VSpacer()
                    with vuetify.VCol(cols="auto"):
                        vuetify.VBtn(
                            icon="mdi-delete",
                            click=ft.partial(
                                server.controller.remove_rotation_event,
                                i,
                            ),
                            color="error",
                            title="Remove this rotation",
                            disabled=(
                                f"!mpr_rotation_data.angles_list[{i}].deletable",
                            ),
                        )
                    vuetify.VSpacer()

    # Angle units selector
    with vuetify.VRow(
        v_if=MPR_ACTIVE,
        no_gutters=True,
        classes="align-center mb-2 mt-2",
    ):
        with vuetify.VCol(cols="4"):
            vuetify.VLabel("Units:")
        with vuetify.VCol(cols="8"):
            vuetify.VSelect(
                v_model=("angle_units", "radians"),
                items=("angle_units_items", []),
                item_title="text",
                item_value="value",
                hide_details=True,
            )

    # Axis convention selector
    with vuetify.VRow(
        v_if=MPR_ACTIVE,
        no_gutters=True,
        classes="align-center mb-2",
    ):
        with vuetify.VCol(cols="4"):
            vuetify.VLabel("Convention:")
        with vuetify.VCol(cols="8"):
            vuetify.VSelect(
                v_model=("index_order", "itk"),
                items=("index_order_items", []),
                item_title="text",
                item_value="value",
                hide_details=True,
            )

    # Fix Camera selector
    with vuetify.VRow(
        v_if="!maximized_view",
        no_gutters=True,
        classes="align-center mb-2",
    ):
        with vuetify.VCol(cols="4"):
            vuetify.VLabel("Fix Camera:")
        with vuetify.VCol(cols="8"):
            vuetify.VSelect(
                v_model=("camera_lock", "free"),
                items=("camera_lock_items", []),
                item_title="title",
                item_value="value",
                hide_details=True,
            )

    # Save rotations button
    vuetify.VBtn(
        "Save Rotations",
        v_if="!maximized_view && active_volume_label && mpr_rotation_data.angles_list && mpr_rotation_data.angles_list.length > 0",
        click=server.controller.save_rotation_angles,
        color="success",
        block=True,
        classes="mb-2",
        prepend_icon="mdi-content-save",
    )

    # Saved indicator
    with vuetify.VRow(
        v_if="rotations_saved_at",
        no_gutters=True,
        classes="align-center mb-2",
    ):
        vuetify.VIcon(
            icon=("rotations_stale ? 'mdi-alert-circle' : 'mdi-check-circle'",),
            color=("rotations_stale ? 'warning' : 'success'",),
            size="small",
            classes="mr-1",
        )
        html.Span(
            "Rotations saved at {{ rotations_saved_at }}{{ rotations_stale ? ' *' : '' }}",
            classes=(
                "'text-caption ' + (rotations_stale ? 'text-warning' : 'text-success')",
            ),
        )

    # Delete rotations button
    vuetify.VBtn(
        "Delete All Rotations",
        v_if="!maximized_view && active_volume_label && mpr_rotation_data.angles_list && mpr_rotation_data.angles_list.length > 0",
        click=server.controller.reset_rotations,
        color="error",
        block=True,
        classes="mb-2",
        prepend_icon="mdi-refresh",
        disabled=("!mpr_rotation_data.metadata.deletable",),
    )
