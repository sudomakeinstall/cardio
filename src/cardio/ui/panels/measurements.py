"""The Measurements panel: the tracing mode, and the region being traced."""

# System
import functools as ft

# Third Party
from trame.widgets import client, html
from trame.widgets import vuetify3 as vuetify

# Internal
from ...planimetry import ContourStyle
from ..common import RESLICE_ACTIVE, SWALLOW_KEYS


def measurements_panel(server, scene):
    """Enter the mode, then close, undo or cancel the region being traced."""
    vuetify.VBtn(
        "Trace a Region",
        v_if=RESLICE_ACTIVE,
        click=server.controller.toggle_measuring,
        color=("measuring ? 'primary' : undefined",),
        variant=("measuring ? 'tonal' : 'text'",),
        prepend_icon="mdi-vector-polyline",
        title="Place the points of a region on a cut (m)",
        block=True,
        classes="mb-2",
    )

    with vuetify.VRow(
        v_if=RESLICE_ACTIVE,
        no_gutters=True,
        classes="align-center mb-2",
    ):
        with vuetify.VCol(cols="5"):
            vuetify.VLabel("New Regions:")
        with vuetify.VCol(cols="7"):
            vuetify.VSelect(
                v_model=("measurement_contour",),
                items=("measurement_contour_items",),
                item_title="text",
                item_value="value",
                hide_details=True,
                density="compact",
                title=(
                    "How the next region is closed: straight between the "
                    "points, or a closed spline through them. A region "
                    "already closed keeps the curve it was measured as."
                ),
            )

    # Only while something is being traced: the buttons are the path that does
    # not have to be remembered, and the keys beside them are for a hand
    # already on the mouse.
    with vuetify.VCard(
        v_if="measurement_pending",
        variant="tonal",
        classes="pa-2 mb-2",
    ):
        html.Span(
            "{{ measurement_pending }} point{{ measurement_pending === 1 ? '' : 's' }}"
            " in {{ measurement_view }}",
            classes="text-caption",
        )
        with vuetify.VRow(no_gutters=True, classes="mt-2"):
            with vuetify.VCol(cols="6", classes="pr-1"):
                vuetify.VBtn(
                    "Close",
                    click=server.controller.close_measurement,
                    disabled=("measurement_pending < 3",),
                    color="success",
                    size="small",
                    block=True,
                    title="Close the region and measure it (right click)",
                )
            with vuetify.VCol(cols="3", classes="px-1"):
                vuetify.VBtn(
                    "Undo",
                    click=server.controller.undo_measurement_point,
                    size="small",
                    block=True,
                    title="Take back the last point (u)",
                )
            with vuetify.VCol(cols="3", classes="pl-1"):
                vuetify.VBtn(
                    "Cancel",
                    click=server.controller.cancel_measurement,
                    color="error",
                    size="small",
                    block=True,
                    title="Give up on this region (x)",
                )

    _traced_regions(server, scene)


# The list is written against ``measurement_data.measurements``, which is the
# whole set as one document key -- so a rename typed into a row is an edit of
# the set, and goes back through the controller like any other.
REGIONS = "measurement_data.measurements"

# The per-region style buttons. The same two the picker offers, but acting on
# one region that has already been measured rather than on the next one.
CONTOUR_BUTTONS = (
    (ContourStyle.POLYGON.value, "Straight"),
    (ContourStyle.SPLINE.value, "Spline"),
)


def _traced_regions(server, scene):
    """One row per region: what it is, what it came to, and where it was taken."""
    vuetify.VDivider(v_if=f"{REGIONS}.length", classes="my-2")

    with client.DeepReactive("measurement_data"):
        for i in range(scene.max_measurements):
            with vuetify.VSheet(
                v_if=f"{REGIONS}.length > {i}",
                border=True,
                rounded=True,
                classes="pa-2 mb-2",
                click=f"measurement_selected = {i}",
            ):
                with vuetify.VRow(no_gutters=True, classes="align-center"):
                    with vuetify.VCol(cols="auto", classes="mr-2"):
                        vuetify.VIcon(
                            "mdi-circle",
                            size="x-small",
                            color=(f"measurement_on_plane[{i}] ? 'success' : 'grey'",),
                            title=(
                                (
                                    f"measurement_on_plane[{i}]"
                                    " ? 'On the cut now showing'"
                                    " : 'Off the cut now showing; press Recall'"
                                ),
                            ),
                        )
                    with vuetify.VCol():
                        vuetify.VTextField(
                            v_model=(f"{REGIONS}[{i}].name",),
                            placeholder="Name",
                            hide_details=True,
                            density="compact",
                            variant="plain",
                            **SWALLOW_KEYS,
                        )
                    with vuetify.VCol(cols="auto"):
                        html.Span(
                            "{{ " + f"{REGIONS}[{i}].area.toFixed(1)" + " }} mm²",
                            classes="text-caption font-weight-medium",
                        )

                with vuetify.VRow(no_gutters=True, classes="align-center mt-1"):
                    with vuetify.VCol():
                        html.Span(
                            "{{ "
                            + f"{REGIONS}[{i}].view"
                            + " }} · frame {{ "
                            + f"{REGIONS}[{i}].frame"
                            + " }}",
                            classes="text-caption text-medium-emphasis",
                        )
                    with vuetify.VCol(cols="auto", classes="mr-1"):
                        # Spelled out rather than generated from the picker's
                        # items: there are two of them, and each carries its own
                        # dispatch, so a loop would only hide that.
                        with vuetify.VBtnToggle(
                            model_value=(f"{REGIONS}[{i}].contour",),
                            density="compact",
                            variant="outlined",
                            divided=True,
                            mandatory=True,
                        ):
                            for style, label in CONTOUR_BUTTONS:
                                vuetify.VBtn(
                                    label,
                                    value=style,
                                    size="x-small",
                                    click=ft.partial(
                                        server.controller.restyle_measurement,
                                        i,
                                        style,
                                    ),
                                    title=(
                                        "Close this region straight or splined "
                                        "and measure it again. The picker above "
                                        "sets the next region, not this one."
                                    ),
                                )
                    with vuetify.VCol(cols="auto"):
                        vuetify.VBtn(
                            icon="mdi-target",
                            click=ft.partial(server.controller.recall_measurement, i),
                            title="Put the cuts back where this was measured",
                            variant="text",
                            density="comfortable",
                        )
                    with vuetify.VCol(cols="auto"):
                        vuetify.VBtn(
                            icon="mdi-delete",
                            click=ft.partial(server.controller.delete_measurement, i),
                            title="Drop this region",
                            color="error",
                            variant="text",
                            density="comfortable",
                        )

    vuetify.VBtn(
        "Save Measurements",
        v_if=f"{REGIONS}.length",
        click=server.controller.save_measurements,
        color="success",
        block=True,
        classes="mb-2",
        prepend_icon="mdi-content-save",
    )

    with vuetify.VRow(
        v_if="measurements_saved_at",
        no_gutters=True,
        classes="align-center mb-2",
    ):
        vuetify.VIcon(
            icon=("measurements_stale ? 'mdi-alert-circle' : 'mdi-check-circle'",),
            color=("measurements_stale ? 'warning' : 'success'",),
            size="small",
            classes="mr-1",
        )
        html.Span(
            "Measurements saved at {{ measurements_saved_at }}"
            "{{ measurements_stale ? ' *' : '' }}",
            classes=(
                "'text-caption ' + (measurements_stale ? 'text-warning' : 'text-success')",
            ),
        )

    vuetify.VBtn(
        "Delete All Measurements",
        v_if=f"{REGIONS}.length",
        click=server.controller.clear_measurements,
        color="error",
        block=True,
        variant="text",
        prepend_icon="mdi-delete-sweep",
    )
