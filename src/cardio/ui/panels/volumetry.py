"""What the volumetry charts measure, and the numbers they come to."""

# System
import functools as ft

# Third Party
from trame.widgets import client, html
from trame.widgets import vuetify3 as vuetify

# Internal
from ...volumetry import INDEXED_COLUMN, PAGE_COLUMNS, STRUCTURES
from ..common import SWALLOW_KEYS

VOLUMETRY_ACTIVE = "maximized_view === 'volumetry'"

# Said where the Add button is, because a scene that has never been told what
# to measure shows an empty chart, which looks the same as a broken one.
EMPTY = "Nothing measured yet. Add a structure and say which labels make it up."

# Said where the tick is, because a scene whose images carry no body size has
# nothing to index by and no way to tell from the tick alone.
UNINDEXED = "No body surface area to index by, from the images or the config"

# What the tick does, as against why it is not offered.
INDEXED = "Also report each volume divided by body surface area"

# A structure only becomes a page once it has both, which is what keeps a row
# that is still being filled in from emptying the chart beside it.
INCOMPLETE = "Named and given a label, this is measured"

# Shown under the density field when what is typed there cannot be used. A bad
# density is ignored rather than refused -- the structure is still a structure
# -- and a number silently not taken is worse than one turned down out loud.
POSITIVE = (
    "v => v === null || v === undefined || v === '' || Number(v) > 0"
    " || 'Must be greater than zero'"
)

# Said on the tick, because what it changes is the vocabulary of the page and
# not only whether two rows appear.
CHAMBER = (
    "A pumping chamber: its extremes are read as end-diastolic and "
    "end-systolic, with a stroke volume and an ejection fraction under them. "
    "Off for myocardium, a great vessel, or anything outside the heart, whose "
    "extremes are a maximum and a minimum"
)

# Whether anything has been measured, which is what the numbers below hang on.
MEASURED = "volumetry_structures.length"


def volumetry_panel(server, scene):
    """The segmentation measured, the structures measured off it, and the numbers."""
    if not scene.segmentations:
        return

    vuetify.VBtn(
        "Volumetry",
        click="maximized_view = maximized_view === 'volumetry' ? '' : 'volumetry'",
        title="Chamber volumes over the cardiac cycle",
        prepend_icon="mdi-chart-line",
        block=True,
        classes="mb-2",
        variant=(f"{VOLUMETRY_ACTIVE} ? 'tonal' : 'text'",),
    )

    vuetify.VSelect(
        v_if="segmentation_items.length >= 2",
        v_model=("volumetry_seg_label",),
        items=("segmentation_items",),
        item_title="title",
        item_value="value",
        label="Segmentation",
        title="The segmentation the volumes are measured off",
        hide_details=True,
        density="compact",
        classes="mb-2",
    )

    structure_editor(server, scene)

    vuetify.VSelect(
        v_if=f"{MEASURED} > 1",
        v_model=("volumetry_structure",),
        items=("volumetry_structures",),
        label="Structure",
        title="Which structure's page is shown; the export writes them all",
        hide_details=True,
        density="compact",
        classes="mb-2",
    )

    vuetify.VCheckbox(
        v_if=MEASURED,
        v_model=("volumetry_indexed",),
        label="Index by BSA",
        title=INDEXED,
        # Disabled and said out loud, rather than hidden: a tick that is not
        # there teaches nobody that indexing exists or what it wants.
        disabled=("!volumetry_indexable",),
        hide_details=True,
        density="compact",
        classes="mb-2",
    )

    html.Span(
        UNINDEXED,
        v_if=f"{MEASURED} && !volumetry_indexable",
        classes="text-caption text-medium-emphasis mx-2 d-block mb-2",
    )

    # The same numbers the chart paints under itself, so that a reader looking
    # at the drawer and one looking at the capture are reading one table.
    with vuetify.VTable(v_if=MEASURED, density="compact", classes="mb-2"):
        with html.Thead():
            with html.Tr():
                for column in (*PAGE_COLUMNS, INDEXED_COLUMN):
                    html.Th(
                        column,
                        v_if=f"volumetry_rows.length && '{column}' in volumetry_rows[0]",
                        classes="text-caption",
                    )
        with html.Tbody():
            with html.Tr(v_for="row in volumetry_rows", key="row.Metric"):
                for column in (*PAGE_COLUMNS, INDEXED_COLUMN):
                    html.Td(
                        f"{{{{ row['{column}'] }}}}",
                        v_if=f"'{column}' in row",
                        classes="text-caption",
                    )

    vuetify.VBtn(
        "Save Volumetry",
        v_if=MEASURED,
        color="info",
        block=True,
        click=server.controller.save_volumetry,
        title=f"Write the tables and one page per structure to {scene.volumetry_directory}",
        prepend_icon="mdi-content-save-outline",
        classes="my-2",
    )

    # What the last export did, in the shape the capture panel reports its own.
    with vuetify.VRow(
        v_if="volumetry_saved_at",
        no_gutters=True,
        classes="align-center mb-2",
    ):
        vuetify.VIcon(
            icon=("volumetry_ok ? 'mdi-check-circle' : 'mdi-alert-circle'",),
            color=("volumetry_ok ? 'success' : 'warning'",),
            size="small",
            classes="mr-1",
        )
        html.Span(
            "{{ volumetry_summary }} at {{ volumetry_saved_at }}",
            classes=(
                "'text-caption ' + (volumetry_ok ? 'text-success' : 'text-warning')",
            ),
        )


def structure_editor(server, scene):
    """One card per structure: what it is called, and what it is made of.

    Unrolled to a fixed number of cards rather than repeated by the client,
    the way the rotation stack is: each delete button has to call the server
    with its own index, which is a thing the page is built knowing and not a
    thing a v-for can hand back.
    """
    vuetify.VListSubheader("Structures")

    with client.DeepReactive("volumetry_groups"):
        for i in range(scene.max_volumetry_groups):
            with vuetify.VSheet(
                v_if=f"volumetry_groups.{STRUCTURES}.length > {i}",
                border=True,
                rounded=True,
                classes="pa-2 mb-2",
            ):
                with vuetify.VRow(no_gutters=True, classes="align-center"):
                    with vuetify.VCol():
                        vuetify.VTextField(
                            v_model=(f"volumetry_groups.{STRUCTURES}[{i}].name",),
                            placeholder="Name",
                            title=INCOMPLETE,
                            hide_details=True,
                            density="compact",
                            variant="plain",
                            **SWALLOW_KEYS,
                        )
                    with vuetify.VCol(cols="auto"):
                        vuetify.VBtn(
                            icon="mdi-delete",
                            click=ft.partial(server.controller.remove_structure, i),
                            title="Stop measuring this structure",
                            color="error",
                            variant="text",
                            density="comfortable",
                        )

                vuetify.VSelect(
                    v_model=(f"volumetry_groups.{STRUCTURES}[{i}].labels",),
                    items=("volumetry_available_labels",),
                    item_title="title",
                    item_value="value",
                    label="Labels",
                    title="The labels summed into this structure",
                    multiple=True,
                    chips=True,
                    hide_details=True,
                    density="compact",
                    classes="mb-2",
                )

                with vuetify.VRow(no_gutters=True, classes="align-center"):
                    with vuetify.VCol(cols="6"):
                        vuetify.VTextField(
                            v_model=(f"volumetry_groups.{STRUCTURES}[{i}].density",),
                            label="Density",
                            title="Grams per millilitre; set, this is reported as a mass",
                            suffix="g/mL",
                            type="number",
                            min="0",
                            step="0.01",
                            rules=(f"[{POSITIVE}]",),
                            # Auto, not hidden: hidden is what kept the rule
                            # above from ever being read.
                            hide_details="auto",
                            density="compact",
                            **SWALLOW_KEYS,
                        )
                    with vuetify.VCol(cols="6", classes="ps-2"):
                        vuetify.VCheckbox(
                            v_model=(f"volumetry_groups.{STRUCTURES}[{i}].chamber",),
                            label="Chamber",
                            title=CHAMBER,
                            hide_details=True,
                            density="compact",
                        )

    html.Span(
        EMPTY,
        v_if=f"!volumetry_groups.{STRUCTURES}.length",
        classes="text-caption text-medium-emphasis mx-2",
    )

    vuetify.VBtn(
        "Add Structure",
        click=server.controller.add_structure,
        title="Measure another structure off this segmentation",
        prepend_icon="mdi-plus",
        variant="text",
        block=True,
        classes="mb-2",
        disabled=(
            f"volumetry_groups.{STRUCTURES}.length >= {scene.max_volumetry_groups}",
        ),
    )
