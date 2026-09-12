"""What the volumetry charts measure, and the numbers they come to."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ...volumetry import INDEXED_COLUMN, PAGE_COLUMNS

VOLUMETRY_ACTIVE = "maximized_view === 'volumetry'"

# The structures themselves are configured rather than chosen here: which
# labels make up a chamber is a clinical fact about the segmentation, and a
# list editor in a drawer is a poor place to state one.
UNCONFIGURED = "No structures configured; name them under [volumetry] in the config"

# Said where the tick is, because a scene whose images carry no body size has
# nothing to index by and no way to tell from the tick alone.
UNINDEXED = "Indexed columns need a height and a weight, from the images or the config"


def volumetry_panel(server, scene):
    """The segmentation measured, how it is laid out, and what it came to."""
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

    if not scene.volumetry.groups:
        html.Span(UNCONFIGURED, classes="text-caption text-medium-emphasis mx-2")
        return

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

    vuetify.VSelect(
        v_model=("volumetry_structure",),
        items=("volumetry_structures",),
        label="Structure",
        title="Which structure's page is shown; the export writes them all",
        hide_details=True,
        density="compact",
        classes="mb-2",
    )

    vuetify.VCheckbox(
        v_model=("volumetry_indexed",),
        label="Index by BSA",
        title=UNINDEXED,
        hide_details=True,
        density="compact",
        classes="mb-2",
    )

    # The same numbers the chart paints under itself, so that a reader looking
    # at the drawer and one looking at the capture are reading one table.
    with vuetify.VTable(density="compact", classes="mb-2"):
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
