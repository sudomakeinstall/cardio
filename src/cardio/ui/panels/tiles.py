"""The tile grid: how many cuts, where they are taken, and how they are arranged."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ...tile import TileSource
from ..common import SLIDER_CLASS, TILE_ACTIVE
from .snap import TRAVERSE_READY

# The sources that step a fixed plane along its own normal, and so want to be
# told which plane that is.
PARALLEL_SOURCE = f"tile_source !== '{TileSource.TRAVERSE.value}'"

# The traverse source is the one that needs a selection made elsewhere.
TRAVERSING = f"tile_source === '{TileSource.TRAVERSE.value}'"

# The labels source keeps a selection of its own, which starts out empty.
SPANNING = f"tile_source === '{TileSource.LABELS.value}'"
LABELS_CHOSEN = "tile_labels.length > 0"


def tiles_panel(server, scene):
    """Entering tile mode, and what the grid shows once there."""
    if not scene.volumes:
        return

    # Two of the three sources read a segmentation; a scene without one is
    # offered the source that does not, rather than a source it cannot use.
    sources = [(TileSource.SPACING, "Spacing")]
    if scene.segmentations:
        sources = [
            (TileSource.TRAVERSE, "Traverse"),
            *sources,
            (TileSource.LABELS, "Labels"),
        ]

    vuetify.VBtn(
        "Tile View",
        click="maximized_view = maximized_view === 'tile' ? '' : 'tile'",
        title="Show several cuts of the volume side by side",
        prepend_icon="mdi-view-grid-outline",
        block=True,
        classes="mb-2",
        variant=(f"{TILE_ACTIVE} ? 'tonal' : 'text'",),
    )

    with vuetify.VBtnToggle(
        v_if=TILE_ACTIVE,
        v_model=("tile_source",),
        mandatory=True,
        classes="mb-2",
    ):
        for source, label in sources:
            vuetify.VBtn(value=source.value, text=label)

    # Only the two sources a segmentation makes possible say anything here, and
    # their bindings read pickers a scene without one never fills.
    if scene.segmentations:
        vuetify.VAlert(
            "Choose Traverse mode and Groups A, B and C to fill the tiles.",
            v_if=f"{TILE_ACTIVE} && {TRAVERSING} && !({TRAVERSE_READY})",
            type="info",
            classes="mb-2",
            variant="tonal",
        )

        vuetify.VSelect(
            v_if=f"{TILE_ACTIVE} && {SPANNING} && segmentation_items.length >= 2",
            v_model=("tile_seg_label",),
            items=("segmentation_items",),
            item_title="title",
            item_value="value",
            label="Segmentation",
            hide_details=True,
            classes="mb-2",
        )

        vuetify.VSelect(
            v_if=f"{TILE_ACTIVE} && {SPANNING}",
            v_model=("tile_labels",),
            items=("tile_available_labels",),
            item_title="title",
            item_value="value",
            label="Labels",
            title="The labels the grid spans, first tile to last",
            multiple=True,
            chips=True,
            hide_details=True,
            density="compact",
            classes="mb-2",
        )

        vuetify.VAlert(
            "Choose the labels the tiles should span.",
            v_if=f"{TILE_ACTIVE} && {SPANNING} && !({LABELS_CHOSEN})",
            type="info",
            classes="mb-2",
            variant="tonal",
        )

    vuetify.VSelect(
        v_if=f"{TILE_ACTIVE} && {PARALLEL_SOURCE}",
        v_model=("tile_plane",),
        items=("tile_plane_items",),
        item_title="title",
        item_value="value",
        label="Plane",
        title="The plane each tile is cut in, and stepped along the normal of",
        hide_details=True,
        density="compact",
        classes="mb-2",
    )

    vuetify.VSlider(
        v_if=f"{TILE_ACTIVE} && tile_source === '{TileSource.SPACING.value}'",
        v_model=("tile_spacing",),
        label="Spacing",
        title="Millimetres between adjacent cuts",
        classes=f"{SLIDER_CLASS} mb-2",
        min=0.5,
        max=25,
        step=0.5,
        hide_details=True,
        thumb_label=True,
    )

    # Every source walks a path, so every source can walk it the other way.
    vuetify.VCheckbox(
        v_if=TILE_ACTIVE,
        v_model=("tile_reverse",),
        label="Reverse",
        title="Take the tiles from the far end of the path, without turning the cut",
        true_icon="mdi-swap-vertical",
        false_icon="mdi-swap-vertical",
        color="primary",
        density="compact",
        hide_details=True,
        classes="mb-2",
    )

    with vuetify.VRow(v_if=TILE_ACTIVE, no_gutters=True, classes="align-center"):
        for variable, label in (("tile_rows", "Rows"), ("tile_cols", "Columns")):
            with vuetify.VCol(classes="pe-1"):
                vuetify.VSelect(
                    v_model=(variable,),
                    items=("tile_sizes",),
                    label=label,
                    hide_details=True,
                    density="compact",
                )
        with vuetify.VCol(cols="auto", classes="ps-1 text-caption"):
            vuetify.VBtn(
                "{{ tile_rows * tile_cols }} tiles",
                click=server.controller.reset_tile_cameras,
                title="Refit every tile to one shared scale",
                variant="text",
                density="compact",
            )
