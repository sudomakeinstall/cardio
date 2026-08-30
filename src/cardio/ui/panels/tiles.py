"""The tile grid: how many cuts along the path, and how they are arranged."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ..common import TILE_ACTIVE
from .snap import TRAVERSE_READY

GRID_SIZES = [1, 2, 3, 4, 5, 6]


def tiles_panel(server, scene):
    """Entering tile mode, and the shape of the grid once there."""
    if not (scene.volumes and scene.segmentations):
        return

    vuetify.VBtn(
        "Tile View",
        click="maximized_view = maximized_view === 'tile' ? '' : 'tile'",
        title="Show several cuts along the traverse path at once",
        prepend_icon="mdi-view-grid-outline",
        block=True,
        classes="mb-2",
        variant=(f"{TILE_ACTIVE} ? 'tonal' : 'text'",),
    )

    vuetify.VAlert(
        "Choose Traverse mode and Groups A, B and C to fill the tiles.",
        v_if=f"{TILE_ACTIVE} && !({TRAVERSE_READY})",
        type="info",
        classes="mb-2",
        variant="tonal",
    )

    with vuetify.VRow(v_if=TILE_ACTIVE, no_gutters=True, classes="align-center"):
        for variable, label in (("tile_rows", "Rows"), ("tile_cols", "Columns")):
            with vuetify.VCol(classes="pe-1"):
                vuetify.VSelect(
                    v_model=(variable, 3),
                    items=("tile_sizes", GRID_SIZES),
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
