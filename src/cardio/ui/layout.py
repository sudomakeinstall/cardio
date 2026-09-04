"""The toolbar and the render viewports."""

# Third Party
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as vuetify

# Internal
from ..view import Theme


def toolbar(server, scene, layout):
    """Theme switch, close button and the busy indicator."""
    with layout.toolbar as bar:
        bar.dense = True

        vuetify.VSpacer()

        vuetify.VCheckbox(
            v_model=("theme_mode", scene.view.theme.value),
            true_value=Theme.DARK.value,
            false_value=Theme.LIGHT.value,
            label="Dark Mode",
            true_icon="mdi-lightbulb-off-outline",
            false_icon="mdi-lightbulb-outline",
            density="compact",
            style="max-width: 150px;",
        )

        # The two reference sheets, which the `i` and `h` keys also toggle
        vuetify.VCheckbox(
            value=False,
            true_icon="mdi-information-outline",
            false_icon="mdi-information-outline",
            label="Scene Metadata",
            click="metadata_overlay_visible = !metadata_overlay_visible",
            readonly=True,
        )

        vuetify.VCheckbox(
            value=False,
            true_icon="mdi-help-circle-outline",
            false_icon="mdi-help-circle-outline",
            label="Help",
            click="help_overlay_visible = !help_overlay_visible",
            readonly=True,
        )

        # Close button
        vuetify.VCheckbox(
            value=False,
            true_icon="mdi-close-circle",
            false_icon="mdi-close-circle",
            label="Close Application",
            click=server.controller.close_application,
            readonly=True,
        )

        vuetify.VProgressLinear(
            indeterminate=True,
            absolute=True,
            bottom=True,
            active=("trame__busy",),
        )


def viewports(server, scene, listeners, handled_events, update_all_views):
    """The quad view, plus a maximized container for each single view.

    ``listeners`` builds the interactor event bindings for a named view.
    """
    # Single VR view (volume maximized)
    with vuetify.VContainer(
        v_if="maximized_view === 'volume'",
        fluid=True,
        classes="pa-0 fill-height",
    ):
        view = vtk_widgets.VtkRemoteView(
            scene.renderWindow,
            interactor_events=("event_types", handled_events),
            **listeners("volume"),
            interactive_ratio=1,
        )
        server.controller.view_update = view.update
        server.controller.view_reset_camera = view.reset_camera
        server.controller.on_server_ready.add(view.update)

    # Quad-view layout (MPR mode - default)
    with vuetify.VContainer(
        v_if="!maximized_view",
        fluid=True,
        classes="pa-0",
        style="height: calc(100vh - 85px);",
    ):
        # Setup MPR render windows in Scene
        scene.setup_mpr_render_windows()

        # First row: Axial and Volume (50% height)
        with vuetify.VRow(classes="ma-0", style="height: 50%;"):
            with vuetify.VCol(cols="6", classes="pa-1", style="height: 100%;"):
                # Axial view
                axial_view = vtk_widgets.VtkRemoteView(
                    scene.mpr_views["axial"],
                    style="height: 100%; width: 100%;",
                    interactor_events=("event_types", handled_events),
                    **listeners("axial"),
                    interactive_ratio=1,
                )
            with vuetify.VCol(cols="6", classes="pa-1", style="height: 100%;"):
                # Volume view
                volume_view = vtk_widgets.VtkRemoteView(
                    scene.renderWindow,
                    style="height: 100%; width: 100%;",
                    interactor_events=("event_types", handled_events),
                    **listeners("volume_mpr"),
                    interactive_ratio=1,
                )

        # Second row: Coronal and Sagittal (50% height)
        with vuetify.VRow(classes="ma-0", style="height: 50%;"):
            with vuetify.VCol(cols="6", classes="pa-1", style="height: 100%;"):
                # Coronal view
                coronal_view = vtk_widgets.VtkRemoteView(
                    scene.mpr_views["coronal"],
                    style="height: 100%; width: 100%;",
                    interactor_events=("event_types", handled_events),
                    **listeners("coronal"),
                    interactive_ratio=1,
                )
            with vuetify.VCol(cols="6", classes="pa-1", style="height: 100%;"):
                # Sagittal view
                sagittal_view = vtk_widgets.VtkRemoteView(
                    scene.mpr_views["sagittal"],
                    style="height: 100%; width: 100%;",
                    interactor_events=("event_types", handled_events),
                    **listeners("sagittal"),
                    interactive_ratio=1,
                )

        # Set up controller functions for MPR mode
        server.controller.view_update = update_all_views
        server.controller.view_reset_camera = volume_view.reset_camera
        server.controller.on_server_ready.add(update_all_views)
        # Finalize MPR initialization after UI is ready to avoid race condition
        server.controller.on_server_ready.add(
            server.controller.finalize_mpr_initialization
        )

        # Store individual view update functions
        server.controller.axial_update = axial_view.update
        server.controller.coronal_update = coronal_view.update
        server.controller.sagittal_update = sagittal_view.update
        server.controller.volume_update = volume_view.update

    # Maximized axial view
    with vuetify.VContainer(
        v_if="maximized_view === 'axial'",
        fluid=True,
        classes="pa-0 fill-height",
    ):
        axial_maximized_view = vtk_widgets.VtkRemoteView(
            scene.mpr_views["axial"],
            interactor_events=("event_types", handled_events),
            **listeners("axial"),
            interactive_ratio=1,
        )

    # Maximized coronal view
    with vuetify.VContainer(
        v_if="maximized_view === 'coronal'",
        fluid=True,
        classes="pa-0 fill-height",
    ):
        coronal_maximized_view = vtk_widgets.VtkRemoteView(
            scene.mpr_views["coronal"],
            interactor_events=("event_types", handled_events),
            **listeners("coronal"),
            interactive_ratio=1,
        )

    # Tile view: one window, one renderer per tile, so the grid can be reshaped
    # without rebuilding the layout
    with vuetify.VContainer(
        v_if="maximized_view === 'tile'",
        fluid=True,
        classes="pa-0 fill-height",
    ):
        scene.setup_tile_render_window()
        tile_view = vtk_widgets.VtkRemoteView(
            scene.tile_views.window,
            interactor_events=("event_types", handled_events),
            **listeners("tile"),
            interactive_ratio=1,
        )
        server.controller.tile_update = tile_view.update

    # Maximized sagittal view
    with vuetify.VContainer(
        v_if="maximized_view === 'sagittal'",
        fluid=True,
        classes="pa-0 fill-height",
    ):
        sagittal_maximized_view = vtk_widgets.VtkRemoteView(
            scene.mpr_views["sagittal"],
            interactor_events=("event_types", handled_events),
            **listeners("sagittal"),
            interactive_ratio=1,
        )
