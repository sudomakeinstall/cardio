import functools as ft
import time

import numpy as np
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as vuetify

from .orientation import EulerAxis, euler_angle_to_rotation_matrix
from .scene import Scene
from .volume_property_presets import list_volume_property_presets
from .window_level import presets


class UI:
    @property
    def handled_events(self):
        return [
            "MouseMove",
            "LeftButtonPress",
            "LeftButtonRelease",
            "RightButtonPress",
            "RightButtonRelease",
            "KeyPress",
        ]

    def event_listeners_for_view(self, view_name):
        """Create event listeners for a specific view."""
        result = {}
        for event in self.handled_events:
            callback = ft.partial(self.on_event, view_name=view_name)
            result[event] = (callback, "[utils.vtk.event($event)]")
        return result

    def on_event(self, *args, view_name=None, **kwargs):
        if not args:
            return

        event = args[0]

        match event["type"]:
            case "KeyPress":
                key = event["key"]
                current_time = time.time() * 1000

                if key in self.last_keypress_time:
                    time_since_last = current_time - self.last_keypress_time[key]
                    if time_since_last < self.keypress_debounce_ms:
                        return

                self.last_keypress_time[key] = current_time

                if key.isdigit():
                    key_num = int(key)
                    if key_num in presets.keys():
                        self.server.state.mpr_window_level_preset = key_num
                elif key.lower() == "l":
                    current = getattr(self.server.state, "mpr_crosshairs_enabled", True)
                    self.server.state.mpr_crosshairs_enabled = not current

            case "LeftButtonPress":
                self.left_dragging = True
                self._store_mouse_position(view_name, event)

            case "LeftButtonRelease":
                self.left_dragging = False

            case "RightButtonPress":
                self.right_dragging = True
                self._store_mouse_position(view_name, event)

            case "RightButtonRelease":
                self.right_dragging = False

            case "MouseMove" if self.left_dragging:
                if (
                    view_name in {"axial", "sagittal", "coronal"}
                    and view_name in self.last_mouse_pos
                    and "position" in event
                ):
                    current_pos = [event["position"]["x"], event["position"]["y"]]
                    dx = current_pos[0] - self.last_mouse_pos[view_name][0]
                    dy = current_pos[1] - self.last_mouse_pos[view_name][1]
                    self.last_mouse_pos[view_name] = current_pos

                    current_window = getattr(self.server.state, "mpr_window", 400.0)
                    current_level = getattr(self.server.state, "mpr_level", 40.0)

                    window_delta = -dx * self.window_sensitivity
                    level_delta = -dy * self.level_sensitivity
                    new_window = max(1.0, current_window + window_delta)
                    new_level = current_level + level_delta

                    self.server.state.mpr_window = new_window
                    self.server.state.mpr_level = new_level

            case "MouseMove" if self.right_dragging:
                if (
                    view_name in {"axial", "sagittal", "coronal"}
                    and view_name in self.last_mouse_pos
                    and "position" in event
                ):
                    current_pos = [event["position"]["x"], event["position"]["y"]]
                    dy = current_pos[1] - self.last_mouse_pos[view_name][1]
                    self.last_mouse_pos[view_name] = current_pos

                    base_slice_delta = dy * self.slice_sensitivity
                    self._handle_slice_scroll(view_name, base_slice_delta)

    def _store_mouse_position(self, view_name, event):
        """Store mouse position for drag operations."""
        if view_name and "position" in event:
            self.last_mouse_pos[view_name] = [
                event["position"]["x"],
                event["position"]["y"],
            ]

    def _get_scroll_vector(self, view_name):
        """Get the current normal vector for a view after rotation."""
        base_normals = {
            "axial": np.array([0.0, 0.0, 1.0]),
            "sagittal": np.array([1.0, 0.0, 0.0]),
            "coronal": np.array([0.0, 1.0, 0.0]),
        }

        if view_name not in base_normals:
            return np.array([0.0, 0.0, 1.0])

        # Build cumulative rotation from visible rotations
        rotation_sequence = getattr(self.server.state, "mpr_rotation_sequence", [])
        cumulative_rotation = np.eye(3)

        for i in range(len(rotation_sequence)):
            is_visible = getattr(self.server.state, f"mpr_rotation_visible_{i}", True)
            if is_visible:
                angle = getattr(self.server.state, f"mpr_rotation_angle_{i}", 0)
                rotation = rotation_sequence[i]
                rotation_matrix = euler_angle_to_rotation_matrix(
                    EulerAxis(rotation["axis"]), angle
                )
                cumulative_rotation = cumulative_rotation @ rotation_matrix

        return cumulative_rotation @ base_normals[view_name]

    def _handle_slice_scroll(self, view_name, base_slice_delta):
        """Handle slice scrolling for a specific view using rotated scroll vectors."""
        if view_name not in {"axial", "sagittal", "coronal"}:
            return

        # Get current origin
        origin = getattr(self.server.state, "mpr_origin", [0.0, 0.0, 0.0])

        # Get scroll vector for this view (considering rotation)
        scroll_vector = self._get_scroll_vector(view_name)

        # Update origin along the rotated scroll direction
        new_origin = [
            origin[0] + base_slice_delta * scroll_vector[0],
            origin[1] + base_slice_delta * scroll_vector[1],
            origin[2] + base_slice_delta * scroll_vector[2],
        ]

        self.server.state.mpr_origin = new_origin

    def __init__(self, server, scene: Scene):
        self.server = server
        self.scene: Scene = scene
        self.left_dragging = False
        self.right_dragging = False
        self.last_mouse_pos = {}
        self.window_sensitivity = 5.0
        self.level_sensitivity = 2.0
        self.slice_sensitivity = 1.0
        self.last_keypress_time = {}
        self.keypress_debounce_ms = 100

        self.setup()

    def setup(self):
        self.server.state.trame__title = self.scene.project_name

        with SinglePageWithDrawerLayout(
            self.server, theme=("theme_mode", "dark")
        ) as layout:
            layout.icon.click = self.server.controller.view_reset_camera
            layout.title.set_text(self.scene.project_name)

            with layout.toolbar as toolbar:
                toolbar.dense = True

                vuetify.VSpacer()

                vuetify.VCheckbox(
                    v_model=("theme_mode", "dark"),
                    true_value="dark",
                    false_value="light",
                    label="Dark Mode",
                    true_icon="mdi-lightbulb-off-outline",
                    false_icon="mdi-lightbulb-outline",
                    density="compact",
                    style="max-width: 150px;",
                )

                # Close button
                vuetify.VCheckbox(
                    value=False,
                    true_icon="mdi-close-circle",
                    false_icon="mdi-close-circle",
                    label="Close Application",
                    click=self.server.controller.close_application,
                    readonly=True,
                )

                vuetify.VProgressLinear(
                    indeterminate=True,
                    absolute=True,
                    bottom=True,
                    active=("trame__busy",),
                )

            with layout.content:
                # Single VR view (default mode)
                with vuetify.VContainer(
                    v_if="!mpr_enabled",
                    fluid=True,
                    classes="pa-0 fill-height",
                ):
                    view = vtk_widgets.VtkRemoteView(
                        self.scene.renderWindow,
                        interactor_events=("event_types", self.handled_events),
                        **self.event_listeners_for_view("volume"),
                        interactive_ratio=1,
                    )
                    self.server.controller.view_update = view.update
                    self.server.controller.view_reset_camera = view.reset_camera
                    self.server.controller.on_server_ready.add(view.update)

                # Quad-view layout (MPR mode) - directly in content like app.py
                with vuetify.VContainer(
                    v_if="mpr_enabled",
                    fluid=True,
                    classes="pa-0",
                    style="height: calc(100vh - 85px);",
                ):
                    # Setup MPR render windows in Scene
                    self.scene.setup_mpr_render_windows()

                    # First row: Axial and Volume (50% height)
                    with vuetify.VRow(classes="ma-0", style="height: 50%;"):
                        with vuetify.VCol(
                            cols="6", classes="pa-1", style="height: 100%;"
                        ):
                            # Axial view
                            axial_view = vtk_widgets.VtkRemoteView(
                                self.scene.axial_renderWindow,
                                style="height: 100%; width: 100%;",
                                interactor_events=("event_types", self.handled_events),
                                **self.event_listeners_for_view("axial"),
                                interactive_ratio=1,
                            )
                        with vuetify.VCol(
                            cols="6", classes="pa-1", style="height: 100%;"
                        ):
                            # Volume view
                            volume_view = vtk_widgets.VtkRemoteView(
                                self.scene.renderWindow,
                                style="height: 100%; width: 100%;",
                                interactor_events=("event_types", self.handled_events),
                                **self.event_listeners_for_view("volume_mpr"),
                                interactive_ratio=1,
                            )

                    # Second row: Coronal and Sagittal (50% height)
                    with vuetify.VRow(classes="ma-0", style="height: 50%;"):
                        with vuetify.VCol(
                            cols="6", classes="pa-1", style="height: 100%;"
                        ):
                            # Coronal view
                            coronal_view = vtk_widgets.VtkRemoteView(
                                self.scene.coronal_renderWindow,
                                style="height: 100%; width: 100%;",
                                interactor_events=("event_types", self.handled_events),
                                **self.event_listeners_for_view("coronal"),
                                interactive_ratio=1,
                            )
                        with vuetify.VCol(
                            cols="6", classes="pa-1", style="height: 100%;"
                        ):
                            # Sagittal view
                            sagittal_view = vtk_widgets.VtkRemoteView(
                                self.scene.sagittal_renderWindow,
                                style="height: 100%; width: 100%;",
                                interactor_events=("event_types", self.handled_events),
                                **self.event_listeners_for_view("sagittal"),
                                interactive_ratio=1,
                            )

                    # Set up controller functions for MPR mode
                    self.server.controller.view_update = self._update_all_mpr_views
                    self.server.controller.view_reset_camera = volume_view.reset_camera
                    self.server.controller.on_server_ready.add(
                        self._update_all_mpr_views
                    )

                    # Store individual view update functions
                    self.server.controller.axial_update = axial_view.update
                    self.server.controller.coronal_update = coronal_view.update
                    self.server.controller.sagittal_update = sagittal_view.update
                    self.server.controller.volume_update = volume_view.update

            with layout.drawer:
                # MPR Mode Toggle (only show when volumes are present)
                if self.scene.volumes:
                    vuetify.VListSubheader("View Mode")
                    vuetify.VCheckbox(
                        v_model=("mpr_enabled", False),
                        label="Multi-Planar Reconstruction (MPR)",
                        title="Toggle between single 3D view and quad-view with slice planes",
                        dense=True,
                        hide_details=True,
                    )

                    # Volume selection dropdown (only show when MPR is enabled)
                    vuetify.VSelect(
                        v_if="mpr_enabled",
                        v_model=("active_volume_label", ""),
                        items=("volume_items", []),
                        item_title="text",
                        item_value="value",
                        title="Select which volume to use for MPR",
                        dense=True,
                        hide_details=True,
                    )

                    # Window/Level controls for MPR
                    vuetify.VListSubheader(
                        "Window/Level", v_if="mpr_enabled && active_volume_label"
                    )

                    vuetify.VSelect(
                        v_if="mpr_enabled && active_volume_label",
                        v_model=("mpr_window_level_preset", 7),
                        items=("mpr_presets", []),
                        item_title="text",
                        item_value="value",
                        label="Preset",
                        dense=True,
                        hide_details=True,
                    )

                    vuetify.VSlider(
                        v_if="mpr_enabled && active_volume_label",
                        v_model="mpr_window",
                        min=1.0,
                        max=2000.0,
                        step=1.0,
                        hint="Window",
                        persistent_hint=True,
                        dense=True,
                        hide_details=False,
                        thumb_label=True,
                    )

                    vuetify.VSlider(
                        v_if="mpr_enabled && active_volume_label",
                        v_model="mpr_level",
                        min=-1000.0,
                        max=1000.0,
                        step=1.0,
                        hint="Level",
                        persistent_hint=True,
                        dense=True,
                        hide_details=False,
                        thumb_label=True,
                    )

                    vuetify.VCheckbox(
                        v_if="mpr_enabled && active_volume_label",
                        v_model=("mpr_crosshairs_enabled", True),
                        label="Show Crosshairs",
                        dense=True,
                        hide_details=True,
                        classes="mt-2",
                    )

                    # MPR Rotation controls
                    vuetify.VListSubheader(
                        "Rotations", v_if="mpr_enabled && active_volume_label"
                    )

                    # Rotation buttons
                    with vuetify.VRow(
                        v_if="mpr_enabled && active_volume_label",
                        no_gutters=True,
                        classes="mb-2",
                    ):
                        with vuetify.VCol(cols="4"):
                            vuetify.VBtn(
                                "X",
                                click=self.server.controller.add_x_rotation,
                                small=True,
                                dense=True,
                                outlined=True,
                                color="primary",
                            )
                        with vuetify.VCol(cols="4"):
                            vuetify.VBtn(
                                "Y",
                                click=self.server.controller.add_y_rotation,
                                small=True,
                                dense=True,
                                outlined=True,
                                color="primary",
                            )
                        with vuetify.VCol(cols="4"):
                            vuetify.VBtn(
                                "Z",
                                click=self.server.controller.add_z_rotation,
                                small=True,
                                dense=True,
                                outlined=True,
                                color="primary",
                            )

                    # Reset rotations button
                    vuetify.VBtn(
                        "Reset",
                        v_if="mpr_enabled && active_volume_label && mpr_rotation_sequence && mpr_rotation_sequence.length > 0",
                        click=self.server.controller.reset_rotations,
                        small=True,
                        dense=True,
                        outlined=True,
                        color="warning",
                        block=True,
                        classes="mb-2",
                        prepend_icon="mdi-refresh",
                    )

                    # Individual rotation sliders
                    for i in range(self.scene.max_mpr_rotations):
                        with vuetify.VRow(
                            v_if=f"mpr_enabled && active_volume_label && mpr_rotation_sequence && mpr_rotation_sequence.length > {i}",
                            no_gutters=True,
                            classes="align-center mb-1",
                        ):
                            with vuetify.VCol(cols="8"):
                                vuetify.VSlider(
                                    v_model=(f"mpr_rotation_angle_{i}", 0),
                                    min=-180,
                                    max=180,
                                    step=1,
                                    hint=(
                                        f"mpr_rotation_axis_{i}",
                                        f"Rotation {i + 1}",
                                    ),
                                    persistent_hint=True,
                                    dense=True,
                                    hide_details=False,
                                    thumb_label=True,
                                )
                            with vuetify.VCol(cols="2"):
                                vuetify.VCheckbox(
                                    v_model=(f"mpr_rotation_visible_{i}", True),
                                    true_icon="mdi-eye",
                                    false_icon="mdi-eye-off",
                                    hide_details=True,
                                    dense=True,
                                    title="Toggle this rotation and all subsequent ones",
                                )
                            with vuetify.VCol(cols="2"):
                                vuetify.VBtn(
                                    icon="mdi-delete",
                                    click=ft.partial(
                                        self.server.controller.remove_rotation_event, i
                                    ),
                                    small=True,
                                    dense=True,
                                    color="error",
                                    title="Remove this rotation and all subsequent ones",
                                )

                    # Angle units selector
                    with vuetify.VRow(
                        v_if="mpr_enabled && active_volume_label",
                        no_gutters=True,
                        classes="align-center mb-2 mt-2",
                    ):
                        with vuetify.VCol(cols="4"):
                            vuetify.VLabel("Units:")
                        with vuetify.VCol(cols="8"):
                            vuetify.VSelect(
                                v_model=("angle_units", "degrees"),
                                items=("angle_units_items", []),
                                item_title="text",
                                item_value="value",
                                dense=True,
                                hide_details=True,
                                outlined=True,
                            )

                    # Save rotations button
                    vuetify.VBtn(
                        "Save Rotations",
                        v_if="mpr_enabled && active_volume_label && mpr_rotation_sequence && mpr_rotation_sequence.length > 0",
                        click=self.server.controller.save_rotation_angles,
                        small=True,
                        dense=True,
                        outlined=True,
                        color="success",
                        block=True,
                        classes="mb-2",
                        prepend_icon="mdi-content-save",
                    )

                    vuetify.VDivider(classes="my-2")

                vuetify.VListSubheader("Playback Controls")

                with vuetify.VToolbar(dense=True, flat=True):
                    # NOTE: Previous/Next controls should be VBtn components, but we use
                    # VCheckbox for consistent sizing/spacing with the other controls.
                    # This may be easier to fix in Vuetify 3.
                    vuetify.VCheckbox(
                        value=False,
                        true_icon="mdi-skip-previous-circle",
                        false_icon="mdi-skip-previous-circle",
                        hide_details=True,
                        title="Previous",
                        click=self.server.controller.decrement_frame,
                        readonly=True,
                    )

                    vuetify.VSpacer()

                    vuetify.VCheckbox(
                        value=False,
                        true_icon="mdi-skip-next-circle",
                        false_icon="mdi-skip-next-circle",
                        hide_details=True,
                        title="Next",
                        click=self.server.controller.increment_frame,
                        readonly=True,
                    )

                    vuetify.VSpacer()

                    vuetify.VCheckbox(
                        v_model=("playing", False),
                        true_icon="mdi-pause-circle",
                        false_icon="mdi-play-circle",
                        title="Play/Pause",
                        hide_details=True,
                    )

                    vuetify.VSpacer()

                    vuetify.VCheckbox(
                        v_model=("incrementing", True),
                        true_icon="mdi-movie-open-outline",
                        false_icon="mdi-movie-open-off-outline",
                        hide_details=True,
                        title="Incrementing",
                    )

                    vuetify.VSpacer()

                    vuetify.VCheckbox(
                        v_model=("rotating", False),
                        true_icon="mdi-autorenew",
                        false_icon="mdi-autorenew-off",
                        hide_details=True,
                        title="Rotating",
                    )

                vuetify.VSlider(
                    v_model=("frame", self.scene.current_frame),
                    hint="Phase",
                    persistent_hint=True,
                    min=0,
                    max=self.scene.nframes - 1,
                    step=1,
                    hide_details=False,
                    dense=True,
                    style="max-width: 300px",
                    ticks=True,
                    thumb_label=True,
                )

                vuetify.VSlider(
                    v_model=("bpm", 60),
                    hint="Speed",
                    persistent_hint=True,
                    min=40,
                    max=1024,
                    step=1,
                    hide_details=False,
                    dense=True,
                    style="max-width: 300px",
                    ticks=True,
                    thumb_label=True,
                )

                vuetify.VSlider(
                    v_model=("bpr", 3),
                    hint="Cycles/Rotation",
                    persistent_hint=True,
                    min=1,
                    max=360,
                    step=1,
                    hide_details=False,
                    dense=True,
                    style="max-width: 300px",
                    ticks=True,
                    thumb_label=True,
                )

                with vuetify.VRow(justify="center", classes="my-3"):
                    vuetify.VBtn(
                        "Capture Cine",
                        small=True,
                        dense=True,
                        outlined=True,
                        color="info",
                        block=True,
                        click=self.server.controller.screenshot,
                        title=f"Capture cine to {self.scene.screenshot_directory}",
                        prepend_icon="mdi-video",
                    )

                vuetify.VListSubheader("Appearance and Visibility")

                if self.scene.meshes:
                    vuetify.VListSubheader("Meshes", classes="text-caption pl-4")
                    for i, m in enumerate(self.scene.meshes):
                        vuetify.VCheckbox(
                            v_model=f"mesh_visibility_{m.label}",
                            on_icon="mdi-eye",
                            off_icon="mdi-eye-off",
                            classes="mx-1",
                            hide_details=True,
                            dense=True,
                            label=m.label,
                        )
                        if m.clipping_enabled:
                            vuetify.VCheckbox(
                                v_model=(
                                    f"mesh_clipping_{m.label}",
                                    m.clipping_enabled,
                                ),
                                on_icon="mdi-content-cut",
                                off_icon="mdi-content-cut",
                                classes="mx-1 ml-4",
                                hide_details=True,
                                dense=True,
                                label="Clip",
                            )

                            # Get initial mesh bounds for sliders
                            if m.actors:
                                bounds = m.combined_bounds
                                with vuetify.VExpansionPanels(
                                    v_model=f"clip_panel_{m.label}",
                                    multiple=True,
                                    flat=True,
                                    classes="ml-4",
                                    dense=True,
                                    style="max-width: 270px;",
                                ):
                                    with vuetify.VExpansionPanel():
                                        vuetify.VExpansionPanelTitle("Clip Bounds")
                                        with vuetify.VExpansionPanelText():
                                            # X bounds
                                            vuetify.VRangeSlider(
                                                v_model=(
                                                    f"clip_x_{m.label}",
                                                    [bounds[0], bounds[1]],
                                                ),
                                                label="X Range",
                                                min=bounds[0],
                                                max=bounds[1],
                                                step=(bounds[1] - bounds[0]) / 100,
                                                hide_details=True,
                                                dense=True,
                                                thumb_label=False,
                                            )
                                            # Y bounds
                                            vuetify.VRangeSlider(
                                                v_model=(
                                                    f"clip_y_{m.label}",
                                                    [bounds[2], bounds[3]],
                                                ),
                                                label="Y Range",
                                                min=bounds[2],
                                                max=bounds[3],
                                                step=(bounds[3] - bounds[2]) / 100,
                                                hide_details=True,
                                                dense=True,
                                                thumb_label=False,
                                            )
                                            # Z bounds
                                            vuetify.VRangeSlider(
                                                v_model=(
                                                    f"clip_z_{m.label}",
                                                    [bounds[4], bounds[5]],
                                                ),
                                                label="Z Range",
                                                min=bounds[4],
                                                max=bounds[5],
                                                step=(bounds[5] - bounds[4]) / 100,
                                                hide_details=True,
                                                dense=True,
                                                thumb_label=False,
                                            )

                if self.scene.volumes:
                    vuetify.VListSubheader("Volumes", classes="text-caption pl-4")
                    for i, v in enumerate(self.scene.volumes):
                        vuetify.VCheckbox(
                            v_model=f"volume_visibility_{v.label}",
                            on_icon="mdi-eye",
                            off_icon="mdi-eye-off",
                            classes="mx-1",
                            hide_details=True,
                            dense=True,
                            label=v.label,
                        )

                        # Preset selection in collapsible panel
                        available_presets = list_volume_property_presets()

                        with vuetify.VExpansionPanels(
                            v_model=f"preset_panel_{v.label}",
                            flat=True,
                            classes="ml-4",
                            dense=True,
                            style="max-width: 270px;",
                        ):
                            with vuetify.VExpansionPanel():
                                vuetify.VExpansionPanelTitle("Transfer Function")
                                with vuetify.VExpansionPanelText():
                                    with vuetify.VRadioGroup(
                                        v_model=f"volume_preset_{v.label}",
                                        dense=True,
                                    ):
                                        for (
                                            preset_key,
                                            preset_desc,
                                        ) in available_presets.items():
                                            vuetify.VRadio(
                                                label=preset_desc, value=preset_key
                                            )

                        # Add clipping controls for volumes
                        if v.clipping_enabled:
                            vuetify.VCheckbox(
                                v_model=(
                                    f"volume_clipping_{v.label}",
                                    v.clipping_enabled,
                                ),
                                on_icon="mdi-cube-outline",
                                off_icon="mdi-cube-off-outline",
                                classes="mx-1 ml-4",
                                hide_details=True,
                                dense=True,
                                label=f"{v.label} Clipping",
                            )

                            # Get initial volume bounds for sliders
                            if v.actors:
                                bounds = v.combined_bounds
                                with vuetify.VExpansionPanels(
                                    v_model=f"clip_panel_{v.label}",
                                    multiple=True,
                                    flat=True,
                                    classes="ml-4",
                                    dense=True,
                                    style="max-width: 270px;",
                                ):
                                    with vuetify.VExpansionPanel():
                                        vuetify.VExpansionPanelTitle("Clip Bounds")
                                        with vuetify.VExpansionPanelText():
                                            # X bounds
                                            vuetify.VRangeSlider(
                                                v_model=(
                                                    f"clip_x_{v.label}",
                                                    [bounds[0], bounds[1]],
                                                ),
                                                label="X Range",
                                                min=bounds[0],
                                                max=bounds[1],
                                                step=(bounds[1] - bounds[0]) / 100,
                                                hide_details=True,
                                                dense=True,
                                                thumb_label=False,
                                            )
                                            # Y bounds
                                            vuetify.VRangeSlider(
                                                v_model=(
                                                    f"clip_y_{v.label}",
                                                    [bounds[2], bounds[3]],
                                                ),
                                                label="Y Range",
                                                min=bounds[2],
                                                max=bounds[3],
                                                step=(bounds[3] - bounds[2]) / 100,
                                                hide_details=True,
                                                dense=True,
                                                thumb_label=False,
                                            )
                                            # Z bounds
                                            vuetify.VRangeSlider(
                                                v_model=(
                                                    f"clip_z_{v.label}",
                                                    [bounds[4], bounds[5]],
                                                ),
                                                label="Z Range",
                                                min=bounds[4],
                                                max=bounds[5],
                                                step=(bounds[5] - bounds[4]) / 100,
                                                hide_details=True,
                                                dense=True,
                                                thumb_label=False,
                                            )

                if self.scene.segmentations:
                    vuetify.VListSubheader("Segmentations", classes="text-caption pl-4")
                    for i, s in enumerate(self.scene.segmentations):
                        vuetify.VCheckbox(
                            v_model=f"segmentation_visibility_{s.label}",
                            on_icon="mdi-eye",
                            off_icon="mdi-eye-off",
                            classes="mx-1",
                            hide_details=True,
                            dense=True,
                            label=s.label,
                        )

                        if s.clipping_enabled:
                            vuetify.VCheckbox(
                                v_model=(
                                    f"segmentation_clipping_{s.label}",
                                    s.clipping_enabled,
                                ),
                                on_icon="mdi-content-cut",
                                off_icon="mdi-content-cut",
                                classes="mx-1 ml-4",
                                hide_details=True,
                                dense=True,
                                label="Clip",
                            )

                            # Get initial segmentation bounds for sliders
                            if s.actors:
                                bounds = s.combined_bounds
                                with vuetify.VExpansionPanels(
                                    v_model=f"clip_panel_{s.label}",
                                    multiple=True,
                                    flat=True,
                                    classes="ml-4",
                                    dense=True,
                                    style="max-width: 270px;",
                                ):
                                    with vuetify.VExpansionPanel():
                                        vuetify.VExpansionPanelTitle("Clip Bounds")
                                        with vuetify.VExpansionPanelText():
                                            # X bounds
                                            vuetify.VRangeSlider(
                                                v_model=(
                                                    f"clip_x_{s.label}",
                                                    [bounds[0], bounds[1]],
                                                ),
                                                label="X Range",
                                                min=bounds[0],
                                                max=bounds[1],
                                                step=(bounds[1] - bounds[0]) / 100,
                                                hide_details=True,
                                                dense=True,
                                                thumb_label=False,
                                            )
                                            # Y bounds
                                            vuetify.VRangeSlider(
                                                v_model=(
                                                    f"clip_y_{s.label}",
                                                    [bounds[2], bounds[3]],
                                                ),
                                                label="Y Range",
                                                min=bounds[2],
                                                max=bounds[3],
                                                step=(bounds[3] - bounds[2]) / 100,
                                                hide_details=True,
                                                dense=True,
                                                thumb_label=False,
                                            )
                                            # Z bounds
                                            vuetify.VRangeSlider(
                                                v_model=(
                                                    f"clip_z_{s.label}",
                                                    [bounds[4], bounds[5]],
                                                ),
                                                label="Z Range",
                                                min=bounds[4],
                                                max=bounds[5],
                                                step=(bounds[5] - bounds[4]) / 100,
                                                hide_details=True,
                                                dense=True,
                                                thumb_label=False,
                                            )

    def _update_all_mpr_views(self, **kwargs):
        """Update all MPR views."""
        if hasattr(self.server.controller, "axial_update"):
            self.server.controller.axial_update()
        if hasattr(self.server.controller, "coronal_update"):
            self.server.controller.coronal_update()
        if hasattr(self.server.controller, "sagittal_update"):
            self.server.controller.sagittal_update()
        if hasattr(self.server.controller, "volume_update"):
            self.server.controller.volume_update()
