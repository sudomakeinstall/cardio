from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify

from . import Scene
from .transfer_functions import list_available_presets


class UI:
    def __init__(self, server, scene: Scene):
        self.server = server
        self.scene: Scene = scene

        self.setup()

    def setup(self):
        self.server.state.trame__title = self.scene.project_name

        with SinglePageWithDrawerLayout(self.server) as layout:
            layout.icon.click = self.server.controller.view_reset_camera
            layout.title.set_text(self.scene.project_name)

            with layout.toolbar as toolbar:
                toolbar.dense = True

                vuetify.VSpacer()

                #        vuetify.VCheckbox(
                #            v_model=("viewMode", "local"),
                #            on_icon="mdi-lan-disconnect",
                #            off_icon="mdi-lan-connect",
                #            true_value="local",
                #            false_value="remote",
                #            classes="mx-1",
                #            hide_details=True,
                #            dense=True,
                #            )

                vuetify.VCheckbox(
                    v_model=("dark_mode", False),
                    on_icon="mdi-lightbulb-outline",
                    off_icon="mdi-lightbulb-off-outline",
                    classes="mx-1",
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    change="$vuetify.theme.dark = $event",
                )

                # NOTE: Reset button should be VBtn, but using VCheckbox for consistent sizing/spacing
                vuetify.VCheckbox(
                    value=False,
                    on_icon="mdi-undo-variant",
                    off_icon="mdi-undo-variant",
                    hide_details=True,
                    title="Reset All",
                    click=self.server.controller.reset_all,
                    readonly=True,
                )

                vuetify.VProgressLinear(
                    indeterminate=True,
                    absolute=True,
                    bottom=True,
                    active=("trame__busy",),
                )

            with layout.content:
                with vuetify.VContainer(
                    fluid=True,
                    classes="pa-0 fill-height",
                ):
                    view = vtk_widgets.VtkRemoteView(
                        self.scene.renderWindow, interactive_ratio=1
                    )
                    # view = vtk_widgets.VtkLocalView(renderWindow)
                    # view = vtk_widgets.VtkRemoteLocalView(
                    #    scene.renderWindow,
                    #    namespace="view",
                    #    mode="local",
                    #    interactive_ratio=1,
                    # )
                    self.server.controller.view_update = view.update
                    self.server.controller.view_reset_camera = view.reset_camera
                    self.server.controller.on_server_ready.add(view.update)

            with layout.drawer:
                vuetify.VSubheader("Playback Controls")

                with vuetify.VToolbar(dense=True, flat=True):
                    # NOTE: Previous/Next controls should be VBtn components, but we use
                    # VCheckbox for consistent sizing/spacing with the other controls.
                    # This may be easier to fix in Vuetify 3.
                    vuetify.VCheckbox(
                        value=False,
                        on_icon="mdi-skip-previous-circle",
                        off_icon="mdi-skip-previous-circle",
                        hide_details=True,
                        title="Previous",
                        click=self.server.controller.decrement_frame,
                        readonly=True,
                    )

                    vuetify.VSpacer()

                    vuetify.VCheckbox(
                        value=False,
                        on_icon="mdi-skip-next-circle",
                        off_icon="mdi-skip-next-circle",
                        hide_details=True,
                        title="Next",
                        click=self.server.controller.increment_frame,
                        readonly=True,
                    )

                    vuetify.VSpacer()

                    vuetify.VCheckbox(
                        v_model=("playing", False),
                        on_icon="mdi-pause-circle",
                        off_icon="mdi-play-circle",
                        title="Play/Pause",
                        hide_details=True,
                    )

                    vuetify.VSpacer()

                    vuetify.VCheckbox(
                        v_model=("incrementing", True),
                        on_icon="mdi-movie-open-outline",
                        off_icon="mdi-movie-open-off-outline",
                        hide_details=True,
                        title="Incrementing",
                    )

                    vuetify.VSpacer()

                    vuetify.VCheckbox(
                        v_model=("rotating", False),
                        on_icon="mdi-autorenew",
                        off_icon="mdi-autorenew-off",
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
                        f"Capture Cine",
                        small=True,
                        click=self.server.controller.screenshot,
                        title=f"Capture cine to {self.scene.screenshot_directory}",
                    )

                vuetify.VSubheader("Appearance and Visibility")

                if self.scene.meshes:
                    vuetify.VSubheader("Meshes", classes="text-caption pl-4")
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

                if self.scene.volumes:
                    vuetify.VSubheader("Volumes", classes="text-caption pl-4")
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
                        available_presets = list_available_presets()
                        current_preset = self.server.state[f"volume_preset_{v.label}"]
                        current_desc = available_presets.get(
                            current_preset, current_preset
                        )

                        with vuetify.VExpansionPanels(
                            v_model=f"preset_panel_{v.label}",
                            flat=True,
                            classes="ml-4",
                            dense=True,
                            style="max-width: 270px;",
                        ):
                            with vuetify.VExpansionPanel():
                                vuetify.VExpansionPanelHeader("Transfer Function")
                                with vuetify.VExpansionPanelContent():
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
                                bounds = v.actors[0].GetBounds()
                                with vuetify.VExpansionPanels(
                                    v_model=f"clip_panel_{v.label}",
                                    multiple=True,
                                    flat=True,
                                    classes="ml-4",
                                    dense=True,
                                    style="max-width: 270px;",
                                ):
                                    with vuetify.VExpansionPanel():
                                        vuetify.VExpansionPanelHeader("Clip Bounds")
                                        with vuetify.VExpansionPanelContent():
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
