from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify

from . import Scene


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
                    v_model="$vuetify.theme.dark",
                    on_icon="mdi-lightbulb-outline",
                    off_icon="mdi-lightbulb-off-outline",
                    classes="mx-1",
                    hide_details=True,
                    dense=True,
                    outlined=True,
                )

                with vuetify.VBtn(icon=True, click=self.server.controller.reset_all):
                    vuetify.VIcon("mdi-undo-variant")

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

                with vuetify.VBtn(
                    icon=True,
                    click=self.server.controller.decrement_frame,
                    title="Previous",
                ):
                    vuetify.VIcon("mdi-skip-previous-circle")

                with vuetify.VBtn(
                    icon=True,
                    click=self.server.controller.increment_frame,
                    title="Next",
                ):
                    vuetify.VIcon("mdi-skip-next-circle")

                vuetify.VCheckbox(
                    v_model=("playing", False),
                    on_icon="mdi-pause-circle",
                    off_icon="mdi-play-circle",
                    classes="mx-1",
                    hide_details=True,
                    dense=True,
                    title="Play/Pause",
                )

                vuetify.VCheckbox(
                    v_model=("incrementing", True),
                    on_icon="mdi-movie-open-outline",
                    off_icon="mdi-movie-open-off-outline",
                    classes="mx-1",
                    hide_details=True,
                    dense=True,
                    label="Incrementing",
                    title="Incrementing",
                )

                vuetify.VCheckbox(
                    v_model=("rotating", False),
                    on_icon="mdi-autorenew",
                    off_icon="mdi-autorenew-off",
                    classes="mx-1",
                    hide_details=True,
                    dense=True,
                    label="Rotating",
                    title="Rotating",
                )

                with vuetify.VBtn(
                    icon=True,
                    click=self.server.controller.screenshot,
                    title="Save Screenshot",
                    label="Take Screenshot",
                ):
                    vuetify.VIcon("mdi-image")

                vuetify.VSlider(
                    v_model=("frame", self.scene.current_frame),
                    label="Frame",
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
                    label="BPM",
                    hint="Beats Per Minute",
                    persistent_hint=True,
                    min=40,
                    max=160,
                    step=1,
                    hide_details=False,
                    dense=True,
                    style="max-width: 300px",
                    ticks=True,
                    thumb_label=True,
                )

                vuetify.VSlider(
                    v_model=("bpr", 3),
                    label="BPR",
                    hint="Beats Per Rotation",
                    persistent_hint=True,
                    min=1,
                    max=5,
                    step=1,
                    hide_details=False,
                    dense=True,
                    style="max-width: 300px",
                    ticks=True,
                    thumb_label=True,
                )

                for i, m in enumerate(self.scene.meshes):
                    vuetify.VCheckbox(
                        v_model=(f"mesh_visibility_{m.label}", m.visible),
                        on_icon="mdi-eye",
                        off_icon="mdi-eye-off",
                        classes="mx-1",
                        hide_details=True,
                        dense=True,
                        label=m.label,
                    )

                for i, v in enumerate(self.scene.volumes):
                    vuetify.VCheckbox(
                        v_model=(f"volume_visibility_{v.label}", v.visible),
                        on_icon="mdi-eye",
                        off_icon="mdi-eye-off",
                        classes="mx-1",
                        hide_details=True,
                        dense=True,
                        label=v.label,
                    )
