import asyncio

from trame.app import asynchronous

from . import Scene


class Logic:
    def __init__(self, server, scene: Scene):
        self.server = server
        self.scene = scene

        self.server.state.change("frame")(self.update_frame)
        self.server.state.change("playing")(self.play)
        self.server.state.change(
            *[f"mesh_visibility_{m.label}" for m in self.scene.meshes]
        )(self.sync_mesh_visibility)
        self.server.state.change(
            *[f"volume_visibility_{v.label}" for v in self.scene.volumes]
        )(self.sync_volume_visibility)

        self.server.controller.increment_frame = self.increment_frame
        self.server.controller.decrement_frame = self.decrement_frame
        self.server.controller.screenshot = self.screenshot
        self.server.controller.reset_all = self.reset_all

    def update_frame(self, frame, **kwargs):
        self.scene.hide_all_frames()
        self.scene.show_frame(frame)
        self.server.controller.view_update()

    @asynchronous.task
    async def play(self, playing, **kwargs):
        if not (self.server.state.incrementing or self.server.state.rotating):
            self.server.state.playing = False
        while self.server.state.playing:
            with self.server.state as state:
                if state.incrementing:
                    state.frame = (state.frame + 1) % self.scene.nframes
                if state.rotating:
                    deg = 360 / (self.scene.nframes * state.bpr)
                    self.scene.renderer.GetActiveCamera().Azimuth(deg)
                self.server.controller.view_update()
            await asyncio.sleep(1 / state.bpm * 60 / self.scene.nframes)

    def sync_mesh_visibility(self, **kwargs):
        for m in self.scene.meshes:
            m.visible = self.server.state[f"mesh_visibility_{m.label}"]
            m.actors[self.server.state.frame].SetVisibility(m.visible)
        self.server.controller.view_update()

    def sync_volume_visibility(self, **kwargs):
        for v in self.scene.volumes:
            v.visible = self.server.state[f"volume_visibility_{v.label}"]
            v.actors[self.server.state.frame].SetVisibility(v.visible)
        self.server.controller.view_update()

    def increment_frame(self):
        if not self.server.state.playing:
            self.server.state.frame = (self.server.state.frame + 1) % self.scene.nframes
            self.server.controller.view_update()

    def decrement_frame(self):
        if not self.server.state.playing:
            self.server.state.frame = (self.server.state.frame - 1) % self.scene.nframes
            self.server.controller.view_update()

    @asynchronous.task
    async def screenshot(self):
        dr = dt.datetime.now().strftime(self.scene.screenshot_subdirectory_format)
        dr = f"{self.scene.screenshot_directory}/{dr}"
        os.makedirs(dr)

        if not (self.server.state.incrementing or self.server.state.rotating):
            ss = cy.Screenshot(self.scene.renderWindow)
            ss.save(f"{dr}/0.png")
        else:
            n = self.scene.nframes
            if self.server.state.rotating:
                n *= self.server.state.bpr
            deg = 360 / (self.scene.nframes * self.server.state.bpr)
            for i in range(n):
                with self.server.state:
                    if self.server.state.rotating:
                        self.scene.renderer.GetActiveCamera().Azimuth(deg)
                    if self.server.state.incrementing:
                        self.increment_frame()
                    self.server.controller.view_update()
                    ss = cy.Screenshot(self.scene.renderWindow)
                    ss.save(f"{dr}/{i}.png")
                    await asyncio.sleep(
                        1 / self.server.state.bpm * 60 / self.scene.nframes
                    )

    def reset_all(self):
        self.server.state.frame = 0
        self.server.state.playing = False
        self.server.state.incrementing = True
        self.server.state.rotating = False
        self.server.state.bpm = 60
        self.server.state.bpr = 5
        self.server.controller.view_update()
