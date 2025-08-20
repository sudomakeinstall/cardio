import asyncio
import datetime as dt
import os

from trame.app import asynchronous

from . import Scene
from .screenshot import Screenshot


class Logic:
    def __init__(self, server, scene: Scene):
        self.server = server
        self.scene = scene

        self.server.state.change("frame")(self.update_frame)
        self.server.state.change("playing")(self.play)
        self.server.state.change("dark_mode")(self.sync_background_color)

        # Initialize visibility state variables
        for m in self.scene.meshes:
            self.server.state[f"mesh_visibility_{m.label}"] = m.visible
        for v in self.scene.volumes:
            self.server.state[f"volume_visibility_{v.label}"] = v.visible

        # Initialize preset state variables
        for v in self.scene.volumes:
            self.server.state[f"volume_preset_{v.label}"] = v.preset_key
        self.server.state.change(
            *[f"mesh_visibility_{m.label}" for m in self.scene.meshes]
        )(self.sync_mesh_visibility)
        self.server.state.change(
            *[f"volume_visibility_{v.label}" for v in self.scene.volumes]
        )(self.sync_volume_visibility)
        self.server.state.change(
            *[f"volume_preset_{v.label}" for v in self.scene.volumes]
        )(self.sync_volume_presets)

        clipping_controls = []
        for v in self.scene.volumes:
            clipping_controls.extend(
                [
                    f"volume_clipping_{v.label}",
                    f"clip_x_{v.label}",
                    f"clip_y_{v.label}",
                    f"clip_z_{v.label}",
                ]
            )
        if clipping_controls:
            self.server.state.change(*clipping_controls)(self.sync_volume_clipping)

        self.server.controller.increment_frame = self.increment_frame
        self.server.controller.decrement_frame = self.decrement_frame
        self.server.controller.screenshot = self.screenshot
        self.server.controller.reset_all = self.reset_all

        # Initialize clipping state variables
        self._initialize_clipping_state()

    def update_frame(self, frame, **kwargs):
        self.scene.hide_all_frames()

        # Show frame with server state visibility
        for mesh in self.scene.meshes:
            visible = self.server.state[f"mesh_visibility_{mesh.label}"]
            if visible:
                mesh.actors[frame % len(mesh.actors)].SetVisibility(True)

        for volume in self.scene.volumes:
            visible = self.server.state[f"volume_visibility_{volume.label}"]
            if visible:
                volume.actors[frame % len(volume.actors)].SetVisibility(True)

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
            visible = self.server.state[f"mesh_visibility_{m.label}"]
            m.actors[self.server.state.frame % len(m.actors)].SetVisibility(visible)
        self.server.controller.view_update()

    def sync_volume_visibility(self, **kwargs):
        for v in self.scene.volumes:
            visible = self.server.state[f"volume_visibility_{v.label}"]
            v.actors[self.server.state.frame % len(v.actors)].SetVisibility(visible)
        self.server.controller.view_update()

    def sync_volume_presets(self, **kwargs):
        """Update volume transfer function presets based on UI selection."""
        from .transfer_functions import load_preset

        for v in self.scene.volumes:
            preset_name = self.server.state[f"volume_preset_{v.label}"]
            preset = load_preset(preset_name)

            # Apply preset to all actors
            for actor in v.actors:
                actor.SetProperty(preset.vtk_property)

        self.server.controller.view_update()

    def sync_volume_clipping(self, **kwargs):
        """Update volume clipping based on UI controls."""
        for v in self.scene.volumes:
            # Toggle clipping on/off
            clipping_enabled = self.server.state[f"volume_clipping_{v.label}"]
            v.toggle_clipping(clipping_enabled)

            # Update clipping bounds if enabled
            if clipping_enabled:
                x_range = self.server.state[f"clip_x_{v.label}"]
                y_range = self.server.state[f"clip_y_{v.label}"]
                z_range = self.server.state[f"clip_z_{v.label}"]

                if x_range and y_range and z_range:
                    bounds = [
                        x_range[0],
                        x_range[1],
                        y_range[0],
                        y_range[1],
                        z_range[0],
                        z_range[1],
                    ]
                    v.update_clipping_bounds(bounds)

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
            ss = Screenshot(self.scene.renderWindow)
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
                    ss = Screenshot(self.scene.renderWindow)
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

    def sync_background_color(self, dark_mode, **kwargs):
        """Sync VTK renderer background with dark mode."""
        if dark_mode:
            # Dark mode: use dark background from config
            self.scene.renderer.SetBackground(*self.scene.background_dark)
        else:
            # Light mode: use light background from config
            self.scene.renderer.SetBackground(*self.scene.background_light)
        self.server.controller.view_update()

    def _initialize_clipping_state(self):
        """Initialize clipping state variables for all volumes."""
        for v in self.scene.volumes:
            # Initialize preset selection state
            preset_key = getattr(v, "preset_key", "cardiac")
            setattr(self.server.state, f"volume_preset_{v.label}", preset_key)

            # Initialize preset panel state (collapsed by default)
            setattr(self.server.state, f"preset_panel_{v.label}", [])
            if hasattr(v, "clipping_enabled"):
                # Initialize clipping checkbox state
                setattr(
                    self.server.state, f"volume_clipping_{v.label}", v.clipping_enabled
                )

                # Initialize panel state
                setattr(self.server.state, f"clip_panel_{v.label}", [])

                # Initialize range sliders with volume bounds if available
                if v.actors:
                    bounds = v.actors[0].GetBounds()
                    setattr(
                        self.server.state, f"clip_x_{v.label}", [bounds[0], bounds[1]]
                    )
                    setattr(
                        self.server.state, f"clip_y_{v.label}", [bounds[2], bounds[3]]
                    )
                    setattr(
                        self.server.state, f"clip_z_{v.label}", [bounds[4], bounds[5]]
                    )
