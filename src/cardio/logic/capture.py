"""Writing screenshots and rotation files to disk."""

# System
import asyncio
import datetime as dt

# Third Party
from trame.app import asynchronous

# Internal
from ..screenshot import Screenshot
from .base import Controller

VIEWPORTS = ("vr", "axial", "coronal", "sagittal")


class CaptureController(Controller):
    """Cine capture and rotation serialisation."""

    def register(self):
        for viewport in VIEWPORTS:
            self.server.state[f"screenshot_viewport_{viewport}"] = (
                viewport in self.scene.screenshot_viewports
            )

        self.server.controller.screenshot = self.screenshot
        self.server.controller.save_rotation_angles = self.save_rotation_angles

    @asynchronous.task
    async def screenshot(self):
        dr = dt.datetime.now().strftime(self.scene.timestamp_format)
        dr = self.scene.screenshot_directory / dr

        mpr_enabled = self.server.state.mpr_enabled
        selected = {
            name
            for name in ("vr", "axial", "coronal", "sagittal")
            if getattr(self.server.state, f"screenshot_viewport_{name}", True)
        }
        render_windows = {}
        if "vr" in selected:
            render_windows["vr"] = self.scene.renderWindow
        if mpr_enabled:
            for name in ("axial", "coronal", "sagittal"):
                if name in selected:
                    render_windows[name] = self.scene.mpr_views[name]

        for folder in render_windows:
            (dr / folder).mkdir(parents=True, exist_ok=True)

        def _save_all(i):
            for folder, rw in render_windows.items():
                Screenshot(rw).save(str(dr / folder / f"{i}.png"))

        if not (self.server.state.incrementing or self.server.state.rotating):
            _save_all(0)
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
                        self.app.playback.increment_frame()
                    self.server.controller.view_update()
                    _save_all(i)
                    await asyncio.sleep(
                        1 / self.server.state.bpm * 60 / self.scene.nframes
                    )

    @asynchronous.task
    async def save_rotation_angles(self):
        """Save current rotation angles to TOML file."""
        timestamp = dt.datetime.now()
        timestamp_str = timestamp.strftime(self.scene.timestamp_format)
        active_volume_label = self.server.state.active_volume_label

        if not active_volume_label:
            print("Warning: No active volume selected")
            return

        save_dir = self.scene.rotations_directory / active_volume_label
        save_dir.mkdir(parents=True, exist_ok=True)

        rotation_seq = self.app.rotations.rotation_sequence()

        # Update only timestamp and volume_label (rest already in metadata)
        rotation_seq.metadata.timestamp = timestamp.isoformat()
        rotation_seq.metadata.volume_label = active_volume_label

        output_path = save_dir / f"{timestamp_str}.toml"
        rotation_seq.to_file(output_path)

        with self.server.state as state:
            state.rotations_saved_at = timestamp.strftime("%H:%M:%S")
            state.rotations_stale = False
