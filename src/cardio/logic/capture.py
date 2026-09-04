"""Writing captures and rotation files to disk."""

# System
import asyncio
import datetime as dt

# Third Party
import pydicom as pd
from trame.app import asynchronous

# Internal
from ..capture import (
    CaptureFormat,
    Context,
    WindowFrames,
    wants_alpha,
    wants_plane,
    writer_for,
)
from ..capture.dicom import IDENTITY_TAGS
from ..capture.geometry import plane_from_reslice
from ..capture.mosaic import compose
from ..reslice import VIEW_TRANSFORMS
from ..view import Layout
from .base import Controller

VIEWPORTS = ("vr", "axial", "coronal", "sagittal", "tile")

# The one viewport the capture and the layout call different things.  Every
# other name is shared, so this is the whole of the translation.
VIEWPORT_FOR_LAYOUT = {Layout.VOLUME: "vr"}

MPR_VIEWPORTS = ("axial", "coronal", "sagittal")


def viewport_of(layout: Layout) -> str:
    """What a capture calls the view ``layout`` shows."""
    return VIEWPORT_FOR_LAYOUT.get(layout, layout.value)


def _now() -> dt.datetime:
    """Local wall-clock time, carrying the offset that makes it unambiguous.

    Local rather than UTC because these name folders a person goes looking in;
    aware rather than naive so a saved record says which local time it meant.
    """
    return dt.datetime.now().astimezone()


def written_files(directory, viewport: str) -> list:
    """What one viewport actually left on disk.

    A viewport is a folder of stills or of DICOM instances, or a single
    animation named after it; asking the filesystem covers both without the
    writers having to agree on how to count themselves.
    """
    folder = directory / viewport
    if folder.is_dir():
        return sorted(path for path in folder.iterdir() if path.is_file())
    return sorted(directory.glob(f"{viewport}.*"))


def summary_of(written: list[str], directory) -> str:
    """The one line the drawer shows about a finished capture."""
    if not written:
        return "Capture wrote nothing"
    return f"Captured {', '.join(written)} to {directory.name}"


class CaptureController(Controller):
    """Cine capture and rotation serialisation."""

    def register(self):
        state = self.server.state

        for viewport in VIEWPORTS:
            state[f"screenshot_viewport_{viewport}"] = (
                viewport in self.scene.screenshot_viewports
            )

        state.capture_format = self.scene.capture_format.value
        state.capture_available = sorted(
            viewport_of(shown) for shown in self.scene.view.layout.on_screen
        )
        state.capture_running = False
        state.capture_progress = 0
        state.capture_saved_at = None
        state.capture_summary = ""
        state.capture_ok = True

        # The layout decides which viewports can be captured, and the drawer
        # greys out the rest; publishing it is what keeps that rule in one place
        # rather than restated as a vue expression.
        state.change("maximized_view")(self.sync_available)

        self.server.controller.screenshot = self.screenshot
        self.server.controller.save_rotation_angles = self.save_rotation_angles

    @property
    def capture_format(self) -> CaptureFormat:
        return CaptureFormat(
            getattr(self.server.state, "capture_format", self.scene.capture_format)
        )

    @property
    def available(self) -> frozenset[str]:
        """The viewports currently on screen, which are the capturable ones."""
        layout = Layout.from_state(getattr(self.server.state, "maximized_view", None))
        return frozenset(viewport_of(shown) for shown in layout.on_screen)

    def sync_available(self, **kwargs):
        self.server.state.capture_available = sorted(self.available)

    def selected_windows(self) -> dict:
        """The render window behind each ticked viewport that is on screen.

        A viewport the layout is not drawing holds whichever frame was last
        rendered into it, and saving that is worse than leaving it out: it is
        indistinguishable from a capture that worked.
        """
        windows = {
            "vr": self.scene.renderWindow,
            "tile": (
                self.scene.tile_views.window
                if self.scene.tile_views is not None
                else None
            ),
        }
        if self.scene.mpr_views is not None:
            for name in MPR_VIEWPORTS:
                windows[name] = self.scene.mpr_views[name]

        available = self.available
        return {
            name: windows[name]
            for name in VIEWPORTS
            if name in available
            and windows.get(name) is not None
            and getattr(self.server.state, f"screenshot_viewport_{name}", True)
        }

    def frame_duration(self) -> float:
        """Seconds one frame is shown for, at the configured heart rate."""
        return 1 / self.server.state.bpm * 60 / self.scene.nframes

    def identity(self) -> dict[str, str]:
        """The patient and study a capture belongs to.

        Taken from the active volume's own header when it was read from DICOM,
        so a derived series lands in the study it was derived from.  A volume
        read from a file carries none of this, so the capture stands alone
        under a study of its own.
        """
        volume = self._active_volume()
        header = {}
        if volume is not None and volume.source is not None:
            header = volume.source.header

        fields = {tag: header[tag] for tag in IDENTITY_TAGS if header.get(tag)}
        fields.setdefault("PatientName", "Anonymous")
        fields.setdefault("PatientID", "CARDIO")
        fields.setdefault("StudyInstanceUID", pd.uid.generate_uid())
        return fields

    def _mpr_plane(self, viewport: str, volume, frame: int):
        """One MPR view's cut, as the view itself is posed for that frame."""
        reslices = volume.get_mpr_actors_for_frame(frame)
        return plane_from_reslice(reslices[viewport]["reslice"])

    def _mosaic_plane(self, viewport: str, volume, frame: int):
        """The tile grid's cuts, composed into one image."""
        views = self.scene.tile_views
        poses = self.app.tiles.tile_poses(frame)
        if views is None or poses is None:
            return None
        return compose(
            volume.mpr_image_data(frame),
            poses,
            VIEW_TRANSFORMS["axial"],
            views.rows,
            views.cols,
        )

    @property
    def plane_sources(self) -> dict:
        """Where each viewport's cut comes from.

        The keys are what "this viewport has a plane" means, so the set cannot
        drift from the code that computes one: a viewport added here gains its
        data capture, and one left out is offered none.
        """
        return {
            **{name: self._mpr_plane for name in MPR_VIEWPORTS},
            "tile": self._mosaic_plane,
        }

    def plane_for(self, viewport: str, frame: int):
        """The cut behind a viewport, or None when it is showing nothing."""
        source = self.plane_sources.get(viewport)
        volume = self._active_volume()
        if source is None or volume is None:
            return None
        return source(viewport, volume, frame)

    def _context(self, directory, viewport: str, number: int, identity, reference):
        state = self.server.state
        return Context(
            directory=directory,
            viewport=viewport,
            frame_duration=self.frame_duration(),
            window=getattr(state, "mpr_window", self.scene.mpr_window),
            level=getattr(state, "mpr_level", self.scene.mpr_level),
            identity=identity,
            series_number=number,
            frame_of_reference=reference,
            has_plane=viewport in self.plane_sources,
        )

    def report(self, summary: str, ok: bool):
        """Say what a capture did, in the shape the rotations save reports."""
        with self.server.state as state:
            state.capture_saved_at = _now().strftime("%H:%M:%S")
            state.capture_summary = summary
            state.capture_ok = ok

    @asynchronous.task
    async def screenshot(self):
        fmt = self.capture_format
        directory = self.scene.screenshot_directory / _now().strftime(
            self.scene.timestamp_format
        )

        windows = self.selected_windows()
        if not windows:
            # Before any writer or any frame: the loop advances playback and
            # turns the camera, which is a strange thing to watch happen for a
            # capture that was never going to write anything.
            self.report("Nothing to capture: no ticked viewport is on screen", False)
            return

        identity = self.identity()
        reference = pd.uid.generate_uid()
        sources = {
            name: WindowFrames(window, alpha=wants_alpha(fmt))
            for name, window in windows.items()
        }
        writers = {
            name: writer_for(
                fmt, self._context(directory, name, number, identity, reference)
            )
            for number, name in enumerate(windows, start=1)
        }
        planes = wants_plane(fmt)

        def _save_all(i, frame):
            for name, writer in writers.items():
                plane = self.plane_for(name, frame) if planes else None
                writer.add(i, sources[name].capture(plane))

        with self.server.state as state:
            state.capture_running = True
            state.capture_progress = 0

        try:
            if not (self.server.state.incrementing or self.server.state.rotating):
                _save_all(0, self._frame)
            else:
                n = self.scene.nframes
                if self.server.state.rotating:
                    n *= self.server.state.bpr
                deg = 360 / (self.scene.nframes * self.server.state.bpr)
                for i in range(n):
                    with self.server.state:
                        if self.server.state.rotating:
                            self.scene.renderer.GetActiveCamera().Azimuth(deg)
                        # The frame the windows are still drawing.  Writing
                        # ``frame`` only reposes the views when the state block
                        # flushes, which is after this pass has captured them,
                        # so the frame taken after the increment is one the
                        # picture does not show and whose cuts may never have
                        # been posed at all.
                        shown = self._frame
                        if self.server.state.incrementing:
                            self.app.playback.increment_frame()
                        self.server.controller.view_update()
                        _save_all(i, shown)
                        self.server.state.capture_progress = round(100 * (i + 1) / n)
                        await asyncio.sleep(self.frame_duration())
        finally:
            for writer in writers.values():
                writer.close()

            # Counted once the writers have closed, and from the files rather
            # than from the calls: an animation is one file however many frames
            # went into it, and a data capture of a viewport with no cut behind
            # it writes nothing at all.  Neither should be reported as saved.
            written = [name for name in windows if written_files(directory, name)]
            self.report(summary_of(written, directory), bool(written))

            with self.server.state as state:
                state.capture_running = False
                state.capture_progress = 0

    @asynchronous.task
    async def save_rotation_angles(self):
        """Save current rotation angles to TOML file."""
        timestamp = _now()
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
