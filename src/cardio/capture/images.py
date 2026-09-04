"""Writers that record what the viewport looked like."""

# System
import pathlib as pl
import tempfile

# Third Party
import imageio_ffmpeg as iio
import numpy as np
import PIL.Image
import vtk

# Internal
from .base import CaptureWriter, Context, Frame


class StillWriter(CaptureWriter):
    """One image file per frame, in a directory named for the viewport.

    Written through VTK rather than from the numpy view, so the bytes are the
    ones VTK has always produced for these formats.
    """

    suffix: str = ""

    def __init__(self, context: Context):
        self.directory = context.directory / context.viewport
        self.directory.mkdir(parents=True, exist_ok=True)
        self.writer = self._make_writer()

    def _make_writer(self) -> vtk.vtkImageWriter:
        raise NotImplementedError

    def add(self, index: int, frame: Frame):
        self.writer.SetFileName(str(self.directory / f"{index}{self.suffix}"))
        self.writer.SetInputData(frame.image)
        self.writer.Write()


class PngWriter(StillWriter):
    suffix = ".png"

    def _make_writer(self):
        return vtk.vtkPNGWriter()


class JpegWriter(StillWriter):
    suffix = ".jpg"

    def _make_writer(self):
        return vtk.vtkJPEGWriter()


def _fit(rgb, size: tuple[int, int] | None):
    """One frame's RGB, sized to match the frames already written.

    An animation is one stream of one size, so a window resized mid-capture is
    cropped or padded rather than ending the stream.  The first frame sets the
    size; ``size`` is None for it.
    """
    rgb = rgb[:, :, :3]
    if size is None:
        rows, columns = rgb.shape[:2]
        return rgb, (rows, columns)
    if rgb.shape[:2] == size:
        return rgb, size

    rows, columns = size
    fitted = np.zeros((rows, columns, 3), dtype=rgb.dtype)
    kept = rgb[:rows, :columns]
    fitted[: kept.shape[0], : kept.shape[1]] = kept
    return fitted, size


class AnimationWriter(CaptureWriter):
    """The whole capture as one file, played at the rate it was taken."""

    suffix: str = ""

    def __init__(self, context: Context):
        self.path = context.directory / f"{context.viewport}{self.suffix}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_duration = context.frame_duration
        self.size: tuple[int, int] | None = None

    def fit(self, frame: Frame):
        rgb, self.size = _fit(frame.rgb, self.size)
        return rgb


class GifWriter(AnimationWriter):
    """Every frame in one animation, spilled to disk on the way.

    Pillow assembles a GIF from all of its frames at once, so holding them
    would cost the whole capture at once: a rotating cine runs to
    ``nframes * bpr`` frames, and bpr reaches 360.  Writing each one out and
    reading them back as the save consumes them keeps the cost at one frame.
    """

    suffix = ".gif"

    def __init__(self, context: Context):
        super().__init__(context)
        self._spill = tempfile.TemporaryDirectory(
            prefix="cardio-gif-", ignore_cleanup_errors=True
        )
        self._frames: list[pl.Path] = []

    def add(self, index: int, frame: Frame):
        path = pl.Path(self._spill.name) / f"{index:06d}.png"
        PIL.Image.fromarray(self.fit(frame)).save(path)
        self._frames.append(path)

    def close(self):
        try:
            if not self._frames:
                return
            with PIL.Image.open(self._frames[0]) as first:
                first.save(
                    self.path,
                    save_all=True,
                    append_images=_reread(self._frames[1:]),
                    duration=round(self.frame_duration * 1000),
                    loop=0,
                )
        finally:
            self._frames.clear()
            self._spill.cleanup()


def _reread(paths):
    """The spilled frames, one at a time, as the save asks for them."""
    for path in paths:
        with PIL.Image.open(path) as image:
            yield image.copy()


class Mp4Writer(AnimationWriter):
    """H.264, encoded a frame at a time through a piped ffmpeg.

    yuv420p needs both dimensions even, so an odd row or column is dropped --
    a pixel of the border, rather than a rescale of the whole picture.
    """

    suffix = ".mp4"

    def __init__(self, context: Context):
        super().__init__(context)
        self._pipe = None

    def add(self, index: int, frame: Frame):
        rgb = self.fit(frame)
        rows, columns = rgb.shape[:2]
        rgb = rgb[: rows - rows % 2, : columns - columns % 2]

        if self._pipe is None:
            self._pipe = iio.write_frames(
                str(self.path),
                (rgb.shape[1], rgb.shape[0]),
                fps=1.0 / self.frame_duration if self.frame_duration > 0 else 1.0,
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                macro_block_size=1,
            )
            self._pipe.send(None)

        self._pipe.send(rgb.tobytes())

    def close(self):
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None
