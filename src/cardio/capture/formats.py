"""The output formats a capture may be written in."""

# System
import enum

# Internal
from .base import CaptureWriter, Context
from .dicom import SecondaryCaptureWriter, SliceWriter
from .images import GifWriter, JpegWriter, Mp4Writer, PngWriter


class CaptureFormat(enum.StrEnum):
    """What a capture is written as.

    Only formats that are wired end to end appear here: the drawer's dropdown
    is built from this enum, so an unimplemented member would be offered.
    """

    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    MP4 = "mp4"
    DICOM_RENDERED = "dicom-rendered"
    DICOM_DATA = "dicom-data"


WRITERS = {
    CaptureFormat.PNG: PngWriter,
    CaptureFormat.JPEG: JpegWriter,
    CaptureFormat.GIF: GifWriter,
    CaptureFormat.MP4: Mp4Writer,
    CaptureFormat.DICOM_RENDERED: SecondaryCaptureWriter,
    CaptureFormat.DICOM_DATA: SliceWriter,
}


def wants_alpha(fmt: CaptureFormat) -> bool:
    """Whether the encoder carries an alpha channel.

    The rest are captured without one rather than having it silently dropped.
    """
    return CaptureFormat(fmt) is CaptureFormat.PNG


def wants_plane(fmt: CaptureFormat) -> bool:
    """Whether the format writes the pixels behind the viewport, not a picture."""
    return CaptureFormat(fmt) is CaptureFormat.DICOM_DATA


def writer_for(fmt: CaptureFormat, context: Context) -> CaptureWriter:
    """The writer that records one viewport in ``fmt``.

    A viewport with no image plane falls back to recording what it looked
    like, which for the volume render is the only thing there is to record.
    """
    fmt = CaptureFormat(fmt)
    if wants_plane(fmt) and not context.has_plane:
        fmt = CaptureFormat.DICOM_RENDERED
    return WRITERS[fmt](context)
