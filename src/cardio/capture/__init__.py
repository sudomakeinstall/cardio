"""Writing a capture to disk, in whichever format was asked for."""

from .base import CaptureWriter, Context, Frame, Location, Plane, image_to_array
from .formats import CaptureFormat, wants_alpha, wants_plane, writer_for
from .frames import WindowFrames

__all__ = [
    "CaptureFormat",
    "CaptureWriter",
    "Context",
    "Frame",
    "Location",
    "Plane",
    "WindowFrames",
    "image_to_array",
    "wants_alpha",
    "wants_plane",
    "writer_for",
]
