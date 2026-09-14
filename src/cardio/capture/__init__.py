"""Writing a capture to disk, in whichever format was asked for."""

from .base import (
    CaptureWriter,
    Context,
    Frame,
    Identity,
    Location,
    Plane,
    image_to_array,
)
from .encoding import TransferSyntax
from .equipment import Equipment
from .formats import (
    CaptureFormat,
    wants_alpha,
    wants_plane,
    writer_for,
    writes_series,
)
from .frames import WindowFrames
from .series import Series, SeriesTags, describe, repeated_numbers

__all__ = [
    "CaptureFormat",
    "CaptureWriter",
    "Context",
    "Equipment",
    "Frame",
    "Identity",
    "Location",
    "Plane",
    "Series",
    "SeriesTags",
    "TransferSyntax",
    "WindowFrames",
    "describe",
    "image_to_array",
    "repeated_numbers",
    "wants_alpha",
    "wants_plane",
    "writer_for",
    "writes_series",
]
