"""What a capture is made of, and what a writer must do with it.

A capture is a sequence: several frames of one viewport, written as one thing.
That is what a DICOM series, a GIF and an MP4 all are, and it is what a folder
of numbered stills has always been -- so writers are opened once per viewport,
fed frames, and closed, rather than handed one file name at a time.
"""

# System
import dataclasses as dc
import pathlib as pl

# Third Party
import numpy as np
import vtk
from vtk.util import numpy_support as vtknp


@dc.dataclass(frozen=True)
class Location:
    """Where a plane of pixels sits in the patient, in LPS.

    ``orientation`` is the DICOM pair of direction cosines: the column
    direction followed by the row direction, both unit length.
    """

    orientation: tuple[float, float, float, float, float, float]
    position: tuple[float, float, float]


@dc.dataclass(frozen=True)
class Plane:
    """Scalars a writer can window, and what is known about where they sit.

    ``location`` is None when the pixels have a scale but no place: the tile
    mosaic composes cuts taken at different poses, so it can be measured but
    not localized, and omitting the location is how that is said.
    """

    scalars: np.ndarray
    pixel_spacing: tuple[float, float]
    thickness: float
    location: Location | None


@dc.dataclass(frozen=True)
class Frame:
    """One viewport at one moment, in whichever form a writer wants it.

    ``image`` is the window capture as VTK produced it, bottom-up, so the VTK
    writers stay byte-exact; ``rgb`` is the same pixels the way every other
    encoder expects them.  ``plane`` is the data behind the picture, absent for
    a volume render, which has a camera rather than an image plane.
    """

    image: vtk.vtkImageData
    plane: Plane | None = None

    @property
    def rgb(self) -> np.ndarray:
        return image_to_array(self.image)


@dc.dataclass(frozen=True)
class Context:
    """What the capture as a whole knows, for the writers that need it.

    ``has_plane`` is what the viewport can offer, not what a given frame did:
    the volume render has a camera rather than an image plane, so no format
    can ask it for one.
    """

    directory: pl.Path
    viewport: str
    frame_duration: float
    window: float
    level: float
    identity: dict[str, str]
    series_number: int = 1
    frame_of_reference: str = ""
    has_plane: bool = True


class CaptureWriter:
    """One viewport's output, for the life of one capture."""

    def add(self, index: int, frame: Frame):
        """Record one frame, or decline it.

        A writer that wants something this frame does not carry may write
        nothing rather than raise -- a cut is absent whenever the viewport is
        showing none, which is a state of the scene rather than an error. The
        caller cannot make that decision for it: within one capture some
        viewports write pixels and others write pictures.
        """
        raise NotImplementedError

    def close(self):
        """Finish the output.  Called even when the capture was cut short."""


def image_to_array(image: vtk.vtkImageData) -> np.ndarray:
    """A window capture as ``(rows, columns, components)`` uint8, top row first.

    VTK's buffer runs bottom-up, which every encoder outside VTK reads upside
    down.  The result is a fresh contiguous array, so a writer may keep it.
    """
    columns, rows, _ = image.GetDimensions()
    flat = vtknp.vtk_to_numpy(image.GetPointData().GetScalars())
    return np.ascontiguousarray(np.flipud(flat.reshape(rows, columns, -1)))
