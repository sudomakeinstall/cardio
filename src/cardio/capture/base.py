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
import pydicom as pd
import vtk
from vtk.util import numpy_support as vtknp

# Internal
from .equipment import Equipment


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
class Identity:
    """Who a capture is of, which study it joins, and what it was made from.

    ``source_images`` are the headers of the instances behind the active
    volume, empty when it was read from a file.  The first of them is handed
    whole to the writer, which copies the patient and study modules off it
    rather than picking tags out one at a time: those fields have to agree with
    each other, and a capture that names a real study under a placeholder
    patient is one an archive either refuses or files against the wrong person.

    The whole sequence is what a derived instance cites as its source, so that
    a reformat can be got back from to the acquisition it is a reformat of.

    Without a source the capture stands alone, under a study of its own, and
    the placeholder patient stands for all of it rather than for whichever
    fields happened to be missing.
    """

    source_images: tuple[pd.dataset.Dataset, ...] = ()
    study_instance_uid: str = ""
    patient_id: str = "CARDIO"
    # A person name is caret-separated components, and a bare word is read as
    # ambiguous rather than as a family name.  The trailing caret is how DICOM
    # says a single-component name was meant.
    patient_name: str = "Anonymous^"
    frame_of_reference: str = ""
    modality: str = "OT"

    @property
    def reference(self) -> pd.dataset.Dataset | None:
        """The one instance the patient and study are copied off."""
        return self.source_images[0] if self.source_images else None


@dc.dataclass(frozen=True)
class Context:
    """What the capture as a whole knows, for the writers that need it.

    ``has_plane`` is what the viewport can offer, not what a given frame did:
    the volume render has a camera rather than an image plane, so no format
    can ask it for one.

    ``banner`` is the line tagged onto the lower margin of what gets written,
    empty for a capture that carries none.  The picture formats receive it
    already stamped into the frame; it is carried here for the writers that
    have to say in their own terms that it is there.
    """

    directory: pl.Path
    viewport: str
    frame_duration: float
    window: float
    level: float
    identity: Identity = dc.field(default_factory=Identity)
    equipment: Equipment = dc.field(default_factory=Equipment)
    series_number: int = 1
    series_description: str = ""
    has_plane: bool = True
    banner: str = ""


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
