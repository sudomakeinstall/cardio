"""Writing a capture as a DICOM series.

One series per viewport, one instance per phase, which is the layout
``cardio.dicom`` already reads: a capture of the MPR views reopens in the app
it came from.

Both writers produce Secondary Capture objects.  Mirroring the source's SOP
class would look more faithful and be less so: an MR Image object requires
acquisition attributes -- scanning sequence, echo time, acquisition type -- that
a reformat of one simply does not have, and inventing them to satisfy the type
would be the only way to write one.  Secondary Capture says what the image
actually is, and DICOM's standard extension allows the Image Plane attributes
to be added to it, so nothing true is lost by saying so.
"""

# System
import pathlib as pl

# Third Party
import numpy as np
import pydicom as pd

# Internal
from .base import CaptureWriter, Context, Frame, Plane

# What a derived series copies from the source so it lands in the same study.
# ``cardio.dicom.DISPLAY_TAGS`` already reads every one of them.
IDENTITY_TAGS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyInstanceUID",
    "StudyDate",
    "StudyTime",
    "StudyDescription",
    "AccessionNumber",
)


def _dataset(context: Context, image_type: list[str]) -> pd.dataset.Dataset:
    """The parts of an instance that do not depend on what it holds."""
    dataset = pd.dataset.Dataset()
    dataset.file_meta = pd.dataset.FileMetaDataset()
    dataset.file_meta.MediaStorageSOPClassUID = pd.uid.SecondaryCaptureImageStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = pd.uid.generate_uid()
    dataset.file_meta.TransferSyntaxUID = pd.uid.ExplicitVRLittleEndian

    dataset.SOPClassUID = pd.uid.SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.ImageType = image_type
    # Type 1 for Secondary Capture: this came off a workstation.
    dataset.ConversionType = "WSD"
    dataset.Modality = "OT"

    for tag, value in context.identity.items():
        setattr(dataset, tag, value)

    return dataset


def encode(scalars: np.ndarray) -> tuple[np.ndarray, float, float]:
    """16-bit pixels, and the rescale that turns them back into the values.

    Integers that fit are written through untouched, so a CT keeps its
    Hounsfield numbers and a viewer's measurements read the same as the app's.
    Anything else is mapped onto the 16-bit range with the slope and intercept
    that invert the mapping, which is as much of the original as a DICOM image
    can carry.
    """
    if np.issubdtype(scalars.dtype, np.integer):
        low, high = int(scalars.min()), int(scalars.max())
        if low >= 0 and high <= 65535:
            return scalars.astype(np.uint16), 1.0, 0.0
        if low >= -32768 and high <= 32767:
            return scalars.astype(np.int16), 1.0, 0.0

    low, high = float(scalars.min()), float(scalars.max())
    slope = (high - low) / 65535 or 1.0
    return np.rint((scalars - low) / slope).astype(np.uint16), slope, low


class SeriesWriter(CaptureWriter):
    """One viewport's frames as one series of single-frame instances."""

    def __init__(self, context: Context):
        self.context = context
        self.directory = context.directory / context.viewport
        self.directory.mkdir(parents=True, exist_ok=True)
        self.series_uid = pd.uid.generate_uid()

    def path_for(self, index: int) -> pl.Path:
        return self.directory / f"{index:04d}.dcm"

    def stamp(self, dataset, index: int, description: str):
        """The attributes that place an instance within its series."""
        dataset.SeriesInstanceUID = self.series_uid
        dataset.SeriesNumber = self.context.series_number
        dataset.SeriesDescription = description
        dataset.InstanceNumber = index + 1
        # Milliseconds into the cycle, which is how the reader orders phases.
        # Rounded because DS holds sixteen characters, and a rate that does not
        # divide the cycle evenly writes more than that in full precision.
        dataset.TriggerTime = round(index * self.context.frame_duration * 1000.0, 3)


class SecondaryCaptureWriter(SeriesWriter):
    """The viewport as it looked: colour, and no geometry to speak of."""

    def add(self, index: int, frame: Frame):
        rgb = np.ascontiguousarray(frame.rgb[:, :, :3])

        dataset = _dataset(self.context, ["DERIVED", "SECONDARY"])
        self.stamp(dataset, index, f"cardio {self.context.viewport} (rendered)")

        dataset.Rows, dataset.Columns = rgb.shape[:2]
        dataset.SamplesPerPixel = 3
        dataset.PhotometricInterpretation = "RGB"
        dataset.PlanarConfiguration = 0
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.PixelRepresentation = 0
        dataset.PixelData = rgb.tobytes()

        dataset.save_as(self.path_for(index), enforce_file_format=True)


class SliceWriter(SeriesWriter):
    """The pixels behind the viewport: the values, and where they came from.

    A frame with nothing behind it is skipped rather than written as a picture:
    an MPR view with no active volume is showing nothing, and a series of blank
    greyscale images would only pretend otherwise.
    """

    def add(self, index: int, frame: Frame):
        if frame.plane is None:
            return

        plane = frame.plane
        stored, slope, intercept = encode(plane.scalars)

        localizable = plane.location is not None
        dataset = _dataset(
            self.context,
            ["DERIVED", "SECONDARY", "MPR" if localizable else "MOSAIC"],
        )
        self.stamp(
            dataset, index, f"cardio {self.context.viewport} ({_kind(localizable)})"
        )

        dataset.Rows, dataset.Columns = stored.shape
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1 if stored.dtype == np.int16 else 0
        dataset.RescaleSlope = slope
        dataset.RescaleIntercept = intercept
        dataset.RescaleType = "US"

        # Left as tags rather than applied, so the values stay the ones the
        # volume holds and the window is only how they are first shown.
        dataset.WindowWidth = float(self.context.window)
        dataset.WindowCenter = float(self.context.level)

        dataset.PixelSpacing = [float(v) for v in plane.pixel_spacing]
        dataset.SliceThickness = float(plane.thickness)
        _locate(dataset, plane, self.context.frame_of_reference)

        dataset.PixelData = np.ascontiguousarray(stored).tobytes()
        dataset.save_as(self.path_for(index), enforce_file_format=True)


def _kind(localizable: bool) -> str:
    return "reformat" if localizable else "mosaic"


def _locate(dataset, plane: Plane, frame_of_reference: str):
    """Say where the plane is, or say nothing.

    A mosaic composes cuts taken at different poses, so it has no orientation
    and no position.  Omitting both is what makes that legible: a viewer then
    declines to localize the image rather than placing it somewhere wrong, and
    ``cardio.dicom`` skips it rather than reading it back as a slice.
    """
    if plane.location is None:
        return

    # Shared by every viewport of one capture: they are cuts of one volume, in
    # one set of world coordinates.
    if frame_of_reference:
        dataset.FrameOfReferenceUID = frame_of_reference
    dataset.ImageOrientationPatient = [float(v) for v in plane.location.orientation]
    dataset.ImagePositionPatient = [float(v) for v in plane.location.position]
