"""Writing a capture as a DICOM series.

One series per viewport, one instance per phase, which is the layout
``cardio.dicom`` already reads: a capture of the MPR views reopens in the app
it came from.

Both writers produce Secondary Capture objects.  Mirroring the source's SOP
class would look more faithful and be less so: an MR Image object requires
acquisition attributes -- scanning sequence, echo time, acquisition type --
that a reformat of one simply does not have, and inventing them to satisfy the
type would be the only way to write one.  Secondary Capture says what the image
actually is, and the Image Plane attributes may be added to it, so nothing true
is lost by saying so.

The instances are built by highdicom rather than assembled here.  Its
constructors will not hand back an object missing what the standard requires:
the Type 2 elements are written empty rather than left out, the patient and
study are copied off a source instance whole rather than tag by tag, a number
too long for its value representation is shortened rather than written past the
limit, and an instance in the patient coordinate system cannot be built without
a Patient Orientation at all.  Getting that by construction is the point of
going through it -- a rule the library enforces is one nobody has to remember.

What is left here is what its Secondary Capture deliberately does not carry:
where the plane sits, what the values mean, and what the image was derived
from.
"""

# System
import pathlib as pl

# Third Party
import highdicom as hd
import numpy as np
import pydicom as pd
from pydicom.sr.codedict import codes

# Internal
from . import uid, values
from .banner import stamp_scalars
from .base import CaptureWriter, Context, Frame, Plane
from .geometry import patient_orientation
from .series import describe

# UTF-8.  The patient's name is copied off the source unchanged, and a receiver
# is entitled to read it as plain ASCII unless the instance says otherwise.
CHARACTER_SET = "ISO_IR 192"

# highdicom will not build a Secondary Capture in the patient coordinate system
# without a Patient Orientation, and an image with no plane behind it has none
# to give.  Such an instance is built with this and then cleared; see
# ``_clear_orientation``.
PLACEHOLDER_ORIENTATION = ("L", "P")

# Why the source images are cited, and what was done to them.
SOURCE_PURPOSE = codes.DCM.SourceImageForImageProcessingOperation
DERIVATION_CODE = codes.DCM.MultiplanarReformatting
DERIVATION_DESCRIPTION = "Reformatted from the source series by cardio"


def encode(scalars: np.ndarray) -> tuple[np.ndarray, float, float]:
    """16-bit pixels, and the rescale that turns them back into the values.

    Unsigned throughout, because that is the only thing Secondary Capture
    carries: a signed range is written as the unsigned one under it with the
    offset as an intercept, which is the same numbers again once a viewer has
    applied the modality LUT.

    Integers spanning less than the 16-bit range are shifted rather than
    scaled, so a CT keeps its Hounsfield numbers exactly and a viewer's
    measurements read the same as the app's.  Anything wider is mapped onto the
    range with the slope and intercept that invert the mapping, which is as
    much of the original as a DICOM image can carry.
    """
    if np.issubdtype(scalars.dtype, np.integer):
        low, high = int(scalars.min()), int(scalars.max())
        if low >= 0 and high <= 65535:
            return scalars.astype(np.uint16), 1.0, 0.0
        if high - low <= 65535:
            return (scalars - low).astype(np.uint16), 1.0, float(low)

    low, high = float(scalars.min()), float(scalars.max())
    slope = (high - low) / 65535 or 1.0
    return np.rint((scalars - low) / slope).astype(np.uint16), slope, low


def rescale_type(modality: str, slope: float) -> str:
    """What the rescaled values are, once the modality LUT has been applied.

    "HU" only where they really are Hounsfield units -- a CT whose integers
    came through at unit slope.  Anything else has been mapped onto the 16-bit
    range and means whatever the source meant, which is what "US" says.
    """
    return "HU" if modality == "CT" and slope == 1.0 else "US"


class SeriesWriter(CaptureWriter):
    """One viewport's frames as one series of single-frame instances."""

    def __init__(self, context: Context):
        self.context = context
        self.directory = context.directory / context.viewport
        self.directory.mkdir(parents=True, exist_ok=True)
        self.series_uid = uid.generate(context.uid_root)

    def path_for(self, index: int) -> pl.Path:
        return self.directory / f"{index:04d}.dcm"

    def image(
        self,
        index: int,
        kind: str,
        pixels: np.ndarray,
        interpretation: str,
        bits: int,
        orientation: tuple[str, str] | None = None,
        spacing=None,
    ):
        """One instance, as much of it as highdicom knows how to fill.

        ``kind`` is what the series holds, which only the writer knows and only
        once it has a frame in hand; it names the series when the capture was
        not given a name of its own.
        """
        context = self.context
        identity = context.identity

        shared = dict(
            pixel_array=pixels,
            photometric_interpretation=interpretation,
            bits_allocated=bits,
            coordinate_system=hd.CoordinateSystemNames.PATIENT,
            series_instance_uid=self.series_uid,
            series_number=context.series_number,
            sop_instance_uid=uid.generate(context.uid_root),
            instance_number=index + 1,
            series_description=describe(
                context.viewport, kind, context.series_description
            ),
            transfer_syntax_uid=pd.uid.ExplicitVRLittleEndian,
            specific_character_set=CHARACTER_SET,
            patient_orientation=orientation or PLACEHOLDER_ORIENTATION,
            pixel_spacing=None if spacing is None else tuple(spacing),
            **context.equipment.attributes,
        )

        if identity.reference is not None:
            dataset = hd.sc.SCImage.from_ref_dataset(
                ref_dataset=identity.reference, **shared
            )
        else:
            dataset = hd.sc.SCImage(
                study_instance_uid=identity.study_instance_uid,
                patient_id=identity.patient_id,
                patient_name=identity.patient_name,
                **shared,
            )

        if orientation is None:
            _clear_orientation(dataset)

        uid.stamp(dataset, context.uid_root)
        return dataset

    def stamp(self, dataset, index: int, image_type: list[str]):
        """The attributes that say what the instance is and where it sits.

        ``ImageType`` and ``ConversionType`` are overridden rather than
        accepted: highdicom writes OTHER and DI, which describe neither a
        reformat nor the workstation it came off.
        """
        context = self.context

        dataset.ImageType = image_type
        # Type 1 for Secondary Capture: this came off a workstation.
        dataset.ConversionType = "WSD"
        dataset.Modality = context.identity.modality
        # Type 3, so an unset name is left out rather than written empty: absent
        # says "not recorded" where empty claims the value itself is blank.
        if context.equipment.station_name:
            dataset.StationName = context.equipment.station_name
        # Type 3, but a validator asks for it and the answer is never in doubt:
        # every transfer syntax this writes is lossless.
        dataset.LossyImageCompression = "00"
        # Type 3, and the one tag a viewer consults before measuring off an
        # image: a banner is lettering standing where pixels would otherwise
        # be, and saying so is the difference between an annotated image and a
        # falsified one.
        dataset.BurnedInAnnotation = "YES" if context.banner else "NO"
        # Milliseconds into the cycle, which is how the reader orders phases.
        dataset.TriggerTime = values.decimal(index * context.frame_duration * 1000.0)

        _derive(dataset, context)

    def save(self, dataset, index: int):
        dataset.save_as(self.path_for(index), enforce_file_format=True)


class SecondaryCaptureWriter(SeriesWriter):
    """The viewport as it looked: colour, and no geometry to speak of."""

    def add(self, index: int, frame: Frame):
        rgb = np.ascontiguousarray(frame.rgb[:, :, :3])

        dataset = self.image(
            index,
            "rendered",
            rgb,
            hd.PhotometricInterpretationValues.RGB,
            bits=8,
        )
        self.stamp(dataset, index, ["DERIVED", "SECONDARY"])
        self.save(dataset, index)


class SliceWriter(SeriesWriter):
    """The pixels behind the viewport: the values, and where they came from.

    A frame with nothing behind it is skipped rather than written as a picture:
    an MPR view with no active volume is showing nothing, and a series of blank
    greyscale images would only pretend otherwise.

    A banner is appended below the cut rather than drawn across it, so every
    row the volume was measured from keeps its value and the position the
    instance declares -- which is that of the first row -- stays true of it.
    """

    def add(self, index: int, frame: Frame):
        if frame.plane is None:
            return

        plane = frame.plane
        stored, slope, intercept = encode(
            stamp_scalars(plane.scalars, self.context.banner)
        )

        localizable = plane.location is not None
        dataset = self.image(
            index,
            _kind(localizable),
            stored,
            hd.PhotometricInterpretationValues.MONOCHROME2,
            bits=16,
            orientation=(patient_orientation(plane.location) if localizable else None),
            spacing=plane.pixel_spacing,
        )
        self.stamp(
            dataset,
            index,
            ["DERIVED", "SECONDARY", "MPR" if localizable else "MOSAIC"],
        )

        dataset.RescaleSlope = values.decimal(slope)
        dataset.RescaleIntercept = values.decimal(intercept)
        dataset.RescaleType = rescale_type(self.context.identity.modality, slope)

        # Left as tags rather than applied, so the values stay the ones the
        # volume holds and the window is only how they are first shown.
        dataset.WindowWidth = values.decimal(max(float(self.context.window), 1.0))
        dataset.WindowCenter = values.decimal(self.context.level)

        dataset.SliceThickness = values.decimal(plane.thickness)
        _locate(dataset, plane, self.context.identity.frame_of_reference)

        self.save(dataset, index)


def _kind(localizable: bool) -> str:
    return "reformat" if localizable else "mosaic"


def _clear_orientation(dataset):
    """Say nothing about which way the rows and columns run.

    Patient Orientation is Type 2C and empty is its truthful value here: a
    volume render has a camera rather than a row and a column, and the tile
    mosaic composes cuts taken at different poses.  highdicom needs a value to
    build the object at all, so the placeholder it was given is cleared once it
    has.
    """
    dataset.PatientOrientation = []


def _derive(dataset, context: Context):
    """Say what the instance was made from.

    A DERIVED image citing no source is a picture that appeared in the study.
    The sequence is what lets a reader get back from a reformat to the
    acquisition it is a reformat of, and it is the whole series rather than one
    frame of it: a cine reformat is taken through all of them.
    """
    sources = context.identity.source_images
    if not sources:
        return

    references = hd.ReferencedImageSequence(referenced_images=list(sources))
    for item in references:
        item.PurposeOfReferenceCodeSequence = [hd.sr.CodedConcept(*SOURCE_PURPOSE)]

    dataset.SourceImageSequence = references
    dataset.DerivationDescription = DERIVATION_DESCRIPTION
    dataset.DerivationCodeSequence = [hd.sr.CodedConcept(*DERIVATION_CODE)]


def _locate(dataset, plane: Plane, frame_of_reference: str):
    """Say where the plane is, or say nothing.

    A mosaic composes cuts taken at different poses, so it has no orientation
    and no position.  Omitting both is what makes that legible: a viewer then
    declines to localize the image rather than placing it somewhere wrong, and
    ``cardio.dicom`` skips it rather than reading it back as a slice.

    The frame of reference is the source's own wherever there is one.  Minting
    a new one would tell the archive that this cut and the series it was cut
    from cannot be compared, which is the opposite of true.
    """
    if plane.location is None:
        return

    if frame_of_reference:
        dataset.FrameOfReferenceUID = frame_of_reference
        # Type 2 wherever a frame of reference is declared, and empty is
        # honest: nothing here is measured from an anatomical landmark.
        dataset.PositionReferenceIndicator = None

    dataset.ImageOrientationPatient = values.decimals(plane.location.orientation)
    dataset.ImagePositionPatient = values.decimals(plane.location.position)
