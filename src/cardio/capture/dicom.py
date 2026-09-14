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
import logging
import math
import pathlib as pl

# Third Party
import highdicom as hd
import numpy as np
import pydicom as pd
from pydicom.sr.codedict import codes

# Internal
from . import encoding, uid, values
from .banner import stamp_scalars
from .base import CaptureWriter, Context, Frame, Plane
from .geometry import patient_orientation
from .series import describe

logger = logging.getLogger(__name__)

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


def encode(
    scalars: np.ndarray, quantum: float = 0.0
) -> tuple[np.ndarray, float, float]:
    """16-bit pixels, and the rescale that turns them back into the values.

    Unsigned throughout, because that is the only thing Secondary Capture
    carries: a signed range is written as the unsigned one under it with the
    offset as an intercept, which is the same numbers again once a viewer has
    applied the modality LUT.

    Integers spanning less than the 16-bit range are shifted rather than
    scaled, so a CT keeps its Hounsfield numbers exactly and a viewer's
    measurements read the same as the app's.

    ``quantum`` does the same for values that arrived as floats but were never
    finer than a step -- the usual case, a CT read from NIfTI and resliced --
    by storing them on that step instead of on the whole 16-bit range.  A cut
    of a study spanning 2000 Hounsfield units stretched over 65535 levels is
    being kept to a thirty-second of a unit it never had, and every one of
    those levels is interpolation noise a lossless encoder is then obliged to
    preserve: on the source's own step the same capture is a little over a
    third the size, and holds the same numbers.

    What does not fit on that step, and anything whose step is unknown, is
    mapped onto the range with the slope and intercept that invert the
    mapping, which is as much of the original as a DICOM image can carry.
    """
    if np.issubdtype(scalars.dtype, np.integer):
        low, high = int(scalars.min()), int(scalars.max())
        if low >= 0 and high <= 65535:
            return scalars.astype(np.uint16), 1.0, 0.0
        if high - low <= 65535:
            return (scalars - low).astype(np.uint16), 1.0, float(low)

    low, high = float(scalars.min()), float(scalars.max())

    if quantum > 0.0:
        # Onto the step's own grid rather than to the lowest value present, so
        # that what comes back out lands on the values the volume held.
        base = math.floor(low / quantum) * quantum
        if (high - base) / quantum <= 65535:
            stored = np.rint((scalars - base) / quantum).astype(np.uint16)
            return stored, quantum, base

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

    # Whether an instance carries the time it stands at in the cycle.  A
    # multi-frame object says its timing once, in the Cine module, so a trigger
    # time on it would be the timing of the whole loop rather than of a frame.
    timed_per_instance = True

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
            # Built uncompressed whatever it is written as: the pixels are
            # encoded once the instance is whole, by ``encoding.apply``.
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

    def stamp(self, dataset, index: int, image_type: list[str], phase=None):
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
        if self.timed_per_instance:
            # Milliseconds into the cycle, which is how the reader orders
            # phases.  Timed by the phase the frame stands at rather than by
            # its place in the capture: a rotation runs through several cycles,
            # and a trigger time past the end of one describes nothing.
            step = index if phase is None else phase
            dataset.TriggerTime = values.decimal(step * context.frame_duration * 1000.0)

        _derive(dataset, context)

    def save(self, dataset, index: int):
        encoding.apply(dataset, self.context.transfer_syntax)
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
        self.stamp(dataset, index, ["DERIVED", "SECONDARY"], frame.phase)
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
            stamp_scalars(plane.scalars, self.context.banner), plane.quantum
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
            frame.phase,
        )

        dataset.RescaleSlope = values.decimal(slope)
        dataset.RescaleIntercept = values.decimal(intercept)
        dataset.RescaleType = rescale_type(self.context.identity.modality, slope)

        # Left as tags rather than applied, so the values stay the ones the
        # volume holds and the window is only how they are first shown.
        dataset.WindowWidth = values.decimal(max(float(self.context.window), 1.0))
        dataset.WindowCenter = values.decimal(self.context.level)

        # Only alongside the rest of the Image Plane module: on its own it is
        # a thickness of a plane the instance never says the place of.
        if localizable:
            dataset.SliceThickness = values.decimal(plane.thickness)
        _locate(dataset, plane, self.context.identity.frame_of_reference)

        self.save(dataset, index)


# --- the cine as one object ----------------------------------------------------

# Frame Time, which is what the frame increment pointer points at: the frames
# are evenly spaced in time, so one interval describes all of them.
FRAME_TIME_TAG = pd.tag.Tag(0x0018, 0x1063)


class MultiFrameWriter(SeriesWriter):
    """One viewport's whole cine as a single multi-frame instance.

    A cine viewer plays a multi-frame object; a series of single-frame ones it
    merely sorts, and whether it plays them depends on the viewer.  Saying the
    frames are a sequence in time -- which is what the Cine module is for --
    is the difference between a loop and a stack of pictures.

    The frames are held until ``close`` because a multi-frame object has to
    declare how many it has before the first of them, and a capture may be cut
    short at any point.  What has been collected by then is what gets written.

    The pixels are encoded once over the whole stack rather than frame by
    frame, so every frame is on the same scale and one rescale describes all
    of them -- which a multi-frame object, carrying one, requires.

    Written for a receiver that asks for it rather than by default.  Two things
    make the single-frame series the one to send: Sectra reads both, but
    TeraRecon iNtuition lists only plain Secondary Capture among the classes it
    stores without being configured to; and these classes have no Image Plane
    module, so the position and orientation ``locate`` writes sit at the root
    of the dataset, where the standard puts nothing for them.  A viewer is
    entitled to ignore them and will.  ``cardio`` reads them back, which is
    what they are there for.  Saying it properly means functional groups --
    Pixel Measures, Plane Position, Plane Orientation -- which nothing here
    builds.
    """

    sop_class_uid: str = ""
    timed_per_instance = False

    def __init__(self, context: Context):
        super().__init__(context)
        self._frames: list[np.ndarray] = []
        self._planes: list[Plane | None] = []

    def collect(self, frame: Frame):
        """This frame's pixels, or None when there is nothing behind it."""
        raise NotImplementedError

    def add(self, index: int, frame: Frame):
        pixels = self.collect(frame)
        if pixels is None:
            return
        self._frames.append(pixels)
        self._planes.append(frame.plane)

    def close(self):
        if not self._frames:
            return

        shapes = {frame.shape for frame in self._frames}
        if len(shapes) > 1:
            # A multi-frame object cannot hold frames of different sizes, and
            # raising here would mask whatever cut the capture short.
            logger.error(
                f"{self.context.viewport} was captured at {len(shapes)} "
                "different sizes, which one multi-frame instance cannot hold; "
                "writing none."
            )
            return

        self.write(np.stack(self._frames))

    def write(self, stack: np.ndarray):
        raise NotImplementedError

    def shell(self, kind: str):
        """The instance every module but the pixels and the timing.

        Built through highdicom's base class rather than by hand, so that a
        multi-frame object gets the same Type 2 elements, the same equipment
        module and the same file meta as the single-frame one beside it.
        """
        context = self.context
        identity = context.identity
        reference = identity.reference

        dataset = hd.base.SOPClass(
            study_instance_uid=(
                str(reference.StudyInstanceUID)
                if reference is not None
                else identity.study_instance_uid
            ),
            series_instance_uid=self.series_uid,
            series_number=context.series_number,
            sop_instance_uid=uid.generate(context.uid_root),
            sop_class_uid=self.sop_class_uid,
            instance_number=1,
            modality=identity.modality,
            transfer_syntax_uid=pd.uid.ExplicitVRLittleEndian,
            specific_character_set=CHARACTER_SET,
            series_description=describe(
                context.viewport, kind, context.series_description
            ),
            patient_id=None if reference is not None else identity.patient_id,
            patient_name=None if reference is not None else identity.patient_name,
            **context.equipment.attributes,
        )

        if reference is not None:
            dataset.copy_patient_and_study_information(reference)

        uid.stamp(dataset, context.uid_root)
        return dataset

    def cine(self, dataset, frames: int):
        """Say that the frames are a sequence in time, and how fast it runs."""
        milliseconds = self.context.frame_duration * 1000.0

        dataset.NumberOfFrames = frames
        # What the frames advance by, named by the tag of the attribute that
        # says how much: evenly spaced, so one interval covers all of them.
        dataset.FrameIncrementPointer = FRAME_TIME_TAG
        dataset.FrameTime = values.decimal(milliseconds)
        dataset.CineRate = round(1000.0 / milliseconds)
        dataset.RecommendedDisplayFrameRate = dataset.CineRate
        dataset.PreferredPlaybackSequencing = 0

    def locate(self, dataset):
        """Say where the frames are, if they are all in the same place.

        A multi-frame Secondary Capture carries one position and one
        orientation, not one per frame.  That is true of a cine of a fixed
        plane and false of one whose plane moves through the cycle -- a snap
        lock following a valve, say -- and the honest thing to do about a pose
        the object cannot express is to leave it out, exactly as the mosaic
        does.
        """
        located = [plane.location for plane in self._planes if plane is not None]
        if not located or any(location is None for location in located):
            return

        poses = {
            (tuple(location.orientation), tuple(location.position))
            for location in located
        }
        if len(poses) > 1:
            logger.warning(
                f"The {self.context.viewport} cut moves over the cycle, which "
                "one multi-frame instance cannot say; it is written without a "
                "position."
            )
            return

        _locate(dataset, self._planes[0], self.context.identity.frame_of_reference)


class MultiFrameRenderedWriter(MultiFrameWriter):
    """The viewport as it looked, all of it in one object."""

    sop_class_uid = pd.uid.MultiFrameTrueColorSecondaryCaptureImageStorage
    bits_allocated = 8

    def collect(self, frame: Frame):
        return np.ascontiguousarray(frame.rgb[:, :, :3])

    def write(self, stack: np.ndarray):
        dataset = self.shell("rendered")
        self.stamp(dataset, 0, ["DERIVED", "SECONDARY"])
        self.cine(dataset, len(stack))

        dataset.SamplesPerPixel = 3
        dataset.PhotometricInterpretation = "RGB"
        dataset.PlanarConfiguration = 0
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.PixelRepresentation = 0
        dataset.Rows, dataset.Columns = stack.shape[1:3]
        dataset.PresentationLUTShape = "IDENTITY"
        dataset.PatientOrientation = []
        dataset.PixelData = _even(np.ascontiguousarray(stack).tobytes())

        self.save(dataset, 0)


class MultiFrameSliceWriter(MultiFrameWriter):
    """The pixels behind the viewport, all of them in one object."""

    sop_class_uid = pd.uid.MultiFrameGrayscaleWordSecondaryCaptureImageStorage

    def collect(self, frame: Frame):
        if frame.plane is None:
            return None
        return stamp_scalars(frame.plane.scalars, self.context.banner)

    def write(self, stack: np.ndarray):
        plane = next((p for p in self._planes if p is not None), None)
        stored, slope, intercept = encode(stack, plane.quantum if plane else 0.0)
        localizable = plane is not None and plane.location is not None

        dataset = self.shell(_kind(localizable))
        self.stamp(
            dataset,
            0,
            ["DERIVED", "SECONDARY", "MPR" if localizable else "MOSAIC"],
        )
        self.cine(dataset, len(stored))

        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.Rows, dataset.Columns = stored.shape[1:3]

        dataset.RescaleSlope = values.decimal(slope)
        dataset.RescaleIntercept = values.decimal(intercept)
        dataset.RescaleType = rescale_type(self.context.identity.modality, slope)
        # Type 1C for a MONOCHROME2 multi-frame capture: the stored values go
        # to the display unchanged, the window being only how they are shown.
        dataset.PresentationLUTShape = "IDENTITY"

        dataset.WindowWidth = values.decimal(max(float(self.context.window), 1.0))
        dataset.WindowCenter = values.decimal(self.context.level)

        if plane is not None:
            dataset.PixelSpacing = values.decimals(plane.pixel_spacing)
            dataset.SliceThickness = values.decimal(plane.thickness)
        self.locate(dataset)
        if not localizable:
            dataset.PatientOrientation = []
        else:
            dataset.PatientOrientation = list(patient_orientation(plane.location))

        dataset.PixelData = _even(np.ascontiguousarray(stored).tobytes())

        self.save(dataset, 0)


def _even(data: bytes) -> bytes:
    """``data`` padded to the even length every DICOM element is written at."""
    return data if len(data) % 2 == 0 else data + b"\x00"


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
