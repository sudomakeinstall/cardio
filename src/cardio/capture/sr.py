"""Writing the volume curves out as a DICOM Structured Report.

Two CSV files are what a person opens; a Structured Report is what a system
reads.  An archive files it against the study, a reporting package pulls the
numbers into a report without anybody retyping them, and every value carries
the units and the coded concept saying what it is -- which a column heading
does not.

The measurement names are deliberately generic.  ``Metrics`` is explicit that
it does not know what phase anything was acquired at: the largest volume a
chamber reaches is an end-diastolic volume only because a reader has said the
structure is a pumping chamber, and LOINC's end-diastolic concepts name the
left ventricle specifically, which nothing here has been told.  So a volume is
a volume, qualified as the maximum or the minimum of the curve, and which
structure it belongs to is said by the finding site and by the segment the
group points at.  Saying less than that would be inventing the rest.
"""

# System
import logging

# Third Party
import highdicom as hd
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

# Internal
from . import uid
from .equipment import Equipment
from .series import DESCRIPTION_LIMIT

logger = logging.getLogger(__name__)

# UCUM units, spelled out because pydicom's dictionary does not carry them all
# and a unit half from the dictionary and half from here would be worse.
MILLILITRE = Code("ml", "UCUM", "milliliter")
MILLILITRE_PER_SQUARE_METRE = Code("ml/m2", "UCUM", "milliliter per square meter")
GRAM = Code("g", "UCUM", "gram")
GRAM_PER_SQUARE_METRE = Code("g/m2", "UCUM", "gram per square meter")
PERCENT = codes.UCUM.Percent

# What is being measured, and which end of the curve the number came from.
VOLUME = codes.SCT.Volume
MASS = codes.SCT.Mass
STROKE_VOLUME = codes.SCT.StrokeVolume
EJECTION_FRACTION = codes.SCT.CardiacEjectionFraction
MAXIMUM = codes.SCT.Maximum
MINIMUM = codes.SCT.Minimum

PROCEDURE = codes.SCT.ImagingProcedure
CODE_SCHEME = "SCT"

DEFAULT_DESCRIPTION = "cardio volumetry"


def units(group, indexed: bool = False):
    """What one structure's curve is quoted in.

    A structure given a density is reported as a mass, which is how myocardium
    is quoted; everything else is a volume.  The same choice the on-screen
    table makes, so the report and the table cannot disagree.
    """
    if group.density is not None:
        return GRAM_PER_SQUARE_METRE if indexed else GRAM
    return MILLILITRE_PER_SQUARE_METRE if indexed else MILLILITRE


def quantity(group):
    """Whether the curve is a volume or a mass, as a coded concept."""
    return MASS if group.density is not None else VOLUME


def finding_sites(group) -> list[hd.sr.FindingSite] | None:
    """Where in the body the measurement is of, when the structure says."""
    if not group.code:
        return None
    return [
        hd.sr.FindingSite(
            anatomic_location=hd.sr.CodedConcept(
                value=str(group.code), scheme_designator=CODE_SCHEME, meaning=group.name
            )
        )
    ]


def measurements(measurement, bsa: float | None) -> list[hd.sr.Measurement]:
    """One structure's numbers, each saying what it is and what it is in.

    The extremes are the curve's own, qualified as maximum and minimum rather
    than as end-diastolic and end-systolic: which phase a frame was acquired at
    is not something this app was ever told.
    """
    group, metrics = measurement.group, measurement.metrics
    what = quantity(group)

    values = [
        hd.sr.Measurement(
            name=what, value=metrics.maximum, unit=units(group), derivation=MAXIMUM
        ),
        hd.sr.Measurement(
            name=what, value=metrics.minimum, unit=units(group), derivation=MINIMUM
        ),
    ]

    if metrics.stroke is not None:
        values.append(
            hd.sr.Measurement(
                name=STROKE_VOLUME, value=metrics.stroke, unit=units(group)
            )
        )
    if metrics.ejection is not None:
        values.append(
            hd.sr.Measurement(
                name=EJECTION_FRACTION, value=metrics.ejection, unit=PERCENT
            )
        )

    if bsa:
        indexed = units(group, indexed=True)
        values.extend(
            [
                hd.sr.Measurement(
                    name=what,
                    value=metrics.maximum / bsa,
                    unit=indexed,
                    derivation=MAXIMUM,
                ),
                hd.sr.Measurement(
                    name=what,
                    value=metrics.minimum / bsa,
                    unit=indexed,
                    derivation=MINIMUM,
                ),
            ]
        )

    return values


def group_for(measurement, bsa, segment, uid_root: str):
    """One structure as a measurement group.

    Where a segmentation was written, the group points at the segment standing
    for this structure.  The reference identifies the structure, not the frame
    a given number came from: the maximum and the minimum are from different
    frames, and a group carrying both cannot honestly point at one of them.
    """
    tracking = hd.sr.TrackingIdentifier(
        uid=uid.generate(uid_root), identifier=measurement.group.name
    )
    shared = {
        "tracking_identifier": tracking,
        "finding_sites": finding_sites(measurement.group),
        "measurements": measurements(measurement, bsa),
    }

    if segment is None:
        return hd.sr.MeasurementsAndQualitativeEvaluations(**shared)
    return hd.sr.VolumetricROIMeasurementsAndQualitativeEvaluations(
        referenced_segment=segment, **shared
    )


def segment_references(segmentations) -> dict[int, hd.sr.ReferencedSegment]:
    """The segment of the first written instance standing for each structure."""
    if not segmentations:
        return {}

    first = segmentations[0]
    return {
        int(item.SegmentNumber): hd.sr.ReferencedSegment(
            sop_class_uid=first.SOPClassUID,
            sop_instance_uid=first.SOPInstanceUID,
            segment_number=int(item.SegmentNumber),
            source_series=hd.sr.SourceSeriesForSegmentation(
                referenced_series_instance_uid=first.ReferencedSeriesSequence[
                    0
                ].SeriesInstanceUID
            ),
        )
        for item in first.SegmentSequence
    }


def observation_context(equipment: Equipment, uid_root: str):
    """Who observed this: a device, which is what an automatic measurement is."""
    return hd.sr.ObservationContext(
        observer_device_context=hd.sr.ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
                uid=uid.implementation_class(uid_root),
                manufacturer_name=equipment.manufacturer,
                model_name=equipment.model_name,
                serial_number=equipment.device_serial_number or None,
                physical_location=equipment.station_name or None,
            ),
        )
    )


def write_measurements(
    result,
    source,
    path,
    segmentations=None,
    equipment: Equipment | None = None,
    uid_root: str = uid.DEFAULT_ROOT,
    series_number: int = 301,
    series_description: str = DEFAULT_DESCRIPTION,
):
    """One study's volume curves as a Structured Report, and where it went.

    ``source`` is the volume the curves were measured off, which has to have
    been read from DICOM: a report is evidence about images, and names them.
    """
    if source is None or not source.instances:
        raise ValueError(
            "A Structured Report is evidence about images and names them, so "
            "the volume measured has to have been read from DICOM."
        )
    if not result.measurements:
        raise ValueError("There is nothing to report until a structure is named.")

    equipment = equipment or Equipment()
    references = source.datasets
    segments = segment_references(segmentations)

    report = hd.sr.MeasurementReport(
        observation_context=observation_context(equipment, uid_root),
        procedure_reported=PROCEDURE,
        imaging_measurements=[
            group_for(measurement, result.bsa, segments.get(number), uid_root)
            for number, measurement in enumerate(result.measurements, start=1)
        ],
    )

    document = hd.sr.Comprehensive3DSR(
        evidence=[*references, *(segmentations or [])],
        content=report[0],
        series_instance_uid=uid.generate(uid_root),
        series_number=series_number,
        sop_instance_uid=uid.generate(uid_root),
        instance_number=1,
        manufacturer=equipment.manufacturer,
        institution_name=equipment.institution_name or None,
        series_description=series_description[:DESCRIPTION_LIMIT],
        is_complete=True,
        is_final=False,
        is_verified=False,
    )
    uid.stamp(document, uid_root)

    document.save_as(path, enforce_file_format=True)
    logger.info(f"{len(result.measurements)} measurement group(s) written to {path}.")
    return path
