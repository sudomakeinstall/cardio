"""What a capture is about to be written without.

A research session has whatever the source series happened to carry, which is
often much less than an archive expects: a de-identified export drops the
accession number, a phantom acquisition never had a study ID, a volume read
from NIfTI has no patient at all.  None of that stops a file being written, and
none of it is visible in the file afterwards -- an empty Type 2 element looks
exactly like a field that was legitimately blank.

So it is said at the one moment somebody can still do something about it.  This
warns rather than refuses, because whether a capture needs an accession number
is a question about where it is going, which the app does not know.  The two
things that are dangerous rather than merely incomplete are refused instead,
and only when a deployment has said it is in production.

Nothing here touches trame or the writers: it is a function of a source header
and a configuration, and can be checked without either.
"""

# System
import dataclasses as dc
import logging

logger = logging.getLogger(__name__)


@dc.dataclass(frozen=True)
class Requirement:
    """One field a receiving archive may want, and what it wants it for."""

    name: str
    reason: str


# Read off the source series.  Ordered by how much trouble their absence
# causes, so a truncated log still shows the worst of it.
SOURCE_FIELDS = (
    Requirement("PatientID", "an archive files every instance under it"),
    Requirement("PatientName", "an archive has no other way to name the patient"),
    Requirement(
        "StudyInstanceUID",
        "without one the capture opens a study of its own rather than joining "
        "the study it was derived from",
    ),
    Requirement(
        "FrameOfReferenceUID",
        "without one a reformat cannot be spatially correlated with the series "
        "it was cut from",
    ),
    Requirement("AccessionNumber", "nothing links the capture back to the order"),
    Requirement("StudyID", "nothing links the capture back to the order"),
    Requirement("StudyDate", "worklist reconciliation and sorting need it"),
    Requirement("StudyTime", "worklist reconciliation and sorting need it"),
    Requirement("Modality", "routing rules key off it"),
    Requirement("PatientBirthDate", "Type 2, and an archive may reconcile on it"),
    Requirement("PatientSex", "Type 2, and an archive may reconcile on it"),
    Requirement("PatientSize", "volumetry falls back to unindexed without it"),
    Requirement("PatientWeight", "volumetry falls back to unindexed without it"),
)

INSTITUTION = Requirement(
    "InstitutionName",
    "nothing says which site produced the capture; set capture_equipment."
    "institution_name",
)

NO_SOURCE = (
    "The active volume was not read from DICOM, so the capture carries no "
    "patient, study or frame of reference of its own"
)


def _absent(source, name: str) -> bool:
    """Whether the source is missing ``name``, empty counting as missing."""
    if source is None:
        return True
    return not getattr(source, name, None)


def missing(source, equipment) -> list[Requirement]:
    """The fields this capture is about to be written without.

    ``source`` is one instance of the series the active volume was read from,
    or None when it was read from a file.
    """
    absent = [field for field in SOURCE_FIELDS if _absent(source, field.name)]
    if not equipment.institution_name:
        absent.append(INSTITUTION)
    return absent


def report(source, equipment) -> list[Requirement]:
    """Say what is missing, and hand it back for the caller to count.

    One warning naming the fields, so a log is readable at a glance, and the
    reasons behind it, so the first person to ask why does not have to read
    this file to find out.
    """
    absent = missing(source, equipment)
    if not absent:
        return absent

    if source is None:
        logger.warning(f"{NO_SOURCE}.")

    logger.warning(
        f"This capture is being written without {len(absent)} field(s) a "
        f"receiving archive may want: {', '.join(f.name for f in absent)}."
    )
    for field in absent:
        logger.info(f"  {field.name}: {field.reason}.")

    return absent


def summarise(absent: list[Requirement]) -> str:
    """The clause the drawer adds to what a capture reports, or nothing."""
    if not absent:
        return ""
    return f"{len(absent)} field(s) a receiver may want are missing"
