"""What a capture is about to be written without.

A research session has whatever the source series happened to carry, which is
often much less than an archive expects: a de-identified export drops the
accession number, a phantom acquisition never had a study ID, a volume read
from NIfTI has no patient at all.  None of that stops a file being written, and
none of it is visible in the file afterwards -- an empty Type 2 element looks
exactly like a field that was legitimately blank.

So it is said at the one moment somebody can still do something about it, and
for a few fields it is refused rather than said.  The ones refused are the ones
nothing downstream can check: an instance filed under a patient and study this
app invented is not an incomplete instance but a wrong one, and an archive that
accepts it has no way to know that.  Those are required by default and a
deployment opts out of them, rather than the other way around -- a session that
has not thought about where its files are going is exactly the session that
cannot be relied on to notice.

The rest is reported and written anyway, because whether a capture needs an
accession number is a question about where it is going, which the app does not
know.

Nothing here touches trame or the writers: it is a function of a source header
and a configuration, and can be checked without either.
"""

# System
import dataclasses as dc
import logging

# Internal
from . import uid

logger = logging.getLogger(__name__)


@dc.dataclass(frozen=True)
class Requirement:
    """One field a receiving archive may want, and what it wants it for."""

    name: str
    reason: str


# The fields a capture is refused for.  Two different rules put a field here,
# and it is worth knowing which one before moving another.
#
# The first three are what this app would otherwise have to invent.  A receiver
# takes them on trust -- an archive files what it is given -- so a stand-in for
# one is indetectable once the file has left.
#
# The last two are honestly empty when they are absent: both are Type 2, and a
# zero-length element is what gets written.  They are here because an instance
# nobody can reconcile with an order is a stray in the archive even though
# nothing about it is untrue.  That is a claim about where files are going
# rather than about the file, so a site that files differently moves them back
# down and gets warnings instead.
REQUIRED = (
    Requirement("PatientID", "an archive files every instance under it"),
    Requirement("PatientName", "an archive has no other way to name the patient"),
    Requirement(
        "StudyInstanceUID",
        "without one the capture opens a study of its own rather than joining "
        "the study it was derived from",
    ),
    Requirement("AccessionNumber", "nothing links the capture back to the order"),
    Requirement(
        "StudyDate",
        "worklist reconciliation and sorting need it, and an acquisition that "
        "really happened has one",
    ),
)

# Wanted, and their absence the person's to accept: what a receiver insists on
# depends on the receiver.  Ordered by how much trouble they cause, so a
# truncated log still shows the worst of it.
ADVISORY = (
    Requirement(
        "FrameOfReferenceUID",
        "without one a reformat cannot be spatially correlated with the series "
        "it was cut from",
    ),
    Requirement("StudyID", "nothing links the capture back to the order"),
    Requirement("StudyTime", "worklist reconciliation and sorting need it"),
    Requirement("Modality", "routing rules key off it"),
    Requirement("PatientBirthDate", "Type 2, and an archive may reconcile on it"),
    Requirement("PatientSex", "Type 2, and an archive may reconcile on it"),
    Requirement("PatientSize", "volumetry falls back to unindexed without it"),
    Requirement("PatientWeight", "volumetry falls back to unindexed without it"),
)

# Read off the source series, worst first.
SOURCE_FIELDS = REQUIRED + ADVISORY

# Not read off anything: the root is configured, and the default is a library's
# own.  It stops a capture for the same reason the fields above do -- a
# receiver cannot tell an instance created under somebody else's root from one
# created by them.
UNREGISTERED_ROOT = Requirement(
    "uid_root",
    "instances written under it claim to have been created by whoever did register it",
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


def blocking(source, root: str) -> list[Requirement]:
    """What makes this capture unsafe to send, whatever else is missing."""
    stopping = [field for field in REQUIRED if _absent(source, field.name)]
    if not uid.is_registered(root):
        stopping.append(UNREGISTERED_ROOT)
    return stopping


def refused(source, root: str) -> str:
    """Why this capture will not be written as it stands, or nothing.

    Named rather than counted, and with the way out of it: the whole of the
    message is what somebody has to change to get their file.
    """
    stopping = blocking(source, root)
    if not stopping:
        return ""

    clauses = []
    fields = [field.name for field in stopping if field is not UNREGISTERED_ROOT]
    if fields:
        clauses.append(f"the source carries no {', '.join(fields)}")
    if UNREGISTERED_ROOT in stopping:
        clauses.append(f"{root} is not this deployment's registered root")

    return f"Refused: {' and '.join(clauses)}; set research to write it anyway"


def summarise(absent: list[Requirement]) -> str:
    """The clause the drawer adds to what a capture reports, or nothing."""
    if not absent:
        return ""
    return f"{len(absent)} field(s) a receiver may want are missing"
