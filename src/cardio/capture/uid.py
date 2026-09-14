"""Minting the UIDs a written instance is identified by.

Every DICOM object a system creates is named under a root its organisation
registered, and the root is how a receiver tells one creator's UIDs from
another's.  pydicom and highdicom each ship their own so that a library can
generate a usable UID out of the box; neither is anyone else's to assert.

Writing to a file under a borrowed root is harmless, and every research
capture this app has ever made has done it.  Sending one to an archive is not:
the instances claim to have been created by an organisation that did not create
them, and nothing downstream can tell that from the truth.  So the root is
configurable, the default says out loud that it is a default, and the
application says which root it is really using at the point of writing.
"""

# System
import logging
from importlib.metadata import version

# Third Party
import pydicom as pd

logger = logging.getLogger(__name__)

# pydicom's own registered root, which is what an unconfigured deployment
# generates under.  Named here so the check below can recognise it.
DEFAULT_ROOT = pd.uid.PYDICOM_ROOT_UID

# The arc under the root reserved for naming the application itself rather than
# an instance it wrote.  Generated UIDs carry a long hash in this position, so
# a short reserved arc cannot collide with one.
IMPLEMENTATION_ARC = "1"

# Implementation Version Name is a Short String, which holds sixteen.
VERSION_NAME_LIMIT = 16


def generate(root: str) -> str:
    """A fresh UID under ``root``."""
    return pd.uid.generate_uid(prefix=_prefix(root))


def implementation_class(root: str) -> str:
    """The UID naming this application as the thing that wrote a file."""
    return f"{_prefix(root)}{IMPLEMENTATION_ARC}"


def implementation_version() -> str:
    """This application and its version, in the sixteen characters SH holds."""
    return f"CARDIO {version('cardio')}"[:VERSION_NAME_LIMIT]


def stamp(dataset, root: str):
    """Say which implementation wrote the file, rather than which library.

    highdicom names itself here, which is true of how the object was built and
    not of what built it: a receiver asking who sent a file wants the product,
    and the library already says its own piece in the Contributing Equipment
    sequence.
    """
    dataset.file_meta.ImplementationClassUID = implementation_class(root)
    dataset.file_meta.ImplementationVersionName = implementation_version()


def is_registered(root: str) -> bool:
    """Whether ``root`` is one the deployment actually registered."""
    return _prefix(root) != _prefix(DEFAULT_ROOT)


def warn_if_unregistered(root: str, research: bool = False) -> bool:
    """Say so at startup, rather than at the capture that gets refused for it.

    A borrowed root stops a DICOM capture unless the session has said it is a
    research one, so what the session is about to be able to do is worth
    knowing before somebody spends a cardiac cycle finding out.
    """
    if is_registered(root):
        return False

    remedy = (
        "set uid_root to the organisation's own"
        if research
        else "DICOM captures are refused until uid_root names the "
        "organisation's own root, or research says this deployment sends "
        "nothing anywhere"
    )
    logger.warning(
        f"{root} is pydicom's registered root rather than this deployment's, "
        f"and files written under it must not be sent to an archive: {remedy}."
    )
    return True


def _prefix(root: str) -> str:
    """``root`` as generate_uid wants it: dot-terminated."""
    return root if root.endswith(".") else f"{root}."
