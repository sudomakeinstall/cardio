"""How a written instance's pixels are encoded.

A cine Secondary Capture of a 1024 by 1024 view is three bytes a pixel and
twenty frames of them, which is 65MB nobody has to send: the same pixels as
JPEG-LS are 8MB, bit for bit the same on the way back.  Lossless throughout --
a capture is what the measurements were taken off, and an export that quietly
rounded them would be worth less than the space it saved.

Which of the two lossless encodings to ask for is a question about the
receiver rather than about the pixels: they cost within a few percent of each
other on a capture.  JPEG 2000 is the default because it is the one an
advanced visualization workstation is likelier to have heard of -- TeraRecon
iNtuition, for one, lists JPEG 2000 lossless and not JPEG-LS.  JPEG-LS is a
little smaller and several times faster, and uncompressed is what a receiver
that reads neither is sent.

JPEG 2000 has a floor the others do not: openjpeg encodes at six resolution
levels, which a frame smaller than 32 pixels either way cannot be halved into,
and it refuses one.  That is a capture of a view zoomed in past any use, and it
is written uncompressed rather than lost.

The pixels are encoded after the instance is built rather than by the
constructor that builds it, so that what a compressed capture holds is exactly
what an uncompressed one holds, and the choice between them touches nothing
else.
"""

# System
import enum
import logging

# Third Party
import pydicom as pd

logger = logging.getLogger(__name__)


class TransferSyntax(enum.StrEnum):
    """What a capture's pixels are written as.

    Only what the app can actually encode appears here, as with
    ``CaptureFormat``: a member it cannot write would be a configuration that
    fails at the end of a capture rather than at the start of one.
    """

    JPEG_LS = "jpeg-ls-lossless"
    JPEG_2000 = "jpeg-2000-lossless"
    UNCOMPRESSED = "uncompressed"


UIDS = {
    TransferSyntax.JPEG_LS: pd.uid.JPEGLSLossless,
    TransferSyntax.JPEG_2000: pd.uid.JPEG2000Lossless,
    TransferSyntax.UNCOMPRESSED: pd.uid.ExplicitVRLittleEndian,
}


def uid_for(syntax: TransferSyntax) -> str:
    """The transfer syntax UID ``syntax`` names."""
    return UIDS[TransferSyntax(syntax)]


def apply(dataset, syntax: TransferSyntax) -> None:
    """Encode ``dataset``'s pixels as ``syntax`` asks, in place.

    An instance is built uncompressed, so the plain syntax is already what it
    holds and there is nothing to do for it.

    ``generate_instance_uid`` is off because pydicom's default is to renumber
    the instance it compresses, under its own root rather than the one this
    deployment registered -- the identity is settled when the instance is
    built, and compressing it is not a new instance.

    An encoder that is missing or that refuses the pixels leaves the capture
    uncompressed rather than losing it: the file is larger than it was meant to
    be, which the log says, and everything in it is still true.  Anything else
    going wrong is not something to write a file about.
    """
    syntax = TransferSyntax(syntax)
    if syntax is TransferSyntax.UNCOMPRESSED:
        return

    try:
        dataset.compress(uid_for(syntax), generate_instance_uid=False)
    except (ImportError, RuntimeError, ValueError, AttributeError) as error:
        logger.error(
            f"Could not encode this capture as {syntax}: {error}. Writing it "
            "uncompressed; set capture_transfer_syntax to 'uncompressed' to "
            "ask for that in the first place."
        )
