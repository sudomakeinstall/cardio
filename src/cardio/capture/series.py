"""How each viewport's exported DICOM series is named.

A capture writes one series per viewport, and the two attributes anybody looks
a series up by -- the number it sorts under and the description it is listed
by -- are the two a PACS shows.  Both are named per viewport rather than once
for the capture: three reformats of one study are three series, and telling
them apart afterwards is the whole reason they carry names at all.
"""

# System
import collections
import logging

# Third Party
import pydantic as pc

# Internal
from ..state import VIEWPORTS

logger = logging.getLogger(__name__)

# The longest a Long String may be, which is the VR SeriesDescription is sent
# as.  A receiver rejects a longer one rather than shortening it, so the limit
# belongs at the point the description is written down.
DESCRIPTION_LIMIT = 64


class Series(pc.BaseModel):
    """What one viewport's exported series is called.

    Extras are forbidden, as they are on ``Scene``: a misspelled key in a
    hand-written config should say so rather than quietly do nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    number: int = pc.Field(
        default=1,
        ge=0,
        le=99999,
        description="SeriesNumber this viewport's capture is written with.",
    )
    description: str = pc.Field(
        default="",
        max_length=DESCRIPTION_LIMIT,
        description=(
            "SeriesDescription this viewport's capture is written with; empty "
            "names the series after the viewport and what it holds."
        ),
    )


def _numbered(viewport: str) -> Series:
    """This viewport's naming, numbered by where it sits among them.

    Distinct by default, so a capture of several viewports arrives as several
    series without anyone having had to number them.
    """
    return Series(number=VIEWPORTS.index(viewport) + 1)


class SeriesTags(pc.BaseModel):
    """The naming of every viewport's series, one entry each.

    A field per viewport rather than a mapping, so that a config names one the
    way it names everything else -- ``--capture_series.axial.number 400`` --
    and a viewport that does not exist is a misspelling rather than a key that
    quietly configures nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    vr: Series = pc.Field(
        default_factory=lambda: _numbered("vr"),
        description="The volume rendering's series.",
    )
    axial: Series = pc.Field(
        default_factory=lambda: _numbered("axial"),
        description="The axial view's series.",
    )
    coronal: Series = pc.Field(
        default_factory=lambda: _numbered("coronal"),
        description="The coronal view's series.",
    )
    sagittal: Series = pc.Field(
        default_factory=lambda: _numbered("sagittal"),
        description="The sagittal view's series.",
    )
    tile: Series = pc.Field(
        default_factory=lambda: _numbered("tile"),
        description="The tile grid's series.",
    )

    def of(self, viewport: str) -> Series:
        """How ``viewport``'s series is named."""
        return getattr(self, viewport)


def describe(viewport: str, kind: str, configured: str) -> str:
    """What a series is called: the configured name, or one saying what it is.

    An unnamed series still has to be tellable from the others written beside
    it, so the fallback carries the viewport and what the series holds rather
    than being blank.
    """
    return configured or f"cardio {viewport} ({kind})"


def repeated_numbers(numbers: dict[str, int]) -> dict[int, list[str]]:
    """The numbers more than one of ``numbers``' viewports is written with.

    Two series sharing a number within one study is legal and unhelpful: a
    viewer sorts them together and offers no way to say which is which.  It is
    the person's to allow, so this reports rather than refuses.
    """
    counts = collections.Counter(numbers.values())
    return {
        number: sorted(name for name, value in numbers.items() if value == number)
        for number, count in sorted(counts.items())
        if count > 1
    }
