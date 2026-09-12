"""What a segmentation measures, over the cardiac cycle.

A label image says how many voxels carry each label; a voxel says how much
space it stands for.  Between the two there is a volume, and over a cine series
there is a curve -- which is the whole of the measurement, and is why nothing
here reads a file or draws anything.

A structure is a group of labels rather than a single one: a ventricle that has
been segmented as a bloodpool, an outflow tract and a trabecular mesh is one
chamber whatever the labelling, and the sum is what a reader means by its
volume.  A group with a density is reported as a mass instead, which is what
myocardium is conventionally quoted as.
"""

# System
import dataclasses as dc
import enum
import logging
import math

# Third Party
import numpy as np
import pydantic as pc

# Internal
from .types import RGBColor
from .utils import label_color

logger = logging.getLogger(__name__)

# A millilitre is a cubic centimetre, and image spacing is in millimetres.
MM3_PER_ML = 1000.0


class BSAFormula(str, enum.Enum):
    """How a body surface area is derived from a height and a weight.

    A ``str`` enum so that comparisons against the trame state variable, which
    carries the choice as a plain string, keep working unchanged.
    """

    NONE = "none"
    MOSTELLER = "mosteller"
    DUBOIS = "dubois"
    HAYCOCK = "haycock"


class StructureGroup(pc.BaseModel):
    """One thing to measure, and the labels it is made of.

    Extras are forbidden, as they are on ``Scene``: a misspelled key in a
    hand-written config should say so rather than quietly do nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    name: str = pc.Field(
        description='What the curve is called. CLI usage: --volumetry.groups \'[{"name":"LV","labels":[6,3,8]}]\''
    )
    labels: list[int] = pc.Field(
        min_length=1,
        description=(
            "Label values summed into this structure. Several, because one "
            "chamber may be segmented as a bloodpool, an outflow tract and a "
            "trabecular mesh."
        ),
    )
    color: RGBColor | None = pc.Field(
        default=None,
        description=(
            "Colour the curve is drawn in, as three components in [0, 1]. Left "
            "unset, it is the colour its first label is drawn in everywhere else."
        ),
    )
    density: float | None = pc.Field(
        default=None,
        gt=0.0,
        description=(
            "Grams per millilitre. Set, the structure is reported as a mass "
            "rather than as a volume -- which is how myocardium is quoted. "
            "CLI usage: 1.05"
        ),
    )
    cycle: bool = pc.Field(
        default=True,
        description=(
            "Whether this structure fills and empties, and so has a stroke "
            "volume and an ejection fraction. False for the ones that only "
            "change shape, whose difference between extremes means nothing."
        ),
    )

    @property
    def unit(self) -> str:
        """What this structure is measured in, given whether it has a density."""
        return "g" if self.density is not None else "mL"

    @property
    def rgb(self) -> tuple[float, float, float]:
        """The colour to draw it in, as three components in [0, 1].

        Falling back to the first label's own colour rather than to a palette
        of its own, so that a group which named none is drawn in the colour the
        MPR overlays already draw it in.
        """
        return (
            tuple(self.color) if self.color is not None else label_color(self.labels[0])
        )


class Volumetry(pc.BaseModel):
    """The volumetry chart's settings, as they should stand at load time.

    Extras are forbidden, as they are on ``Scene``: a misspelled key in a
    hand-written config should say so rather than quietly do nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    segmentation_label: str = pc.Field(
        default="",
        description=(
            "Segmentation the volumes are measured off; empty means the first "
            'one. CLI usage: --volumetry.segmentation_label "seg"'
        ),
    )
    groups: list[StructureGroup] = pc.Field(
        default_factory=list,
        description=(
            "What to measure, one entry per curve. Declared rather than "
            "discovered: which labels make up a chamber is a clinical fact "
            "about the segmentation, not something an image can be asked."
        ),
    )
    indexed: bool = pc.Field(
        default=True,
        description=(
            "Whether to also report volumes divided by body surface area, when "
            "a height and a weight are known. CLI usage: --volumetry.indexed false"
        ),
    )
    bsa_formula: BSAFormula = pc.Field(
        default=BSAFormula.MOSTELLER,
        description=(
            "How the body surface area is derived. Options: "
            + ", ".join(BSAFormula)
            + ". CLI usage: --volumetry.bsa_formula dubois"
        ),
    )
    patient_height_m: float | None = pc.Field(
        default=None,
        gt=0.0,
        description=(
            "Height in metres, overriding the DICOM PatientSize tag. "
            "CLI usage: --volumetry.patient_height_m 1.78"
        ),
    )
    patient_weight_kg: float | None = pc.Field(
        default=None,
        gt=0.0,
        description=(
            "Weight in kilograms, overriding the DICOM PatientWeight tag. "
            "CLI usage: --volumetry.patient_weight_kg 74"
        ),
    )

    @pc.model_validator(mode="after")
    def validate_unique_names(self):
        """Two groups by one name are two curves nothing can tell apart."""
        names = [group.name for group in self.groups]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate volumetry group names: {duplicates}")
        return self


def bsa_mosteller(height_m: float, weight_kg: float) -> float:
    """Mosteller's body surface area, in square metres.

    Note the unit: the formula is written in centimetres, and DICOM's
    ``PatientSize`` is in metres.  Passing the tag through unconverted gives an
    area out by a factor of ten, which is the bug the prototype script carried.
    """
    return math.sqrt(height_m * 100.0 * weight_kg / 3600.0)


def bsa_dubois(height_m: float, weight_kg: float) -> float:
    """Du Bois and Du Bois' body surface area, in square metres."""
    return 0.007184 * (height_m * 100.0) ** 0.725 * weight_kg**0.425


def bsa_haycock(height_m: float, weight_kg: float) -> float:
    """Haycock's body surface area, in square metres."""
    return 0.024265 * (height_m * 100.0) ** 0.3964 * weight_kg**0.5378


BSA_FORMULAS = {
    BSAFormula.MOSTELLER: bsa_mosteller,
    BSAFormula.DUBOIS: bsa_dubois,
    BSAFormula.HAYCOCK: bsa_haycock,
}


def body_surface_area(
    formula: BSAFormula, height_m: float | None, weight_kg: float | None
) -> float | None:
    """The area the formula gives, or None if it cannot be had.

    None rather than a default, because an indexed volume computed against a
    body somebody else's size is worse than no indexed volume at all.
    """
    if formula is BSAFormula.NONE or not height_m or not weight_kg:
        return None
    return BSA_FORMULAS[BSAFormula(formula)](float(height_m), float(weight_kg))


def body_size(header: dict[str, str]) -> tuple[float | None, float | None]:
    """The height in metres and weight in kilograms a DICOM header carries.

    Either may be absent, empty, or unparseable -- a header is whatever the
    writer put in it -- so each is read on its own and missing rather than
    guessed.
    """

    def number(tag: str) -> float | None:
        try:
            value = float(header.get(tag, "") or 0.0)
        except (TypeError, ValueError):
            return None
        return value or None

    return number("PatientSize"), number("PatientWeight")


def voxel_volume_ml(image) -> float:
    """How much space one voxel of ``image`` stands for, in millilitres."""
    return float(np.prod(image.GetSpacing())) / MM3_PER_ML


def group_amount(
    counts: dict[int, int], group: StructureGroup, voxel_ml: float
) -> float:
    """One frame's measurement of one structure, in its own unit.

    A label the frame does not carry contributes nothing rather than raising:
    a chamber can empty of a sub-label between phases, and a group naming a
    label the segmentation never had is reported once by ``measure`` instead.
    """
    volume_ml = sum(counts.get(label, 0) for label in group.labels) * voxel_ml
    return volume_ml * group.density if group.density is not None else volume_ml


@dc.dataclass(frozen=True)
class Metrics:
    """What a curve comes to: its extremes, and the difference between them.

    Named for the curve rather than for the heart -- ``maximum`` is what a
    reader calls end-diastolic only once they know the structure fills and
    empties.  ``stroke`` and ``ejection`` are None for a structure that does
    not, where the difference between extremes is shape rather than flow.
    """

    maximum: float
    minimum: float
    maximum_frame: int
    minimum_frame: int
    stroke: float | None
    ejection: float | None


def metrics(values: np.ndarray, cycle: bool = True) -> Metrics:
    """The extremes of a curve, and the stroke and ejection they imply."""
    values = np.asarray(values, dtype=np.float64)
    maximum, minimum = float(values.max()), float(values.min())

    stroke = maximum - minimum if cycle else None
    ejection = stroke / maximum * 100.0 if stroke is not None and maximum else None

    return Metrics(
        maximum=maximum,
        minimum=minimum,
        maximum_frame=int(values.argmax()),
        minimum_frame=int(values.argmin()),
        stroke=stroke,
        ejection=ejection,
    )


@dc.dataclass(frozen=True)
class Measurement:
    """One structure's curve, and what it comes to."""

    group: StructureGroup
    values: np.ndarray
    metrics: Metrics


@dc.dataclass(frozen=True)
class Result:
    """Every structure measured off one segmentation, over every frame."""

    measurements: list[Measurement]
    bsa: float | None
    frames: int

    def of(self, name: str) -> Measurement | None:
        return next((m for m in self.measurements if m.group.name == name), None)

    @property
    def names(self) -> list[str]:
        return [m.group.name for m in self.measurements]


def measure(
    counts_by_frame: list[dict[int, int]],
    voxel_ml: float,
    groups: list[StructureGroup],
    bsa: float | None = None,
) -> Result:
    """Every group's curve over the frames whose label counts are given.

    Takes counts rather than images, so the measurement can be tested and the
    counting cached independently of it.  A group naming a label no frame
    carries is reported here, once, rather than at every frame: a mistyped
    label id is a config error, and measuring zero in silence is how it would
    otherwise be discovered.
    """
    present = {label for counts in counts_by_frame for label in counts}

    measurements = []
    for group in groups:
        missing = sorted(set(group.labels) - present)
        if missing:
            logger.warning(
                f"Volumetry group {group.name!r} names labels the segmentation "
                f"does not carry: {missing}."
            )

        values = np.array(
            [group_amount(counts, group, voxel_ml) for counts in counts_by_frame],
            dtype=np.float64,
        )
        measurements.append(
            Measurement(
                group=group, values=values, metrics=metrics(values, group.cycle)
            )
        )

    return Result(measurements=measurements, bsa=bsa, frames=len(counts_by_frame))


# A page's table reads down rather than across: one row per metric, so that a
# metric added later costs a row on every page instead of a column on a table
# that is already as wide as the page.
PAGE_COLUMNS = ("Metric", "Value")
INDEXED_COLUMN = "Indexed"

# What a cell says when the number behind it would mean nothing: a structure
# outside the cardiac cycle has no stroke volume, and an ejection fraction is
# already a ratio, so dividing one by a body surface area says nothing.
ABSENT = "--"

# Indexed quantities are per square metre of body surface.
PER_BSA = "/m\u00b2"


def number(value: float | None, digits: int = 1) -> str:
    """One number, or the mark that says it would mean nothing."""
    return ABSENT if value is None else f"{value:.{digits}f}"


def quantity(value: float | None, unit: str) -> str:
    """A number with the unit it is in, which on a pivoted table varies by row."""
    return ABSENT if value is None else f"{number(value)} {unit}"


def page_table(
    measurement: Measurement, bsa: float | None = None
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """One structure's metrics, a row each, ready to be shown.

    Formatted here rather than by whatever is showing it, so that the table
    painted beside the chart and the one listed in the drawer cannot come to
    disagree about a rounding or a unit.

    The rows are named for the heart -- EDV, ESV -- where the measurement
    itself is named for the curve. This is where that translation belongs: a
    maximum is only an end-diastolic volume to a reader who knows the structure
    fills and empties.
    """
    group, found = measurement.group, measurement.metrics
    indexed_unit = group.unit + PER_BSA

    # Each row: what it is called, what it reads, and the quantity an indexed
    # column would divide -- None where indexing it would say nothing.
    entries = [
        ("EDV", quantity(found.maximum, group.unit), found.maximum),
        ("ESV", quantity(found.minimum, group.unit), found.minimum),
        ("SV", quantity(found.stroke, group.unit), found.stroke),
        ("EF", quantity(found.ejection, "%"), None),
    ]

    if not bsa:
        return PAGE_COLUMNS, [(name, value) for name, value, _ in entries]

    return (*PAGE_COLUMNS, INDEXED_COLUMN), [
        (name, value, quantity(None if source is None else source / bsa, indexed_unit))
        for name, value, source in entries
    ]


def timeseries(result: Result) -> tuple[list[str], list[list]]:
    """The curves themselves: one row per frame, one column per structure.

    Written alongside the metrics because the two answer different questions.
    The metrics say what the study came to; this says what it did, and is what
    somebody re-plotting or re-analysing the series actually needs.
    """
    header = ["frame"] + [f"{m.group.name}_{m.group.unit}" for m in result.measurements]
    rows = [
        [frame] + [float(m.values[frame]) for m in result.measurements]
        for frame in range(result.frames)
    ]
    return header, rows


def metrics_table(result: Result) -> tuple[list[str], list[list]]:
    """What each structure came to, as numbers rather than as formatted text.

    Unlike ``table``, which is for reading: an empty cell here rather than a
    dash, so that whatever opens the file sees a missing value and not a string
    in a column of numbers.
    """
    header = ["structure", "unit", "EDV", "ESV", "SV", "EF_pct"]
    if result.bsa:
        header.extend(["BSA_m2", "EDVi", "ESVi", "SVi"])

    rows = []
    for measurement in result.measurements:
        group, found = measurement.group, measurement.metrics
        row = [
            group.name,
            group.unit,
            found.maximum,
            found.minimum,
            found.stroke,
            found.ejection,
        ]
        if result.bsa:
            row.append(result.bsa)
            row.extend(
                None if value is None else value / result.bsa
                for value in (found.maximum, found.minimum, found.stroke)
            )
        rows.append(row)

    return header, rows
