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
    chamber: bool = pc.Field(
        default=True,
        description=(
            "Whether this structure is a pumping chamber, whose extremes are "
            "an end-diastolic and an end-systolic volume and whose difference "
            "between them is a stroke volume. False for everything else -- "
            "myocardium, a great vessel, anything outside the heart -- whose "
            "extremes are reported as a maximum and a minimum and nothing more."
        ),
    )

    @pc.field_validator("density", mode="before")
    @classmethod
    def blank_density_is_unset(cls, value):
        """An emptied number field leaves a blank string, which means unset.

        The drawer's density field starts empty and can be cleared again, and
        what a cleared one hands back is "" -- or a space, if that is what was
        left behind -- rather than nothing at all.
        """
        return None if isinstance(value, str) and not value.strip() else value

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


# The drawer edits the structures through trame's deep-reactive wrapper, which
# mirrors a state variable into a plain object -- ``reactive({})``, assigned
# over.  Handed a bare list it mirrors ``{"0": ..., "1": ...}``, which has no
# length to test and renders as nothing at all, and pushes that shape back as
# the new value.  So what it is handed is an object with the list inside it,
# which is why the rotation sequence works and why this is spelled the same way.
STRUCTURES = "structures"


def structure_state(groups: list[StructureGroup]) -> dict:
    """The structures as the drawer holds them."""
    return {STRUCTURES: [group.model_dump(mode="json") for group in groups]}


def structure_rows(held) -> list:
    """The rows inside what the drawer holds, however little of it there is."""
    return (held or {}).get(STRUCTURES) or []


# The fields a form may leave in a state the model refuses without the row
# ceasing to be a structure.  Derived rather than listed, so that an optional
# field added later is covered without anyone having to remember this.
OPTIONAL_FIELDS = frozenset(
    name for name, field in StructureGroup.model_fields.items() if field.default is None
)


def as_structure(row) -> StructureGroup | None:
    """``row`` as a structure, ignoring the optional fields it got wrong.

    A row with a name and labels is a structure whatever else has been typed
    into it, so a density of ``0`` unsets the density rather than deleting the
    curve.  Deleting it is what this used to do, and typing ``0.95`` passes
    through ``0`` on the way -- so a structure disappeared and came back
    mid-keystroke, with nothing said anywhere about why.  What says why is the
    rule under the field; nothing is logged here, because this runs on every
    keystroke and a log of half-typed numbers is not a log.
    """
    try:
        return StructureGroup.model_validate(row)
    except pc.ValidationError as refused:
        wrong = {
            error["loc"][0] if error["loc"] else None for error in refused.errors()
        }
        if not wrong <= OPTIONAL_FIELDS:
            return None

    try:
        return StructureGroup.model_validate({**row, **dict.fromkeys(wrong)})
    except pc.ValidationError:
        return None


def usable_groups(rows) -> list[StructureGroup]:
    """The structures among ``rows`` that are ready to be measured.

    A row being edited is not yet a structure: it has no labels picked, no name
    typed, or the name of one above it.  The models refuse all three, which is
    right for a config file and wrong for a half-filled form -- so this is the
    seam between the two, and the drawer can hold a blank row without the app
    having to pretend it means something.
    """
    usable: list[StructureGroup] = []
    taken: set[str] = set()

    for row in rows:
        group = as_structure(row)
        if group is None:
            continue

        name = group.name.strip()
        if not name or name in taken:
            continue

        taken.add(name)
        usable.append(group.model_copy(update={"name": name}))

    return usable


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

    Named for the curve rather than for the heart, because that is all the
    measurement knows: nothing here was told what phase anything was acquired
    at.  ``maximum`` becomes an end-diastolic volume only where a reader has
    declared the structure a pumping chamber, which is what ``page_table``
    does and this does not.  ``stroke`` and ``ejection`` are None otherwise,
    where the difference between extremes is shape or pulsation rather than
    ejection.
    """

    maximum: float
    minimum: float
    maximum_frame: int
    minimum_frame: int
    stroke: float | None
    ejection: float | None


def metrics(values: np.ndarray, chamber: bool = True) -> Metrics:
    """The extremes of a curve, and the stroke and ejection they imply."""
    values = np.asarray(values, dtype=np.float64)
    maximum, minimum = float(values.max()), float(values.min())

    stroke = maximum - minimum if chamber else None
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
                group=group, values=values, metrics=metrics(values, group.chamber)
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

# What the extremes of a curve are called, by whether the structure has been
# declared a pumping chamber. A maximum is only an end-diastolic volume to a
# reader who has said the structure fills and empties over a cardiac cycle;
# for a great vessel or a lung it is a maximum, and there is no stroke volume
# or ejection fraction under it.
CHAMBER_ROWS = ("EDV", "ESV")
EXTREME_ROWS = ("Maximum", "Minimum")


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

    This is the one place the cardiac vocabulary is used, and it is used only
    where a reader has declared the structure a pumping chamber.  Nothing in
    the measurement knows what phase anything was acquired at, or that the
    series covered a cycle at all: a maximum is an end-diastolic volume by
    somebody's say-so, and for an aorta or a lung it is a maximum and nothing
    else.
    """
    group, found = measurement.group, measurement.metrics
    indexed_unit = group.unit + PER_BSA
    largest, smallest = CHAMBER_ROWS if group.chamber else EXTREME_ROWS

    # Each row: what it is called, what it reads, and the quantity an indexed
    # column would divide -- None where indexing it would say nothing.
    entries = [
        (largest, quantity(found.maximum, group.unit), found.maximum),
        (smallest, quantity(found.minimum, group.unit), found.minimum),
    ]

    # No rows at all rather than two rows of dashes: a stroke volume is what a
    # chamber has, and printing the absence of one under an aorta invites the
    # reader to wonder which of the two it is.
    if group.chamber:
        entries += [
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

    Unlike ``page_table``, which is for reading: an empty cell here rather than
    a dash, so that whatever opens the file sees a missing value and not a
    string in a column of numbers -- and the columns are named for the curve
    rather than for the heart, because one table holds every structure.
    """
    # Neutral names, unlike the page's: one table holds every structure, so a
    # column called EDV would carry a ventricle's end-diastolic volume and an
    # aorta's maximum in the same place.
    header = ["structure", "unit", "maximum", "minimum", "stroke", "ejection_pct"]
    if result.bsa:
        header.extend(["BSA_m2", "maximum_i", "minimum_i", "stroke_i"])

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
