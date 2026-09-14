"""The tile grid: how many cuts, and what path they are taken along."""

# System
import enum

# Third Party
import pydantic as pc

# Internal
from .reslice import VIEWS
from .tile_views import MAX_COLS, MAX_ROWS


class TileSource(str, enum.Enum):
    """What path the tiles are taken along.

    A ``str`` enum so that comparisons against the trame state variable, which
    carries the source as a plain string, keep working unchanged.
    """

    TRAVERSE = "traverse"
    SPACING = "spacing"
    LABELS = "labels"


class Tile(pc.BaseModel):
    """The tile panel's settings, as they should stand at load time.

    Extras are forbidden, as they are on ``Scene``: a misspelled key in a
    hand-written config should say so rather than quietly do nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    rows: int = pc.Field(
        default=3,
        ge=1,
        le=MAX_ROWS,
        description="Rows in the tile view grid. CLI usage: --tile.rows 2",
    )
    cols: int = pc.Field(
        default=3,
        ge=1,
        le=MAX_COLS,
        description="Columns in the tile view grid. CLI usage: --tile.cols 4",
    )
    source: TileSource = pc.Field(
        default=TileSource.TRAVERSE,
        description=(
            "What path the tiles are taken along. Options: "
            + ", ".join(TileSource)
            + ". CLI usage: --tile.source spacing"
        ),
    )
    plane: str = pc.Field(
        default="ul",
        description=(
            "Plane the parallel sources cut in, and step along the normal of. "
            "CLI usage: --tile.plane ll"
        ),
    )
    spacing: float = pc.Field(
        default=10.0,
        gt=0.0,
        description=(
            "Millimetres between adjacent cuts, in the spacing source. "
            "CLI usage: --tile.spacing 5"
        ),
    )
    reverse: bool = pc.Field(
        default=False,
        description=(
            "Take the tiles in the other order, without turning the cut. "
            "CLI usage: --tile.reverse true"
        ),
    )
    segmentation_label: str = pc.Field(
        default="",
        description="Segmentation the labels index into; empty selects the first.",
    )
    labels: list[int] = pc.Field(
        default_factory=list,
        description=(
            "Labels the labels source spans, end to end. "
            'CLI usage: --tile.labels "[1,2]"'
        ),
    )

    @pc.field_validator("plane")
    @classmethod
    def plane_is_a_view(cls, plane: str) -> str:
        """The three plane names are the reslice's to say, not this model's."""
        if plane not in VIEWS:
            raise ValueError(f"Unknown plane {plane!r}; expected one of {list(VIEWS)}.")
        return plane
