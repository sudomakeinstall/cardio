"""The snap selection: which segmentation feature the MPR views lock onto."""

# System
import enum
import logging

# Third Party
import pydantic as pc

logger = logging.getLogger(__name__)


class SnapMode(str, enum.Enum):
    """Which feature of a segmentation the MPR views snap to.

    A ``str`` enum so that comparisons against the trame state variable, which
    carries the mode as a plain string, keep working unchanged.
    """

    LABEL = "label"
    INTERFACE = "interface"
    TRAVERSE = "traverse"


# Modes that fit a plane, and so can align and lock orientation.
PLANAR_MODES = (SnapMode.INTERFACE, SnapMode.TRAVERSE)


class Snap(pc.BaseModel):
    """The snap panel's selection, as it should stand at load time.

    Reset returns here too, so this is the selection for a dataset rather than
    merely its starting point.

    Extras are forbidden, as they are on ``Scene``: a misspelled key in a
    hand-written config should say so rather than quietly do nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    segmentation_label: str = pc.Field(
        default="",
        description="Segmentation the label groups index into; empty selects the first.",
    )
    mode: SnapMode = pc.Field(
        default=SnapMode.LABEL,
        description="Snap mode. CLI usage: --snap.mode traverse",
    )
    labels_a: list[int] = pc.Field(
        default_factory=list,
        description='Group A, which every mode needs. CLI usage: --snap.labels_a "[1,2]"',
    )
    labels_b: list[int] = pc.Field(
        default_factory=list,
        description="Group B, which the interface and traverse modes need.",
    )
    labels_c: list[int] = pc.Field(
        default_factory=list,
        description="Group C, which traverse mode needs.",
    )
    traverse: int = pc.Field(
        default=0,
        ge=0,
        le=100,
        description="Position along the traverse path, as a percentage from A|B to B|C.",
    )
    locked: bool = pc.Field(
        default=False,
        description="Hold the MPR origin at the snapped centroid on every frame.",
    )
    orientation_locked: bool = pc.Field(
        default=False,
        description="Hold the MPR views aligned to the fitted plane on every frame.",
    )

    @property
    def groups(self) -> tuple[list[int], list[int], list[int]]:
        """The three label groups, in A, B, C order."""
        return (self.labels_a, self.labels_b, self.labels_c)

    @property
    def required_groups_chosen(self) -> bool:
        """Whether every group this mode needs has at least one label."""
        if not self.labels_a:
            return False
        if self.mode in PLANAR_MODES and not self.labels_b:
            return False
        # Kept as a third guard rather than folded into the return, which would
        # turn "traverse needs C" into a double negative and break the parallel.
        if self.mode is SnapMode.TRAVERSE and not self.labels_c:  # noqa: SIM103
            return False
        return True

    @pc.model_validator(mode="after")
    def warn_ineffective_locks(self) -> "Snap":
        """Flag a lock that cannot take effect, rather than refusing to launch.

        An incomplete selection is a working state in the panel -- it simply
        does not snap -- so a config that asks for one is a warning, not an
        error.
        """
        if (self.locked or self.orientation_locked) and not self.required_groups_chosen:
            logger.warning(
                f"Snap lock requested, but {self.mode.value} mode's label groups are incomplete."
            )
        if self.orientation_locked and self.mode not in PLANAR_MODES:
            logger.warning(
                f"Snap orientation lock requested, but {self.mode.value} mode does not fit a plane."
            )
        return self
