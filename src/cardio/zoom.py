"""The zoom selection: which labels the MPR views are fitted to."""

# System
import logging

# Third Party
import pydantic as pc

# Internal
from .reslice import VIEWS

logger = logging.getLogger(__name__)


class Zoom(pc.BaseModel):
    """The zoom panel's selection, as it should stand at load time.

    A selection of its own rather than a reading of the snap groups: the labels
    worth framing are not always the labels worth centring on, and a control
    that changed meaning with the snap mode would be a different control in each
    of them.

    Extras are forbidden, as they are on ``Scene``: a misspelled key in a
    hand-written config should say so rather than quietly do nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    segmentation_label: str = pc.Field(
        default="",
        description="Segmentation the labels index into; empty selects the first.",
    )
    labels: list[int] = pc.Field(
        default_factory=list,
        description='Labels the views are fitted to. CLI usage: --zoom.labels "[1,2]"',
    )
    plane: str = pc.Field(
        default="ul",
        description="Plane the labels are projected onto. CLI usage: --zoom.plane ll",
    )
    fill: int = pc.Field(
        default=80,
        ge=10,
        le=100,
        description="How much of the viewport the labels should fill, as a percentage.",
    )
    locked: bool = pc.Field(
        default=False,
        description="Hold the fit as the origin and the orientation move under it.",
    )

    @pc.field_validator("plane")
    @classmethod
    def plane_is_a_view(cls, plane: str) -> str:
        """The three plane names are the reslice's to say, not this model's."""
        if plane not in VIEWS:
            raise ValueError(f"Unknown plane {plane!r}; expected one of {list(VIEWS)}.")
        return plane

    @pc.model_validator(mode="after")
    def warn_ineffective_lock(self) -> "Zoom":
        """Flag a lock that cannot take effect, rather than refusing to launch.

        An empty selection is a working state in the panel -- it simply does not
        fit anything -- so a config that asks to lock one is a warning, the way
        an incomplete snap selection is.
        """
        if self.locked and not self.labels:
            logger.warning("Zoom lock requested, but no labels are selected.")
        return self
