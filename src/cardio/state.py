"""The trame state variable names belonging to each scene object.

Logic registers and reads these; the UI binds them. Both used to build the same
f-strings independently, so a mismatch was a silent no-op rather than an error.
Spelling each key in one place makes that impossible.
"""

# System
import dataclasses as dc

THEME_DARK = "dark"
THEME_LIGHT = "light"
DEFAULT_THEME_MODE = THEME_DARK


@dc.dataclass(frozen=True)
class ObjectState:
    """The state keys for one renderable object.

    ``kind`` distinguishes the per-type keys (a mesh and a volume may share a
    label); the clip keys are keyed by label alone, as they always have been.
    """

    kind: str
    label: str

    @classmethod
    def of(cls, obj) -> "ObjectState":
        return cls(kind=obj.kind, label=obj.label)

    @property
    def visibility(self) -> str:
        return f"{self.kind}_visibility_{self.label}"

    @property
    def clipping(self) -> str:
        return f"{self.kind}_clipping_{self.label}"

    @property
    def clip_panel(self) -> str:
        return f"clip_panel_{self.label}"

    @property
    def clip_x(self) -> str:
        return f"clip_x_{self.label}"

    @property
    def clip_y(self) -> str:
        return f"clip_y_{self.label}"

    @property
    def clip_z(self) -> str:
        return f"clip_z_{self.label}"

    @property
    def clip_bounds(self) -> tuple[str, str, str]:
        """The three range-slider keys, in x, y, z order."""
        return (self.clip_x, self.clip_y, self.clip_z)

    @property
    def clip_controls(self) -> list[str]:
        """Every key a change to which should re-apply this object's clipping."""
        return [self.clipping, *self.clip_bounds]

    @property
    def preset(self) -> str:
        return f"volume_preset_{self.label}"

    @property
    def preset_panel(self) -> str:
        return f"preset_panel_{self.label}"

    @property
    def mpr_overlay(self) -> str:
        return f"mpr_segmentation_overlay_{self.label}"
