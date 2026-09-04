# System
import datetime as dt
import pathlib as pl
import typing as ty

# Third Party
import numpy as np
import pydantic as pc
import tomlkit as tk

# Internal
from .convention import (
    exchange_angle,
    exchange_axis,
    exchange_point,
    exchange_quaternion,
)
from .orientation import (
    AngleUnits,
    EulerAxis,
    IndexOrder,
    euler_angle_to_rotation_matrix,
    quaternion_to_rotation_matrix,
)


class RotationStep(pc.BaseModel):
    """Single rotation step.

    Either (axis + angle) for Euler steps, or quaternion [x, y, z, w] for quaternion steps.
    Euler: angle stored in units from parent metadata.angle_units,
           axis stored in convention from parent metadata.index_order.
    Quaternion: fixed, not user-editable.
    """

    axis: ty.Literal["X", "Y", "Z"] | None = None
    angle: float | None = None
    quaternion: list[float] | None = None
    visible: bool = True
    name: str = ""
    name_editable: bool = True
    deletable: bool = True

    @pc.model_validator(mode="after")
    def validate_rotation_form(self) -> "RotationStep":
        has_euler = self.axis is not None
        has_quat = self.quaternion is not None
        if has_euler == has_quat:
            raise ValueError(
                "RotationStep requires exactly one of: (axis) or (quaternion)"
            )
        if has_quat and len(self.quaternion) != 4:
            raise ValueError("quaternion must be a 4-element list [x, y, z, w]")
        if has_euler and self.angle is None:
            self.angle = 0.0
        return self

    def to_rotation_matrix(self, units: AngleUnits) -> np.ndarray:
        """Return 3x3 rotation matrix for this step."""
        if self.quaternion is not None:
            return quaternion_to_rotation_matrix(self.quaternion)
        return euler_angle_to_rotation_matrix(EulerAxis(self.axis), self.angle, units)


class RotationMetadata(pc.BaseModel):
    """Metadata for TOML files."""

    coordinate_system: ty.Literal["LPS"] = "LPS"
    index_order: IndexOrder = IndexOrder.ITK
    angle_units: AngleUnits = AngleUnits.RADIANS
    timestamp: str = pc.Field(
        default_factory=lambda: dt.datetime.now().astimezone().isoformat()
    )
    volume_label: str = ""
    deletable: bool = True


class RotationSequence(pc.BaseModel):
    """Complete rotation sequence.

    All data (angle, axis, and origin) are stored in the current convention/units:
    - Angle: stored in units specified by metadata.angle_units
    - Axis: stored in convention specified by metadata.index_order
    - Origin: stored in axis order specified by metadata.index_order

    When convention/units change in the UI, all existing data is converted.
    """

    model_config = pc.ConfigDict(frozen=False)

    metadata: RotationMetadata = pc.Field(default_factory=RotationMetadata)
    angles_list: list[RotationStep] = pc.Field(default_factory=list)
    mpr_origin: list[float] = pc.Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="MPR origin position [x, y, z] in current index_order convention",
    )

    @pc.field_validator("mpr_origin")
    @classmethod
    def validate_mpr_origin(cls, v):
        """Ensure mpr_origin is a 3-element list of floats."""
        if not isinstance(v, list) or len(v) != 3:
            raise ValueError("mpr_origin must be a 3-element list [x, y, z]")
        return [float(x) for x in v]

    def with_units(self, units: AngleUnits) -> "RotationSequence":
        """This sequence with its Euler angles re-expressed in ``units``.

        Quaternion steps carry no angle, so they are left alone.
        """
        if units == self.metadata.angle_units:
            return self

        converted = self.model_copy(deep=True)
        for step in converted.angles_list:
            if step.quaternion is not None:
                continue
            match units:
                case AngleUnits.RADIANS:
                    step.angle = float(np.radians(step.angle))
                case AngleUnits.DEGREES:
                    step.angle = float(np.degrees(step.angle))
        converted.metadata.angle_units = units
        return converted

    def with_index_order(self, index_order: IndexOrder) -> "RotationSequence":
        """This sequence re-expressed in ``index_order``.

        The origin moves with the steps: all three are stored in whichever order
        the metadata names, so they have to change together or the stored
        numbers would keep their values while changing meaning.
        """
        if index_order == self.metadata.index_order:
            return self

        converted = self.model_copy(deep=True)
        for step in converted.angles_list:
            if step.quaternion is not None:
                step.quaternion = exchange_quaternion(step.quaternion)
            else:
                step.axis = exchange_axis(step.axis)
                step.angle = exchange_angle(step.angle)
        converted.mpr_origin = exchange_point(converted.mpr_origin)
        converted.metadata.index_order = index_order
        return converted

    def to_toml(self) -> str:
        """Serialize to TOML using stored serialization preferences."""
        # Deferred: cardio/__init__ imports this module, so a module-scope
        # import here would be circular.
        from . import __version__

        data = self.model_dump(mode="json", exclude_none=True)
        toml_str = tk.dumps(data)

        version_comment = f"# Generated by cardio version {__version__}\n\n"
        return version_comment + toml_str

    @classmethod
    def from_toml(cls, toml_content: str) -> "RotationSequence":
        """Deserialize from TOML (no conversions - loads as-is)."""
        doc = tk.loads(toml_content)
        data = dict(doc)
        return cls(**data)

    @classmethod
    def from_file(cls, path: pl.Path) -> "RotationSequence":
        """Load from TOML file."""
        with open(path, "r") as f:
            return cls.from_toml(f.read())

    def to_file(self, path: pl.Path):
        """Save to TOML file using stored serialization preferences."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_toml())
