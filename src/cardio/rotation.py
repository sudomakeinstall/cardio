# System
import datetime as dt
import pathlib as pl
import typing as ty

# Third Party
import numpy as np
import pydantic as pc
import tomlkit as tk

# Internal
from .orientation import AngleUnits, AxisConvention


class RotationStep(pc.BaseModel):
    """Single rotation step (stored in ITK convention, radians)."""

    axes: ty.Literal["X", "Y", "Z"]
    angle: float = 0.0
    visible: bool = True
    name: str = ""
    name_editable: bool = True
    deletable: bool = True

    @pc.model_validator(mode="before")
    @classmethod
    def handle_legacy_format(cls, data):
        """Handle legacy 'angles' list format and 'axis' field."""
        if isinstance(data, dict):
            if "angles" in data and "angle" not in data:
                angles = data["angles"]
                if isinstance(angles, list) and len(angles) > 0:
                    data["angle"] = angles[0]
                else:
                    data["angle"] = angles
            if "axis" in data and "axes" not in data:
                data["axes"] = data["axis"]
        return data

    def to_convention(
        self, convention: AxisConvention, units: AngleUnits
    ) -> tuple[str, float]:
        """Convert to target convention/units for serialization."""
        axis = self.axes
        angle = self.angle

        if convention == AxisConvention.ROMA:
            axis = {"X": "Z", "Y": "Y", "Z": "X"}[axis]
            angle = -angle

        if units == AngleUnits.DEGREES:
            angle = np.degrees(angle)

        return axis, angle

    @classmethod
    def from_convention(
        cls,
        axes: str,
        angle: float,
        convention: AxisConvention,
        units: AngleUnits,
        **kwargs,
    ) -> "RotationStep":
        """Create from target convention/units (for deserialization)."""
        if units == AngleUnits.DEGREES:
            angle = np.radians(angle)

        if convention == AxisConvention.ROMA:
            axes = {"X": "Z", "Y": "Y", "Z": "X"}[axes]
            angle = -angle

        return cls(axes=axes, angle=angle, **kwargs)


class RotationMetadata(pc.BaseModel):
    """Metadata for TOML files."""

    coordinate_system: ty.Literal["LPS"] = "LPS"
    axis_convention: AxisConvention = AxisConvention.ITK
    units: AngleUnits = AngleUnits.RADIANS
    timestamp: str = pc.Field(default_factory=lambda: dt.datetime.now().isoformat())
    volume_label: str = ""


class RotationSequence(pc.BaseModel):
    """Complete rotation sequence (stored in ITK convention, radians)."""

    model_config = pc.ConfigDict(frozen=False)

    metadata: RotationMetadata = pc.Field(default_factory=RotationMetadata)
    angles_list: list[RotationStep] = pc.Field(default_factory=list)
    mpr_origin: list[float] = pc.Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="MPR origin position [x, y, z] in LPS coordinates",
    )

    @pc.field_validator("mpr_origin")
    @classmethod
    def validate_mpr_origin(cls, v):
        """Ensure mpr_origin is a 3-element list of floats."""
        if not isinstance(v, list) or len(v) != 3:
            raise ValueError("mpr_origin must be a 3-element list [x, y, z]")
        return [float(x) for x in v]

    @pc.field_validator("angles_list", mode="before")
    @classmethod
    def convert_legacy_list(cls, v):
        """Convert legacy list of dicts to list of RotationStep."""
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            return [RotationStep(**item) for item in v]
        return v

    def to_dict_for_ui(self) -> dict:
        """Convert to UI format (always ITK/radians internally).

        Note: UI expects 'angles' as a list for backward compatibility.
        """
        return {
            "angles_list": [
                {
                    "axes": step.axes,
                    "angles": [step.angle],
                    "visible": step.visible,
                    "name": step.name,
                    "name_editable": step.name_editable,
                    "deletable": step.deletable,
                }
                for step in self.angles_list
            ],
            "mpr_origin": self.mpr_origin,
        }

    @classmethod
    def from_ui_dict(cls, data: dict, volume_label: str = "") -> "RotationSequence":
        """Create from UI format (assumes ITK/radians)."""
        angles_list = [RotationStep(**step) for step in data.get("angles_list", [])]
        metadata = RotationMetadata(volume_label=volume_label)
        mpr_origin = data.get("mpr_origin", [0.0, 0.0, 0.0])
        return cls(metadata=metadata, angles_list=angles_list, mpr_origin=mpr_origin)

    def to_toml(
        self, target_convention: AxisConvention, target_units: AngleUnits
    ) -> str:
        """Serialize to TOML with conversions."""
        doc = tk.document()

        metadata_table = tk.table()
        metadata_table["coordinate_system"] = self.metadata.coordinate_system
        metadata_table["axis_convention"] = target_convention.value
        metadata_table["units"] = target_units.value
        metadata_table["timestamp"] = self.metadata.timestamp
        metadata_table["volume_label"] = self.metadata.volume_label
        doc["metadata"] = metadata_table

        origin_table = tk.table()
        origin_table["x"] = float(self.mpr_origin[0])
        origin_table["y"] = float(self.mpr_origin[1])
        origin_table["z"] = float(self.mpr_origin[2])
        doc["mpr_origin"] = origin_table

        rotations_array = tk.aot()
        for step in self.angles_list:
            rotation = tk.table()
            converted_axis, converted_angle = step.to_convention(
                target_convention, target_units
            )
            rotation["axes"] = converted_axis
            rotation["angle"] = converted_angle
            rotation["visible"] = step.visible
            rotation["name"] = step.name
            rotation["name_editable"] = step.name_editable
            rotation["deletable"] = step.deletable
            rotations_array.append(rotation)

        doc["angles_list"] = rotations_array

        return tk.dumps(doc)

    @classmethod
    def from_toml(cls, toml_content: str) -> "RotationSequence":
        """Deserialize from TOML with conversions."""
        doc = tk.loads(toml_content)

        metadata_dict = doc.get("metadata", {})
        convention = AxisConvention(metadata_dict.get("axis_convention", "itk"))
        units = AngleUnits(metadata_dict.get("units", "radians"))

        metadata = RotationMetadata(
            coordinate_system=metadata_dict.get("coordinate_system", "LPS"),
            axis_convention=convention,
            units=units,
            timestamp=metadata_dict.get("timestamp", dt.datetime.now().isoformat()),
            volume_label=metadata_dict.get("volume_label", ""),
        )

        angles_list_data = doc.get("angles_list", [])
        angles_list = []
        for item in angles_list_data:
            axes = item.get("axes", "X")
            angle = item.get("angle", 0.0)
            step = RotationStep.from_convention(
                axes=axes,
                angle=angle,
                convention=convention,
                units=units,
                visible=item.get("visible", True),
                name=item.get("name", ""),
                name_editable=item.get("name_editable", True),
                deletable=item.get("deletable", True),
            )
            angles_list.append(step)

        origin_dict = doc.get("mpr_origin", {})
        mpr_origin = [
            float(origin_dict.get("x", 0.0)),
            float(origin_dict.get("y", 0.0)),
            float(origin_dict.get("z", 0.0)),
        ]

        return cls(metadata=metadata, angles_list=angles_list, mpr_origin=mpr_origin)

    @classmethod
    def from_file(cls, path: pl.Path) -> "RotationSequence":
        """Load from TOML file."""
        with open(path, "r") as f:
            return cls.from_toml(f.read())

    def to_file(
        self,
        path: pl.Path,
        target_convention: AxisConvention,
        target_units: AngleUnits,
    ):
        """Save to TOML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_toml(target_convention, target_units))
