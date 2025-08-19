import re
import pathlib as pl
import pydantic as pc
from vtkmodules.vtkRenderingCore import vtkRenderer


class Object(pc.BaseModel):
    """Base class for renderable objects with validated configuration."""

    model_config = pc.ConfigDict(arbitrary_types_allowed=True)

    label: str = pc.Field(
        description="Object identifier (must contain only letters, numbers, underscores)"
    )
    directory: pl.Path = pc.Field(description="Directory containing object files")
    suffix: str = pc.Field(description="File extension for object files")
    visible: bool = pc.Field(
        default=True, description="Whether object is initially visible"
    )
    renderer: vtkRenderer = pc.Field(
        exclude=True, description="VTK renderer (excluded from serialization)"
    )

    @pc.field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate that label contains only letters, numbers, and underscores."""
        if not isinstance(v, str):
            raise ValueError("label must be a string")

        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError(
                f"label '{v}' contains invalid characters. "
                "Labels must contain only letters, numbers, and underscores."
            )

        return v

    def path_for_frame(self, frame: int) -> str:
        """Generate file path for a specific frame."""
        return f"{self.directory}/{frame}.{self.suffix}"
