import re
import string
import pathlib as pl
import pydantic as pc
from vtkmodules.vtkRenderingCore import vtkRenderer


class Object(pc.BaseModel):
    """Base class for renderable objects with validated configuration."""

    model_config = pc.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    label: str = pc.Field(description="Object identifier (only [a-zA-Z0-9_] allowed)")
    directory: pl.Path = pc.Field(description="Directory containing object files")
    pattern: str = pc.Field(description="Filename pattern with $frame placeholder")
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

    @pc.field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("pattern must be a string")

        if not re.match(r"^[a-zA-Z0-9_\-.${}]+$", v):
            raise ValueError("Pattern contains unsafe characters")

        if "${frame}" not in v and "$frame" not in v:
            raise ValueError("Pattern must contain $frame placeholder")

        return v

    def path_for_frame(self, frame: int) -> pl.Path:
        template = string.Template(self.pattern)
        filename = template.safe_substitute(frame=frame)

        full_path = self.directory / filename

        if not full_path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("Pattern would access files outside base directory")

        return full_path

    @property
    def path_list(self) -> list[pl.Path]:
        """Precompute list of existing file paths for all frames."""
        paths = []
        frame = 0
        while True:
            path = self.path_for_frame(frame)
            if not path.is_file():
                break
            paths.append(path)
            frame += 1
        return paths
