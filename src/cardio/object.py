# System
import functools
import logging
import pathlib as pl
import re

# Third Party
import pydantic as pc
import vtk

from .utils import calculate_combined_bounds


class Object(pc.BaseModel):
    """Base class for renderable objects with validated configuration."""

    model_config = pc.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    label: str = pc.Field(description="Object identifier (only [a-zA-Z0-9_] allowed)")
    directory: pl.Path = pc.Field(description="Directory containing object files")
    pattern: str | None = pc.Field(
        default=None, description="Filename pattern with ${frame} placeholder"
    )
    frame_start: pc.NonNegativeInt = 0
    frame_interval: pc.PositiveInt = 1
    file_paths: list[str] | None = pc.Field(
        default=None, description="Static list of file paths relative to directory"
    )
    visible: bool = pc.Field(
        default=True, description="Whether object is initially visible"
    )
    clipping_enabled: bool = pc.Field(default=True)

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
    def validate_pattern(cls, v: str | None) -> str | None:
        if v is None:
            return v

        if not isinstance(v, str):
            raise ValueError("pattern must be a string")

        if not re.match(r"^[a-zA-Z0-9_\-.${}:]+$", v):
            raise ValueError("Pattern contains unsafe characters")

        if "frame" not in v:
            raise ValueError("Pattern must contain $frame placeholder")

        return v

    @pc.model_validator(mode="after")
    def validate_pattern_or_file_paths(self):
        if self.pattern is None and self.file_paths is None:
            raise ValueError("Either pattern or file_paths must be provided")
        if self.pattern is not None and self.file_paths is not None:
            logging.info("Both pattern and file_paths specified; using file_paths.")

        for path in self.path_list:
            if not path.is_file():
                raise ValueError(f"File does not exist: {path}")

        return self

    def path_for_frame(self, frame: int) -> pl.Path:
        if self.pattern is None:
            raise ValueError("Cannot use path_for_frame with static file_paths")
        filename = self.pattern.format(frame=frame)
        return self.directory / filename

    @functools.cached_property
    def path_list(self) -> list[pl.Path]:
        """Return list of file paths, using static paths if provided, otherwise dynamic pattern-based paths."""
        if self.file_paths is not None:
            return [self.directory / path for path in self.file_paths]

        paths = []
        frame = self.frame_start
        while True:
            path = self.path_for_frame(frame)
            if not path.is_file():
                break
            paths.append(path)
            frame += self.frame_interval
        return paths

    @property
    def combined_bounds(self) -> list[float]:
        """Get combined bounds encompassing all actors."""
        if not hasattr(self, "actors") or not self.actors:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return calculate_combined_bounds(self.actors)

    @functools.cached_property
    def clipping_planes(self) -> vtk.vtkPlanes:
        """Generate clipping planes based on combined bounds of all actors."""
        if not hasattr(self, "actors") or not self.actors:
            return None

        bounds = self.combined_bounds
        planes = vtk.vtkPlanes()
        self._create_clipping_planes_from_bounds(planes, bounds)
        return planes

    def _create_clipping_planes_from_bounds(self, planes: vtk.vtkPlanes, bounds):
        """Create 6 clipping planes from box bounds."""

        # Create 6 planes for the box faces
        normals = [
            [1, 0, 0],
            [-1, 0, 0],  # x-min, x-max
            [0, 1, 0],
            [0, -1, 0],  # y-min, y-max
            [0, 0, 1],
            [0, 0, -1],  # z-min, z-max
        ]
        origins = [
            [bounds[0], 0, 0],
            [bounds[1], 0, 0],  # x-min, x-max
            [0, bounds[2], 0],
            [0, bounds[3], 0],  # y-min, y-max
            [0, 0, bounds[4]],
            [0, 0, bounds[5]],  # z-min, z-max
        ]

        points = vtk.vtkPoints()
        norms = vtk.vtkDoubleArray()
        norms.SetNumberOfComponents(3)
        norms.SetName("Normals")

        for normal, origin in zip(normals, origins):
            points.InsertNextPoint(origin)
            norms.InsertNextTuple(normal)

        planes.SetPoints(points)
        planes.SetNormals(norms)

    def toggle_clipping(self, enabled: bool):
        """Enable or disable clipping for all actors."""
        if not hasattr(self, "actors"):
            return

        if enabled and self.clipping_planes:
            # Apply clipping to all actors
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.SetClippingPlanes(self.clipping_planes)
        else:
            # Remove clipping from all actors
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.RemoveAllClippingPlanes()

    def update_clipping_bounds(self, bounds):
        """Update clipping bounds from UI controls."""
        if not self.clipping_planes:
            return

        # Update clipping planes with new bounds
        self._create_clipping_planes_from_bounds(self.clipping_planes, bounds)

        # Apply to all actors if clipping is enabled
        if self.clipping_enabled:
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.SetClippingPlanes(self.clipping_planes)
