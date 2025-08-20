import re
import string
import pathlib as pl
import pydantic as pc
import functools
from vtkmodules.vtkRenderingCore import vtkRenderer
from vtkmodules.vtkCommonDataModel import vtkPlanes


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
    clipping_enabled: bool = pc.Field(default=False)

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

    @functools.cached_property
    def clipping_planes(self) -> vtkPlanes:
        """Generate clipping planes based on first actor bounds."""
        if not hasattr(self, "actors") or not self.actors:
            return None

        bounds = self.actors[0].GetBounds()
        planes = vtkPlanes()
        self._create_clipping_planes_from_bounds(planes, bounds)
        return planes

    def _create_clipping_planes_from_bounds(self, planes: vtkPlanes, bounds):
        """Create 6 clipping planes from box bounds."""
        import vtkmodules.vtkCommonCore as vtk

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
