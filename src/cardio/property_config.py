# System
import enum
import functools

# Third Party
import pydantic as pc
import vtk

# Internal
from .types import RGBColor, ScalarComponent


class Representation(enum.IntEnum):
    """Mesh representation modes for VTK rendering."""

    Points = 0
    Wireframe = 1
    Surface = 2


class Interpolation(enum.IntEnum):
    """Mesh interpolation modes for VTK rendering."""

    Flat = 0
    Gouraud = 1
    Phong = 2
    PBR = 3


class vtkPropertyConfig(pc.BaseModel):
    """Configuration for mesh rendering properties."""

    representation: Representation = pc.Field(
        default=Representation.Surface, description="Rendering representation mode"
    )
    color: RGBColor = (1.0, 1.0, 1.0)
    edge_visibility: bool = pc.Field(default=False, description="Show edges")
    vertex_visibility: bool = pc.Field(default=False, description="Show vertices")
    shading: bool = pc.Field(default=True, description="Enable shading")
    interpolation: Interpolation = pc.Field(
        default=Interpolation.Gouraud, description="Interpolation mode"
    )
    opacity: ScalarComponent = 1.0

    @functools.cached_property
    def vtk_property(self) -> vtk.vtkProperty:
        """Create a fully configured VTK property from this configuration."""
        _vtk_property = vtk.vtkProperty()
        _vtk_property.SetRepresentation(self.representation.value)
        _vtk_property.SetColor(*self.color)
        _vtk_property.SetEdgeVisibility(self.edge_visibility)
        _vtk_property.SetVertexVisibility(self.vertex_visibility)
        _vtk_property.SetShading(self.shading)
        _vtk_property.SetInterpolation(self.interpolation.value)
        _vtk_property.SetOpacity(self.opacity)
        return _vtk_property
