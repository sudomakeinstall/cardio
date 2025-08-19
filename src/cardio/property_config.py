"""Property configuration for mesh rendering."""

import enum
import pydantic as pc
from vtkmodules.vtkRenderingCore import vtkProperty


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


class PropertyConfig(pc.BaseModel):
    """Configuration for mesh rendering properties."""

    representation: Representation = pc.Field(
        default=Representation.Surface, description="Rendering representation mode"
    )
    r: float = pc.Field(ge=0, le=1, default=1.0, description="Red component")
    g: float = pc.Field(ge=0, le=1, default=1.0, description="Green component")
    b: float = pc.Field(ge=0, le=1, default=1.0, description="Blue component")
    edge_visibility: bool = pc.Field(default=False, description="Show edges")
    vertex_visibility: bool = pc.Field(default=False, description="Show vertices")
    shading: bool = pc.Field(default=True, description="Enable shading")
    interpolation: Interpolation = pc.Field(
        default=Interpolation.Gouraud, description="Interpolation mode"
    )
    opacity: float = pc.Field(ge=0, le=1, default=1.0, description="Transparency level")

    @property
    def vtk_property(self) -> vtkProperty:
        """Create a fully configured VTK property from this configuration."""
        property = vtkProperty()
        property.SetRepresentation(self.representation.value)
        property.SetColor(self.r, self.g, self.b)
        property.SetEdgeVisibility(self.edge_visibility)
        property.SetVertexVisibility(self.vertex_visibility)
        property.SetShading(self.shading)
        property.SetInterpolation(self.interpolation.value)
        property.SetOpacity(self.opacity)
        return property
