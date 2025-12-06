import pydantic as pc
import vtk

from .types import RGBColor


class ColorTransferFunctionPoint(pc.BaseModel):
    """A single point in a color transfer function."""

    x: float = pc.Field(description="Scalar value")
    color: RGBColor


class ColorTransferFunctionConfig(pc.BaseModel):
    """Configuration for a VTK color transfer function."""

    points: list[ColorTransferFunctionPoint] = pc.Field(
        min_length=1, description="Points defining the color transfer function"
    )

    @property
    def vtk_function(self) -> vtk.vtkColorTransferFunction:
        """Create VTK color transfer function from this configuration."""
        ctf = vtk.vtkColorTransferFunction()
        for point in self.points:
            ctf.AddRGBPoint(point.x, *point.color)
        return ctf
