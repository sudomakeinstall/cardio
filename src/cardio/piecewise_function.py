# Third Party
import pydantic as pc
import vtk

# Internal
from .types import ScalarComponent


class PiecewiseFunctionPoint(pc.BaseModel):
    """A single point in a piecewise function."""

    x: float = pc.Field(description="Scalar value")
    y: ScalarComponent


class PiecewiseFunctionConfig(pc.BaseModel):
    """Configuration for a VTK piecewise function (opacity)."""

    points: list[PiecewiseFunctionPoint] = pc.Field(
        min_length=1, description="Points defining the piecewise function"
    )

    @property
    def vtk_function(self) -> vtk.vtkPiecewiseFunction:
        """Create VTK piecewise function from this configuration."""
        otf = vtk.vtkPiecewiseFunction()
        for point in self.points:
            otf.AddPoint(point.x, point.y)
        return otf
