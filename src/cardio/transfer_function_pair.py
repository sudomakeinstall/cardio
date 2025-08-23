# Third Party
import pydantic as pc
import vtk

# Internal
from .color_transfer_function import ColorTransferFunctionConfig
from .piecewise_function import PiecewiseFunctionConfig


class TransferFunctionPairConfig(pc.BaseModel):
    """Configuration for a pair of opacity and color transfer functions."""

    opacity: PiecewiseFunctionConfig = pc.Field(
        description="Opacity transfer function configuration"
    )
    color: ColorTransferFunctionConfig = pc.Field(
        description="Color transfer function configuration"
    )

    @property
    def vtk_functions(
        self,
    ) -> tuple[vtk.vtkPiecewiseFunction, vtk.vtkColorTransferFunction]:
        """Create VTK transfer functions from this pair configuration."""
        return self.opacity.vtk_function, self.color.vtk_function
