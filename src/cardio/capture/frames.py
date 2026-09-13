"""Successive captures of a render window."""

# Third Party
import vtk

# Internal
from .banner import stamp_image
from .base import Frame, Plane


class WindowFrames:
    """A render window, captured once per call.

    The filter is built once and told it has changed before each update: it
    caches its output otherwise, and every frame of a cine would be the first.

    ``banner`` is stamped onto the capture rather than onto each writer's
    output: every picture format reads this one image -- the VTK writers
    directly, the encoders through ``Frame.rgb`` -- so stamping it here is what
    puts the same band on all of them.
    """

    def __init__(
        self,
        render_window: vtk.vtkRenderWindow,
        alpha: bool = False,
        banner: str = "",
    ):
        self._filter = vtk.vtkWindowToImageFilter()
        self._filter.SetInput(render_window)
        self._filter.SetScale(1)
        if alpha:
            self._filter.SetInputBufferTypeToRGBA()
        else:
            self._filter.SetInputBufferTypeToRGB()
        self._filter.ReadFrontBufferOff()
        self.banner = banner

    def capture(self, plane: Plane | None = None, phase: int | None = None) -> Frame:
        self._filter.Modified()
        self._filter.Update()

        image = vtk.vtkImageData()
        image.ShallowCopy(self._filter.GetOutput())
        return Frame(image=stamp_image(image, self.banner), plane=plane, phase=phase)
