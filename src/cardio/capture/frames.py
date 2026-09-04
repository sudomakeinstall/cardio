"""Successive captures of a render window."""

# Third Party
import vtk

# Internal
from .base import Frame, Plane


class WindowFrames:
    """A render window, captured once per call.

    The filter is built once and told it has changed before each update: it
    caches its output otherwise, and every frame of a cine would be the first.
    """

    def __init__(self, render_window: vtk.vtkRenderWindow, alpha: bool = False):
        self._filter = vtk.vtkWindowToImageFilter()
        self._filter.SetInput(render_window)
        self._filter.SetScale(1)
        if alpha:
            self._filter.SetInputBufferTypeToRGBA()
        else:
            self._filter.SetInputBufferTypeToRGB()
        self._filter.ReadFrontBufferOff()

    def capture(self, plane: Plane | None = None) -> Frame:
        self._filter.Modified()
        self._filter.Update()

        image = vtk.vtkImageData()
        image.ShallowCopy(self._filter.GetOutput())
        return Frame(image=image, plane=plane)
