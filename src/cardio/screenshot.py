from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkWindowToImageFilter


class Screenshot:
    def __init__(self, renderWindow: vtkRenderWindow):
        self.windowToImageFilter = vtkWindowToImageFilter()
        self.windowToImageFilter.SetInput(renderWindow)
        self.windowToImageFilter.SetScale(1)
        self.windowToImageFilter.SetInputBufferTypeToRGBA()
        self.windowToImageFilter.ReadFrontBufferOff()

        self.writer = vtkPNGWriter()
        self.writer.SetInputConnection(self.windowToImageFilter.GetOutputPort())

    def save(self, fileName: str):
        self.windowToImageFilter.Update()
        self.writer.SetFileName(fileName)
        self.writer.Write()
