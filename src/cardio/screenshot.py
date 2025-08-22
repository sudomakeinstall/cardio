import vtk


class Screenshot:
    def __init__(self, renderWindow: vtk.vtkRenderWindow):
        self.windowToImageFilter = vtk.vtkWindowToImageFilter()
        self.windowToImageFilter.SetInput(renderWindow)
        self.windowToImageFilter.SetScale(1)
        self.windowToImageFilter.SetInputBufferTypeToRGBA()
        self.windowToImageFilter.ReadFrontBufferOff()

        self.writer = vtk.vtkPNGWriter()
        self.writer.SetInputConnection(self.windowToImageFilter.GetOutputPort())

    def save(self, fileName: str):
        self.windowToImageFilter.Update()
        self.writer.SetFileName(fileName)
        self.writer.Write()
