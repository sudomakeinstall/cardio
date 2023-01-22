import logging
import os

from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer

from . import Object


class Mesh(Object):
    def __init__(self, cfg: str, renderer: vtkRenderer):
        super().__init__(cfg, renderer)
        self.actors: list[vtkActor] = []

        self.color_r: float = cfg["color_r"]
        self.color_g: float = cfg["color_g"]
        self.color_b: float = cfg["color_b"]

        frame = 0
        while os.path.exists(self.path_for_frame(frame)):
            logging.info(f"{self.label}: Loading frame {frame}.")
            reader = vtkOBJReader()
            mapper = vtkPolyDataMapper()
            actor = vtkActor()
            reader.SetFileName(self.path_for_frame(frame))
            mapper.SetInputConnection(reader.GetOutputPort())
            actor.SetMapper(mapper)
            reader.Update()
            self.actors += [actor]
            frame += 1

    def setup_pipeline(self, frame: int):
        for a in self.actors:
            self.renderer.AddActor(a)
            a.SetVisibility(False)
            a.GetProperty().SetColor(self.color_r, self.color_g, self.color_b)
        if self.visible:
            self.actors[frame].SetVisibility(True)
        self.renderer.ResetCamera()
