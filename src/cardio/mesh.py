import logging
import os

from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer

from . import Object
from .property_config import PropertyConfig


class Mesh(Object):
    def __init__(self, cfg: str, renderer: vtkRenderer):
        super().__init__(cfg, renderer)
        self.actors: list[vtkActor] = []

        self.property_config = PropertyConfig.model_validate(cfg["property"])

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
            a.SetProperty(self.property_config.vtk_property)
        if self.visible:
            self.actors[frame].SetVisibility(True)
        self.renderer.ResetCamera()
