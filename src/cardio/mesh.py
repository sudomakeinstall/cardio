import logging
import os

import pydantic as pc
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
from vtkmodules.vtkFiltersModeling import vtkLoopSubdivisionFilter

from . import Object
from .property_config import PropertyConfig


class Mesh(Object):
    """Mesh object with subdivision support."""

    actors: list[vtkActor] = pc.Field(default_factory=list, exclude=True)
    property_config: PropertyConfig = pc.Field(default=None, exclude=True)
    loop_subdivision_iterations: int = pc.Field(ge=0, le=5, default=0)

    def __init__(self, cfg: str, renderer: vtkRenderer):
        # Validate loop subdivision iterations
        iterations = cfg.get("loop_subdivision_iterations", 0)

        super().__init__(
            label=cfg["label"],
            directory=cfg["directory"],
            suffix=cfg["suffix"],
            visible=cfg["visible"],
            renderer=renderer,
            loop_subdivision_iterations=iterations,
        )

        self.property_config = PropertyConfig.model_validate(cfg["property"])

        frame = 0
        while os.path.exists(self.path_for_frame(frame)):
            logging.info(f"{self.label}: Loading frame {frame}.")
            reader = vtkOBJReader()
            reader.SetFileName(self.path_for_frame(frame))

            # Apply loop subdivision if requested
            if self.loop_subdivision_iterations > 0:
                subdivision_filter = vtkLoopSubdivisionFilter()
                subdivision_filter.SetInputConnection(reader.GetOutputPort())
                subdivision_filter.SetNumberOfSubdivisions(
                    self.loop_subdivision_iterations
                )
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(subdivision_filter.GetOutputPort())
            else:
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtkActor()
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
