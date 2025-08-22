# System
import logging
import pathlib as pl

# Third Party
import numpy as np
import vtk
import pydantic as pc
from pydantic import Field, model_validator, PrivateAttr

# Internal
from . import Mesh, Volume, Segmentation


class Color(pc.BaseModel):
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0


class Background(pc.BaseModel):
    light: Color = {1.0, 1.0, 1.0}
    dark: Color = {0.0, 0.0, 0.0}


class Scene(pc.BaseModel):
    model_config = pc.ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    project_name: str = "Cardio"
    current_frame: int = 0
    screenshot_directory: pl.Path
    screenshot_subdirectory_format: pl.Path
    rotation_factor: float
    background: Background = Background()
    meshes: list[Mesh] = Field(default_factory=list)
    volumes: list[Volume] = Field(default_factory=list)
    segmentations: list[Segmentation] = Field(default_factory=list)

    # VTK objects as private attributes
    _renderer: vtk.vtkRenderer = PrivateAttr(default_factory=vtk.vtkRenderer)
    _renderWindow: vtk.vtkRenderWindow = PrivateAttr(
        default_factory=vtk.vtkRenderWindow
    )
    _renderWindowInteractor: vtk.vtkRenderWindowInteractor = PrivateAttr(
        default_factory=vtk.vtkRenderWindowInteractor
    )

    @property
    def renderer(self) -> vtk.vtkRenderer:
        return self._renderer

    @property
    def renderWindow(self) -> vtk.vtkRenderWindow:
        return self._renderWindow

    @property
    def renderWindowInteractor(self) -> vtk.vtkRenderWindowInteractor:
        return self._renderWindowInteractor

    @model_validator(mode="after")
    def setup_scene(self):
        # Configure VTK objects
        self._renderer.SetBackground(
            self.background.light.r,
            self.background.light.g,
            self.background.light.b,
        )
        self._renderWindow.AddRenderer(self._renderer)
        self._renderWindow.SetOffScreenRendering(True)
        self._renderWindowInteractor.SetRenderWindow(self._renderWindow)
        self._renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        # Configure all objects
        for mesh in self.meshes:
            mesh.configure_actors()
        for volume in self.volumes:
            volume.configure_actors()
        for segmentation in self.segmentations:
            segmentation.configure_actors()

        # Set current frame using nframes property
        self.current_frame = self.current_frame % self.nframes

        # Setup rendering pipeline
        self.setup_pipeline()

        return self

    @property
    def nframes(self) -> int:
        ns = []
        ns += [len(m.actors) for m in self.meshes]
        ns += [len(v.actors) for v in self.volumes]
        ns += [len(s.actors) for s in self.segmentations]
        if not len(ns) > 0:
            logging.warning("No objects were found to display.")
            return 1
        result = int(max(ns))
        ns = np.array(ns)
        if not np.all(ns == ns[0]):
            logging.warning(f"Unequal number of frames: {ns}.")
        return result

    def setup_pipeline(self):
        """Add all actors to the renderer and configure initial visibility."""
        # Add mesh actors
        for mesh in self.meshes:
            for actor in mesh.actors:
                self.renderer.AddActor(actor)

        # Add volume actors
        for volume in self.volumes:
            for actor in volume.actors:
                self.renderer.AddVolume(actor)

        # Add segmentation actors
        for segmentation in self.segmentations:
            for actor in segmentation.actors:
                self.renderer.AddActor(actor)

        # Show current frame
        self.show_frame(self.current_frame)
        self.renderer.ResetCamera()

    def hide_all_frames(self):
        for a in self.renderer.GetActors():
            a.SetVisibility(False)
        for a in self.renderer.GetVolumes():
            a.SetVisibility(False)

    def show_frame(self, frame: int):
        for mesh in self.meshes:
            if mesh.visible:
                mesh.actors[frame % len(mesh.actors)].SetVisibility(True)
        for volume in self.volumes:
            if volume.visible:
                volume.actors[frame % len(volume.actors)].SetVisibility(True)
        for segmentation in self.segmentations:
            if segmentation.visible:
                actor = segmentation.actors[frame % len(segmentation.actors)]
                actor.SetVisibility(True)
