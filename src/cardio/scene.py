import logging

import numpy as np
import tomlkit as tk

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle

# Required for rendering initialization, not necessary for
# local rendering, but doesn't hurt to include it
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2  # noqa

# Required for interactor initialization
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

from . import Mesh, Volume


class Scene:
    def __init__(self, cfg_file: str):
        self.meshes: list[Mesh] = []
        self.volumes: list[Volume] = []
        self.nframes: int = None
        self.renderer: vtkRenderer = vtkRenderer()
        self.renderWindow: vtkRenderWindow = vtkRenderWindow()
        self.renderWindowInteractor: vtkRenderWindowInteractor = (
            vtkRenderWindowInteractor()
        )
        self.renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        with open(cfg_file, mode="rt", encoding="utf-8") as fp:
            self.cfg = tk.load(fp)
        for mesh_cfg in self.cfg["meshes"]:
            self.meshes += [Mesh(mesh_cfg, self.renderer)]
        for volume_cfg in self.cfg["volumes"]:
            self.volumes += [Volume(volume_cfg, self.renderer)]
        self.set_nframes()

        self.project_name = self.cfg["project_name"]
        self.current_frame = self.cfg["current_frame"] % self.nframes
        self.screenshot_directory = self.cfg["screenshot_directory"]
        self.screenshot_subdirectory_format = self.cfg["screenshot_subdirectory_format"]
        self.rotation_factor = self.cfg["rotation_factor"]
        self.background_r = self.cfg["background_r"]
        self.background_g = self.cfg["background_g"]
        self.background_b = self.cfg["background_b"]

        self.setup_rendering()
        for mesh in self.meshes:
            mesh.setup_pipeline(self.current_frame)
        for volume in self.volumes:
            volume.setup_pipeline(self.current_frame)

    def set_nframes(self):
        ns = []
        ns += [len(m.actors) for m in self.meshes]
        ns += [len(v.actors) for v in self.volumes]
        ns = np.array(ns)
        assert len(ns) > 0
        assert np.all(ns == ns[0])
        self.nframes = int(ns[0])

    def setup_rendering(self):
        self.renderer.SetBackground(
            self.background_r, self.background_g, self.background_b
        )
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)

    def hide_all_frames(self):
        for a in self.renderer.GetActors():
            a.SetVisibility(False)
        for a in self.renderer.GetVolumes():
            a.SetVisibility(False)

    def show_frame(self, frame: int):
        for mesh in self.meshes:
            if mesh.visible:
                mesh.actors[frame].SetVisibility(True)
        for volume in self.volumes:
            if volume.visible:
                volume.actors[frame].SetVisibility(True)
