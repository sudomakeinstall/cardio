# System
import os
from importlib.metadata import version

# Third Party
import vtk

__version__ = version("cardio")

# Every window this app renders into is offscreen, whether a browser is
# watching or a script is. Set here rather than in ``main`` because a scene
# builds its render window as it is constructed, so by the time a script has a
# ``Session`` it is already too late; ``setdefault`` leaves an explicit choice
# alone.
if hasattr(vtk, "vtkEGLRenderWindow"):
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")

from . import window_level
from .logic import Logic
from .mesh import Mesh
from .object import Object
from .scene import Scene
from .scripting import script
from .segmentation import Segmentation
from .session import Session
from .ui import UI
from .volume import Volume
from .volume_property_presets import (
    list_volume_property_presets,
    load_volume_property_preset,
)

__all__ = [
    "UI",
    "Logic",
    "Mesh",
    "Object",
    "Scene",
    "Segmentation",
    "Session",
    "Volume",
    "list_volume_property_presets",
    "load_volume_property_preset",
    "script",
    "window_level",
]
