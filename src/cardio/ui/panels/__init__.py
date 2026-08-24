"""The drawer's panels, one module each."""

# Internal
from .appearance import appearance_panel, clip_depth_panel
from .capture import capture_panel
from .overlays import overlays_panel
from .playback import playback_panel
from .rotations import rotations_panel
from .snap import snap_panel

__all__ = [
    "appearance_panel",
    "capture_panel",
    "clip_depth_panel",
    "overlays_panel",
    "playback_panel",
    "rotations_panel",
    "snap_panel",
]
