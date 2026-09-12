"""The drawer's panels, one module each."""

# Internal
from .appearance import volume_rendering_panel
from .capture import capture_panel
from .console import console_panel
from .overlays import slice_views_panel
from .playback import playback_panel
from .rotations import rotations_panel
from .snap import snap_panel, volume_panel
from .tiles import tiles_panel
from .volumetry import volumetry_panel
from .zoom import zoom_panel

__all__ = [
    "capture_panel",
    "console_panel",
    "playback_panel",
    "rotations_panel",
    "slice_views_panel",
    "snap_panel",
    "tiles_panel",
    "volume_panel",
    "volume_rendering_panel",
    "volumetry_panel",
    "zoom_panel",
]
