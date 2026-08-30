"""The encode quality of the images pushed to the browser.

Every rendered frame is JPEG-encoded and published over the websocket with no
acknowledgement from the client, so quality is the one lever that changes what
a frame costs without changing how often one is sent. Playback lowers it and
restores it on pause, so a view being inspected is always at full quality.

Resolution is the second lever. ``set_view_quality`` resizes the render window
whenever the ratio is not 1, so a resolution below full costs a resize on every
play and pause -- which is why it defaults to full and is opted into.
"""

# Third Party
import trame_vtk.modules.vtk as tvtk

FULL_QUALITY = 100
FULL_RESOLUTION = 100
DEFAULT_PLAYBACK_QUALITY = 60

# Resolution costs a diagnostic image more than compression does, so playback
# leaves it alone until the user asks for it.
DEFAULT_PLAYBACK_RESOLUTION = FULL_RESOLUTION


def ratio_from_percent(percent: float) -> float:
    """The sliders are percentages; trame wants a ratio of the full size."""
    return percent / 100.0


def render_windows(scene) -> list:
    """Every render window a client subscribes to.

    The maximized views wrap the same windows as the quad view, so setting a
    window's quality covers whichever layout is on screen.
    """
    windows = [scene.renderWindow]
    if scene.mpr_views is not None:
        windows.extend(scene.mpr_views.windows.values())
    if scene.tile_views is not None:
        windows.append(scene.tile_views.window)
    return windows


def set_image_quality(server, scene, quality: int, ratio: float = 1.0) -> bool:
    """Set the encode quality and size of every image pushed for ``scene``.

    False when there is nothing to set it on: the render helper and the
    protocol both arrive with the first connected client.
    """
    helper = tvtk.get_helper(server)
    if helper is None or not server.protocol:
        return False

    for window in render_windows(scene):
        helper.set_image_quality(window, quality, ratio)
    return True
