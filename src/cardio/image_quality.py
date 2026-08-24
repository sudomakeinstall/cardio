"""The encode quality of the images pushed to the browser.

Every rendered frame is JPEG-encoded and published over the websocket with no
acknowledgement from the client, so quality is the one lever that changes what
a frame costs without changing how often one is sent. Playback lowers it and
restores it on pause, so a view being inspected is always at full quality.

Resolution is left alone: ``set_view_quality`` resizes the render window when
the ratio is not 1, and doing that on every play and pause would churn VTK's
buffers for a lever that hurts a diagnostic image more than compression does.
"""

# Third Party
import trame_vtk.modules.vtk as tvtk

FULL_QUALITY = 100
DEFAULT_PLAYBACK_QUALITY = 60


def render_windows(scene) -> list:
    """Every render window a client subscribes to.

    The maximized views wrap the same windows as the quad view, so setting a
    window's quality covers whichever layout is on screen.
    """
    windows = [scene.renderWindow]
    if scene.mpr_views is not None:
        windows.extend(scene.mpr_views.windows.values())
    return windows


def set_image_quality(server, scene, quality: int, ratio: float = 1) -> bool:
    """Set the JPEG quality of every image pushed for ``scene``'s windows.

    False when there is nothing to set it on: the render helper and the
    protocol both arrive with the first connected client.
    """
    helper = tvtk.get_helper(server)
    if helper is None or not server.protocol:
        return False

    for window in render_windows(scene):
        helper.set_image_quality(window, quality, ratio)
    return True
