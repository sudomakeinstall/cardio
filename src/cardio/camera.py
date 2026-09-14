"""Where each camera is looking, as something that can be written down.

A camera is the one part of what is on screen that never passed through state:
the volume rendering's is dragged by VTK's own trackball, and the MPR and tile
cameras are moved by the zoom gestures. Nothing else recorded them, so nothing
could put them back -- which is the whole of what undo and a saved session are
for.
"""

# System
import math

# Third Party
import pydantic as pc

Point = tuple[float, float, float]

# What the near/far slider moves in. A whole unit is as fine as anybody drags a
# depth in millimetres, and it is short enough to read under the thumb.
DEPTH_STEP = 1.0


class Pose(pc.BaseModel):
    """A camera's placement, in the terms VTK sets and reads it by.

    ``parallel_scale`` matters only to a camera in parallel projection, which
    the tiles are and the rest are not. It is carried for all of them anyway:
    reading and writing every camera the same way is worth more than leaving a
    field out of the ones that will ignore it.
    """

    model_config = pc.ConfigDict(extra="forbid")

    position: Point
    focal_point: Point
    view_up: Point
    parallel_scale: float

    @classmethod
    def of(cls, camera) -> "Pose":
        return cls(
            position=camera.GetPosition(),
            focal_point=camera.GetFocalPoint(),
            view_up=camera.GetViewUp(),
            parallel_scale=camera.GetParallelScale(),
        )

    def apply_to(self, camera) -> None:
        camera.SetPosition(*self.position)
        camera.SetFocalPoint(*self.focal_point)
        camera.SetViewUp(*self.view_up)
        camera.SetParallelScale(self.parallel_scale)


class Cameras(pc.BaseModel):
    """Every camera in the app, by the view it looks into.

    The tiles share one scale by design -- they exist to be compared -- so they
    are one entry however many of them there are, and each keeps the centre its
    own fit gave it.
    """

    model_config = pc.ConfigDict(extra="forbid")

    volume: Pose | None = None
    ul: Pose | None = None
    ll: Pose | None = None
    lr: Pose | None = None
    tile_scale: float | None = None

    @property
    def slices(self) -> dict[str, Pose]:
        """The MPR poses that are set, by view name."""
        return {
            view: pose
            for view in ("ul", "ll", "lr")
            if (pose := getattr(self, view)) is not None
        }


def fit_about_origin(renderer) -> None:
    """Fit the camera to what is drawn, looking at the reslice origin.

    A cut is posed by its reslice axes, so the point the views were aimed at is
    always the output's (0, 0, 0) -- which is what the crosshairs, drawn at the
    centre of the viewport, are pointing at. ``ResetCamera`` centres on the
    auto-cropped extent instead, and that is the same point only when the origin
    happens to be the image's own centre.

    Passing it a box symmetric about the origin, large enough to hold the one it
    would have chosen, leaves the fit it makes but takes away the choice of
    centre.
    """
    bounds = renderer.ComputeVisiblePropBounds()
    if bounds[0] > bounds[1]:
        renderer.ResetCamera()
        return

    x, y, z = (max(abs(bounds[2 * i]), abs(bounds[2 * i + 1])) for i in range(3))
    renderer.ResetCamera(-x, x, -y, y, -z, z)


def _focal_display_point(renderer) -> tuple[float, float, float]:
    """The camera's focal point in display coordinates, with its depth."""
    renderer.SetWorldPoint(*renderer.GetActiveCamera().GetFocalPoint(), 1.0)
    renderer.WorldToDisplay()
    return renderer.GetDisplayPoint()


def _display_to_world(renderer, x: float, y: float, depth: float) -> list[float]:
    """One display point at a fixed depth, in world units."""
    renderer.SetDisplayPoint(x, y, depth)
    renderer.DisplayToWorld()
    *point, w = renderer.GetWorldPoint()
    return [value / w for value in point] if w else point


def world_per_pixel(renderer) -> float:
    """World units spanned by one display pixel at the focal plane.

    The scale a pan needs to keep the image under the cursor, and the scale
    ``fit_factor`` sizes a fit through. Measured through the camera rather than
    from the parallel scale, so it holds for a perspective camera too. Zero if
    the viewport has never been sized, when every display point projects onto
    the same spot.

    Here rather than on ``MPRViews`` because a tile renderer is measured the
    same way, and one of the two viewports the fit is sized against is a tile.
    """
    if not all(renderer.GetSize()):
        return 0.0

    depth = _focal_display_point(renderer)[2]
    start = _display_to_world(renderer, 0.0, 0.0, depth)
    end = _display_to_world(renderer, 1.0, 0.0, depth)
    return math.dist(start, end)


def visible_rectangle(renderer):
    """What the viewport shows, in world units at the focal plane.

    ``((low x, high x), (low y, high y))``, from one edge of the viewport to
    the other: the rectangle the capture of a view has to cover for the picture
    and the data behind it to be framed alike.

    Measured through the camera's own display mapping, like ``world_per_pixel``
    and for the same reason: the MPR cameras are perspective, and a parallel
    scale would only describe the tiles.  The cut lies at the focal plane, so
    the depth a perspective camera opens out with is not in question here.

    None when the viewport has never been sized, where every display point
    projects onto the same spot.
    """
    width, height = renderer.GetSize()
    if not (width and height):
        return None

    # Display coordinates are the window's, not the viewport's, so a renderer
    # that holds one tile of a grid is measured from where its tile sits rather
    # than from the corner of the window.  Corner to corner rather than centre
    # of pixel to centre of pixel, so the rectangle is the area the view covers
    # and stays centred on what the camera is looking at.
    left, bottom = renderer.GetOrigin()
    depth = _focal_display_point(renderer)[2]
    low = _display_to_world(renderer, float(left), float(bottom), depth)
    high = _display_to_world(
        renderer, float(left + width), float(bottom + height), depth
    )
    return (low[0], high[0]), (low[1], high[1])


def fit_factor(
    half_extent: tuple[float, float],
    size: tuple[int, int],
    world_per_pixel: float,
    fill: float,
) -> float | None:
    """How much to zoom so a box about the origin fills ``fill`` of the viewport.

    ``half_extent`` is how far the box reaches from the origin along the view's
    own right and up axes. The camera looks at the origin and is never moved off
    it, so a fit is made by moving the origin onto the box's centre and passing
    the half-width from there; a caller that leaves the origin where it is has
    to pass the farthest edge instead, and buys the asymmetry as empty space.

    Sized through ``world_per_pixel`` rather than the parallel scale, so the one
    piece of arithmetic serves the perspective MPR cameras as well as a parallel
    one -- the same reason ``MPRViews.world_per_pixel`` measures through the
    camera rather than reading a scale off it.

    None when the window has never been sized, and when the box has no extent in
    either direction: a fit to a single voxel would otherwise ask to be
    magnified without limit.
    """
    width, height = size
    if not (width and height and world_per_pixel > 0.0):
        return None

    visible = (world_per_pixel * width / 2.0, world_per_pixel * height / 2.0)
    factors = [
        fill * half / reach for half, reach in zip(visible, half_extent) if reach > 0.0
    ]
    return min(factors) if factors else None


def depth_top(far: float) -> float:
    """The top of the near/far slider.

    ``far`` is where the camera's own clipping range ends once the scene is
    built, which is the far side of everything drawn: past it the slider would
    only offer more of the nothing beyond. Rounded up to a whole step, so that
    the far thumb has a notch to sit on at its own end.
    """
    return float(math.ceil(far / DEPTH_STEP) * DEPTH_STEP)


def snapped_depth(near: float, far: float) -> list[float]:
    """A clipping range moved out onto the notches of the near/far slider.

    Outwards in both directions, so that the range the app opens on is one the
    slider can hold exactly and one that clips nothing the camera was already
    showing. The near plane stops at one step rather than at zero, which is not
    a distance a perspective camera can be given.
    """
    return [
        max(DEPTH_STEP, math.floor(near / DEPTH_STEP) * DEPTH_STEP),
        depth_top(far),
    ]
