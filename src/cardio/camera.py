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
    axial: Pose | None = None
    coronal: Pose | None = None
    sagittal: Pose | None = None
    tile_scale: float | None = None

    @property
    def slices(self) -> dict[str, Pose]:
        """The MPR poses that are set, by view name."""
        return {
            view: pose
            for view in ("axial", "coronal", "sagittal")
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
