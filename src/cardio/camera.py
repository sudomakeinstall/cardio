"""Where each camera is looking, as something that can be written down.

A camera is the one part of what is on screen that never passed through state:
the volume rendering's is dragged by VTK's own trackball, and the MPR and tile
cameras are moved by the zoom gestures. Nothing else recorded them, so nothing
could put them back -- which is the whole of what undo and a saved session are
for.
"""

# Third Party
import pydantic as pc

Point = tuple[float, float, float]


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
