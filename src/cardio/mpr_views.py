"""The three MPR render windows, addressed by view name rather than by field.

Every caller used to unroll the same block once per orientation, which is how
the axial, coronal and sagittal paths drifted apart. Going through this type
means a change reaches all three views or none.
"""

# System
import math

# Third Party
import vtk

# Internal
from .reslice import VIEWS, ResliceSet


class MPRViews:
    """One offscreen render window per MPR orientation."""

    def __init__(self, background: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.windows: dict[str, vtk.vtkRenderWindow] = {}

        for view in VIEWS:
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(*background)

            window = vtk.vtkRenderWindow()
            window.AddRenderer(renderer)
            window.SetOffScreenRendering(True)

            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetInteractorStyle(vtk.vtkInteractorStyle())
            window.SetInteractor(interactor)

            self.windows[view] = window

    def __getitem__(self, view: str) -> vtk.vtkRenderWindow:
        return self.windows[view]

    def __iter__(self):
        return iter(self.windows)

    def renderer(self, view: str) -> vtk.vtkRenderer:
        return self.windows[view].GetRenderers().GetFirstRenderer()

    def clear(self):
        """Drop every prop from all three renderers."""
        for view in self.windows:
            self.renderer(view).RemoveAllViewProps()

    def set_image(self, slices: ResliceSet):
        """Show a frame's resliced image in each view."""
        for view in self.windows:
            actor = slices[view]["actor"]
            self.renderer(view).AddActor(actor)
            actor.SetVisibility(True)

    def add_overlay(self, slices: ResliceSet):
        """Add a resliced overlay on top of whatever each view already shows."""
        self.set_image(slices)

    def add_crosshairs(self, crosshairs: dict, visible: bool):
        """Add the 2D crosshair overlays, if the active volume has any."""
        if not crosshairs:
            return

        for view in self.windows:
            if view not in crosshairs:
                continue
            for line in crosshairs[view].values():
                self.renderer(view).AddActor2D(line["actor"])
                line["actor"].SetVisibility(visible)

    def show(
        self,
        slices: ResliceSet,
        crosshairs: dict | None = None,
        crosshairs_visible: bool = True,
        reset_camera: bool = False,
    ):
        """Replace the contents of all three views with one frame's slices."""
        self.clear()
        self.set_image(slices)
        self.add_crosshairs(crosshairs or {}, crosshairs_visible)
        if reset_camera:
            self.reset_cameras()

    def reset_cameras(self):
        for view in self.windows:
            self.renderer(view).ResetCamera()

    def origin_on_screen(self, view: str) -> tuple[float, float] | None:
        """Where the camera's focal point lands, in display coordinates.

        The origin sits at the centre of every reslice and the camera looks
        straight at it, so this is where the crosshair is drawn -- the point a
        rotation gesture turns about. None until the window has been sized.
        """
        renderer = self.renderer(view)
        if not all(renderer.GetSize()):
            return None

        x, y, _ = _focal_display_point(renderer)
        return x, y

    def zoom(self, factor: float):
        """Zoom every view by the same factor, each about its own centre.

        All three move together so the MPRs stay comparable, the way the tile
        grid shares one scale. The focal point does not move, so the origin
        stays under the crosshair.
        """
        if factor <= 0.0:
            return

        for view in self.windows:
            renderer = self.renderer(view)
            camera = renderer.GetActiveCamera()
            if camera.GetParallelProjection():
                camera.SetParallelScale(camera.GetParallelScale() / factor)
            else:
                camera.Dolly(factor)
            renderer.ResetCameraClippingRange()

    def world_per_pixel(self, view: str) -> float:
        """World units spanned by one display pixel at the focal plane.

        The scale a pan needs to keep the image under the cursor. Measured
        through the camera rather than from the parallel scale, so it holds for
        a perspective camera too. Zero if the window has never been sized, when
        every display point projects onto the same spot.
        """
        renderer = self.renderer(view)
        if not all(renderer.GetSize()):
            return 0.0

        depth = _focal_display_point(renderer)[2]
        start = _display_to_world(renderer, 0.0, 0.0, depth)
        end = _display_to_world(renderer, 1.0, 0.0, depth)
        return math.dist(start, end)


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
