"""The three MPR render windows, addressed by view name rather than by field.

Every caller used to unroll the same block once per orientation, which is how
the axial, coronal and sagittal paths drifted apart. Going through this type
means a change reaches all three views or none.
"""

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
