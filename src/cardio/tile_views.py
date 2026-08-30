"""The tile grid's render window: one renderer per tile, in one window.

The trame layout is built once, so a grid whose shape changes at runtime cannot
be one remote view per tile. Putting every tile in its own viewport of a single
window means the shape is ours to change -- add renderers, recompute rectangles
-- and the client still subscribes to exactly one image stream.
"""

# Third Party
import vtk

# Internal
from .reslice import TileSet

# Six each way is already 36 reslices per frame per object; past that the tiles
# are too small to read anyway.
MAX_ROWS = 6
MAX_COLS = 6


def tile_viewport(
    index: int, rows: int, cols: int
) -> tuple[float, float, float, float]:
    """The (x0, y0, x1, y1) rectangle of one tile, in row-major order.

    Tile 0 is top left, which is where a reader starts; VTK's y runs bottom up,
    hence the subtraction.
    """
    row, column = divmod(index, cols)
    return (
        column / cols,
        1.0 - (row + 1) / rows,
        (column + 1) / cols,
        1.0 - row / rows,
    )


class TileViews:
    """One offscreen render window whose renderers tile a grid."""

    def __init__(self, background: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.background = background
        self._renderers: list[vtk.vtkRenderer] = []
        self.rows = 0
        self.cols = 0

        self._window = vtk.vtkRenderWindow()
        self._window.SetOffScreenRendering(True)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyle())
        self._window.SetInteractor(interactor)

    @property
    def window(self) -> vtk.vtkRenderWindow:
        return self._window

    @property
    def renderers(self) -> list[vtk.vtkRenderer]:
        return list(self._renderers)

    def __len__(self) -> int:
        return len(self._renderers)

    def set_grid(self, rows: int, cols: int):
        """Reshape the grid to ``rows`` by ``cols``, adding or dropping tiles."""
        rows = max(1, min(MAX_ROWS, int(rows)))
        cols = max(1, min(MAX_COLS, int(cols)))
        count = rows * cols

        while len(self._renderers) > count:
            self._window.RemoveRenderer(self._renderers.pop())

        while len(self._renderers) < count:
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(*self.background)
            renderer.GetActiveCamera().ParallelProjectionOn()
            self._window.AddRenderer(renderer)
            self._renderers.append(renderer)

        for index, renderer in enumerate(self._renderers):
            renderer.SetViewport(*tile_viewport(index, rows, cols))

        self.rows, self.cols = rows, cols

    def clear(self):
        """Drop every prop from all the tiles."""
        for renderer in self._renderers:
            renderer.RemoveAllViewProps()

    def set_images(self, tiles: TileSet):
        """Show one resliced tile in each renderer."""
        for renderer, parts in zip(self._renderers, tiles.values()):
            renderer.AddActor(parts["actor"])
            parts["actor"].SetVisibility(True)

    def add_overlay(self, tiles: TileSet):
        """Add a resliced overlay on top of whatever each tile already shows."""
        self.set_images(tiles)

    def show(self, tiles: TileSet, reset_cameras: bool = False):
        """Replace the contents of every tile with one frame's cuts."""
        self.clear()
        self.set_images(tiles)
        if reset_cameras:
            self.reset_cameras()

    def zoom(self, factor: float):
        """Zoom every tile by the same factor, keeping their one shared scale.

        The tiles exist to be compared, so they hold a single parallel scale;
        scaling them all by the same amount is the only zoom that leaves that
        true. Each tile keeps its own centre, so the grid magnifies in place.
        """
        if factor <= 0.0:
            return

        for renderer in self._renderers:
            camera = renderer.GetActiveCamera()
            camera.SetParallelScale(camera.GetParallelScale() / factor)

    def reset_cameras(self):
        """Refit the tiles, then put them all on one scale.

        ``AutoCropOutputOn`` gives each oblique cut its own extent, so fitting
        each tile independently would zoom every tile differently and defeat the
        comparison the grid exists for.
        """
        if not self._renderers:
            return

        for renderer in self._renderers:
            renderer.ResetCamera()

        scale = max(r.GetActiveCamera().GetParallelScale() for r in self._renderers)
        for renderer in self._renderers:
            renderer.GetActiveCamera().SetParallelScale(scale)
