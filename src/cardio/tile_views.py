"""The tile grid's render window: one renderer per tile, in one window.

The trame layout is built once, so a grid whose shape changes at runtime cannot
be one remote view per tile. Putting every tile in its own viewport of a single
window means the shape is ours to change -- add renderers, recompute rectangles
-- and the client still subscribes to exactly one image stream.
"""

# Third Party
import vtk

# Internal
from .camera import (
    fit_about_origin,
    visible_pixels,
    visible_rectangle,
    world_per_pixel,
)
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
        self.focus: int | None = None

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

    @property
    def drawn(self) -> list[vtk.vtkRenderer]:
        """The renderers the window is actually drawing.

        Every tile, or the focused one alone. What the grid can say about
        itself -- how much of a cut a tile shows, how many pixels it is drawn
        on -- is asked of these rather than of the first renderer, which may be
        one of the hidden ones.
        """
        if self.focus is None:
            return list(self._renderers)
        return [self._renderers[self.focus]]

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

        self.rows, self.cols = rows, cols
        if self.focus is not None and self.focus >= count:
            self.focus = None
        self._apply_viewports()

    def set_focus(self, index: int | None):
        """Give one tile the whole window, or give the grid back.

        An index outside the grid focuses nothing, so a focus held across a
        reshape that no longer has room for it falls away rather than raising.
        """
        if index is not None and not 0 <= index < len(self._renderers):
            index = None

        self.focus = index
        self._apply_viewports()

    def _apply_viewports(self):
        """Give every renderer the rectangle its tile is drawn in.

        A focused tile takes the window and the rest stop drawing. They keep
        their own rectangles all the same: the grid comes back by drawing them
        again, and nothing has to remember where they were.
        """
        for index, renderer in enumerate(self._renderers):
            focused = index == self.focus
            renderer.SetViewport(
                *(
                    (0.0, 0.0, 1.0, 1.0)
                    if focused
                    else tile_viewport(index, self.rows, self.cols)
                )
            )
            renderer.SetDraw(self.focus is None or focused)

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

    def tile_at(self, x: float, y: float) -> int | None:
        """Which tile a display point landed in, or None outside the grid.

        Display coordinates are the window's rather than a tile's -- the grid
        is one window of viewports, not one window each -- so a tile is found
        by the rectangle it was given rather than by asking it. None before the
        window has been sized, when every tile is nothing by nothing.

        A hidden tile keeps the rectangle it had in the grid, so the ones a
        focus turned off are skipped: the point landed on what is drawn there
        now, not on what used to be.
        """
        for index, renderer in enumerate(self._renderers):
            if not renderer.GetDraw():
                continue
            left, bottom = renderer.GetOrigin()
            width, height = renderer.GetSize()
            if not (width and height):
                continue
            if left <= x <= left + width and bottom <= y <= bottom + height:
                return index
        return None

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

    def world_per_pixel(self) -> float:
        """World units spanned by one display pixel of a tile.

        One answer for the whole grid: the tiles hold a single parallel scale
        and share the window evenly, so every tile measures the same. Zero
        before the window has been sized, as the MPR views' does.
        """
        if not self.drawn:
            return 0.0
        return world_per_pixel(self.drawn[0])

    def shown_rectangle(self):
        """What a tile is showing of its own cut, in world units.

        One answer for the whole grid, for the same reason ``world_per_pixel``
        is one: the tiles hold a single parallel scale and share the window
        evenly, so each shows the same rectangle about its own pose.  None
        before the window has been sized, when there is nothing for a capture
        to be framed like.
        """
        if not self.drawn:
            return None
        return visible_rectangle(self.drawn[0])

    def shown_pixels(self):
        """How many pixels one tile is drawn on, as ``(columns, rows)``.

        One tile rather than the whole grid, because a tile is what a cut is
        resampled to fill: the mosaic is that many pixels again in each
        direction.  None before the window has been sized.
        """
        if not self.drawn:
            return None
        return visible_pixels(self.drawn[0])

    def reset_cameras(self):
        """Refit the tiles, then put them all on one scale.

        ``AutoCropOutputOn`` gives each oblique cut its own extent, so fitting
        each tile independently would zoom every tile differently and defeat the
        comparison the grid exists for.

        Every tile rather than only the ones being drawn, so that a refit taken
        while one tile fills the window does not hand the grid back with that
        tile at a scale of its own. A hidden tile keeps its props and its cell,
        so it fits the way it always did, and the shared maximum frames every
        tile in whichever viewport it is drawn into.
        """
        if not self._renderers:
            return

        for renderer in self._renderers:
            fit_about_origin(renderer)

        scale = max(r.GetActiveCamera().GetParallelScale() for r in self._renderers)
        for renderer in self._renderers:
            renderer.GetActiveCamera().SetParallelScale(scale)
