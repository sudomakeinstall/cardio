"""Drawing a traced region on the cut it belongs to.

A cut renderer's world coordinates are the cut's own millimetres, so a region
is drawn in the numbers it was traced in -- nothing here converts anything.
What this module is for is the three props that make one region legible: the
curve, the points somebody placed, and the area against it.
"""

# Third Party
import numpy as np
import vtk

# How far above the cut a region is drawn, in millimetres.  The image is a
# textured quad at zero and every camera here looks from +z -- nothing turns an
# MPR or a tile camera, ``rotate_view`` turns the reslice underneath one -- so
# this is toward the viewer, and small enough to stay inside any clipping range
# the cut itself is inside of.
CONTOUR_DEPTH = 0.01

# A region that has been closed, one still being traced, the row the drawer has
# highlighted, and the one a hand is correcting.  Constants rather than
# configured: the crosshairs are configurable because three of them have to be
# told apart, and these four are four states of one thing.
TRACED_COLOR = (1.0, 0.85, 0.1)
TRACING_COLOR = (0.3, 0.9, 1.0)
SELECTED_COLOR = (1.0, 1.0, 1.0)
EDITING_COLOR = (0.4, 1.0, 0.4)

LINE_WIDTH = 2.0
POINT_SIZE = 7.0
LABEL_FONT_SIZE = 14


def _closed_line(count: int) -> vtk.vtkCellArray:
    """One polyline through ``count`` points and back to the first."""
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(count + 1)
    for i in range(count):
        line.GetPointIds().SetId(i, i)
    line.GetPointIds().SetId(count, 0)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    return cells


def _open_line(count: int) -> vtk.vtkCellArray:
    """One polyline through ``count`` points, left open."""
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(count)
    for i in range(count):
        line.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    return cells


def _vertices(count: int) -> vtk.vtkCellArray:
    """One vertex cell per point, so they draw as marks rather than a line."""
    cells = vtk.vtkCellArray()
    for i in range(count):
        cells.InsertNextCell(1)
        cells.InsertCellPoint(i)
    return cells


def _points(planar) -> vtk.vtkPoints:
    """Cut millimetres as VTK points, lifted clear of the image."""
    points = vtk.vtkPoints()
    for u, v in np.asarray(planar, dtype=float).reshape(-1, 2):
        points.InsertNextPoint(float(u), float(v), CONTOUR_DEPTH)
    return points


def _surface_actor(polydata: vtk.vtkPolyData) -> vtk.vtkActor:
    """An actor for one overlay polydata, kept out of the camera's reckoning.

    ``fit_about_origin`` sizes a view from ``ComputeVisiblePropBounds`` and the
    tile grid takes the widest parallel scale across its tiles, so a contour
    that voted on the bounds would reframe the cut it was traced on the next
    time anything reset a camera.
    """
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    # The contour sits a hundredth of a millimetre above the image; saying so
    # to the depth buffer as well costs nothing and settles it on a driver
    # whose z precision is coarser than that.
    mapper.SetResolveCoincidentTopologyToPolygonOffset()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.UseBoundsOff()
    actor.SetVisibility(False)
    return actor


class ContourActors:
    """The props one traced region is drawn with, in one view.

    Built once and re-pointed rather than rebuilt: a cine repose costs a point
    array per frame this way, and a pipeline per frame the other.
    """

    def __init__(self):
        self._curve = vtk.vtkPolyData()
        self._handles = vtk.vtkPolyData()

        self.line = _surface_actor(self._curve)
        self.line.GetProperty().SetLineWidth(LINE_WIDTH)

        self.marks = _surface_actor(self._handles)
        self.marks.GetProperty().SetPointSize(POINT_SIZE)
        self.marks.GetProperty().SetRenderPointsAsSpheres(True)

        self.label = vtk.vtkBillboardTextActor3D()
        self.label.GetTextProperty().SetFontSize(LABEL_FONT_SIZE)
        self.label.GetTextProperty().SetJustificationToCentered()
        self.label.UseBoundsOff()
        self.label.SetVisibility(False)

        self.set_color(TRACED_COLOR)

    @property
    def props(self) -> list:
        return [self.line, self.marks, self.label]

    @property
    def showing(self) -> bool:
        """Whether these props are drawn, so that hiding them twice is one act.

        Read off the actor rather than kept beside it: the visibility is what
        the renderer is actually going by, and a flag alongside it would be one
        more thing able to disagree.
        """
        return bool(self.line.GetVisibility())

    def set_points(self, curve, handles, closed: bool) -> None:
        """Point the props at the curve drawn and the points placed.

        The two differ for a splined region: the curve is sampled through the
        handles, and it is the handles a hand can recognise as its own clicks.
        """
        curve = np.asarray(curve, dtype=float).reshape(-1, 2)
        handles = np.asarray(handles, dtype=float).reshape(-1, 2)

        self._curve.SetPoints(_points(curve))
        self._curve.SetLines(
            _closed_line(len(curve))
            if closed and len(curve) > 2
            else _open_line(len(curve))
            if len(curve) > 1
            else vtk.vtkCellArray()
        )
        self._curve.Modified()

        self._handles.SetPoints(_points(handles))
        self._handles.SetVerts(_vertices(len(handles)))
        self._handles.Modified()

    def set_label(self, text: str, at) -> None:
        """Say ``text`` at a point of the cut, or nothing at all when empty."""
        self.label.SetInput(text)
        self.label.SetPosition(float(at[0]), float(at[1]), CONTOUR_DEPTH)

    def set_color(self, color) -> None:
        self.line.GetProperty().SetColor(*color)
        self.marks.GetProperty().SetColor(*color)
        self.label.GetTextProperty().SetColor(*color)

    def set_visible(self, visible: bool) -> None:
        for prop in self.props:
            prop.SetVisibility(visible)
        # A region with no area yet is a trace in progress, and labelling it
        # would put a number beside something that is not a region.
        self.label.SetVisibility(visible and bool(self.label.GetInput()))

    def add_to(self, renderer) -> None:
        """Put the props in ``renderer``, which is safe to repeat.

        ``AddViewProp`` checks whether it is already there, so this is how a
        region gets back onto a view that has just been cleared and rebuilt
        without anything having to remember whether it was.
        """
        for prop in self.props:
            renderer.AddViewProp(prop)

    def remove_from(self, renderer) -> None:
        for prop in self.props:
            renderer.RemoveViewProp(prop)
