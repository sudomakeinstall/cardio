"""The volumetry chart's render window: one structure's curve, and its numbers.

One window rather than a picture pushed to the browser, because a render window
is what this app already knows how to show and what its whole capture pipeline
already knows how to write.  A chart drawn here reaches the page through the
same remote view as the slices, and is exported as PNG, MP4 or a DICOM
Secondary Capture without anything new being taught how.

A page at a time rather than every structure at once.  Five charts and a shared
table fit on one page only while there are five of them and four metrics; the
moment either grows, everything on it shrinks.  A page per structure is a
sequence, and a sequence is what a Secondary Capture series is made of -- so the
thing that solves the crowding is also the thing that makes the report.

Two renderers, each with a viewport of its own, for the same reason the tile
grid has one per tile: the chart and the table are laid out by different things
-- one by the curve, the other by how many rows of text -- and a viewport
apiece is how that is said.
"""

# System
import logging

# Third Party
import vtk

# Internal
from .volumetry import Result, page_table

logger = logging.getLogger(__name__)

# How much of the window the metrics table takes: a base for the header, plus a
# share per row, capped so a page with many metrics still shows its curve.
# A fraction rather than a pixel budget, so the split survives a window the
# browser resizes without anything here having to hear about it.
BAND_BASE = 0.10
BAND_PER_ROW = 0.045
BAND_LIMIT = 0.40

# Padding inside the table band, as a share of its height.
BAND_MARGIN = 0.12

# How the lettering is sized: a share of the row height, bounded so that a
# two-row table is not set in headlines and a twelve-row one stays readable.
FONT_OF_ROW = 0.55
MIN_FONT = 7
MAX_FONT = 22

# The space left between one column and the next, in pixels.
COLUMN_GAP = 18.0

# The page's only label for which structure it is about, so it is set well
# above VTK's default twelve: a reader holding page three of a series has the
# title and nothing else to tell them it is the right ventricle.
TITLE_FONT = 22
AXIS_FONT = 14

CURVE_WIDTH = 2.0
MARKER_SIZE = 13.0
RULE_WIDTH = 1.0

# Light on dark, as the slice and tile views are, whatever theme the page is in.
FOREGROUND = (1.0, 1.0, 1.0)
GRID = (0.24, 0.24, 0.24)
RULE = (255, 255, 255, 90)

# The headroom a fixed axis leaves above the largest value, so a peak is not
# drawn along the top edge of its chart.
HEADROOM = 1.05


def byte_color(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    """A unit-interval colour as the bytes VTK's pens and brushes take."""
    return tuple(round(255 * component) for component in rgb)


def band_fraction(rows: int) -> float:
    """How much of the window's height the metrics table is given."""
    return min(BAND_LIMIT, BAND_BASE + BAND_PER_ROW * max(rows, 1))


def paired_table(name_x: str = "x", name_y: str = "y") -> vtk.vtkTable:
    """An empty two-column table, which is what every plot here is fed from."""
    table = vtk.vtkTable()
    for name in (name_x, name_y):
        column = vtk.vtkFloatArray()
        column.SetName(name)
        table.AddColumn(column)
    return table


def fill(table: vtk.vtkTable, points: list[tuple[float, float]]):
    """Rewrite a table's rows in place, and say that they moved."""
    table.SetNumberOfRows(len(points))
    for row, (x, y) in enumerate(points):
        table.SetValue(row, 0, float(x))
        table.SetValue(row, 1, float(y))
    table.Modified()


class MetricsTable:
    """The numbers under the charts, painted rather than laid out as actors.

    A ``vtkPythonItem`` drives this: ``vtkContextItem`` is abstract and cannot
    be subclassed from Python, and a row of ``vtkTextActor`` could not measure
    its own strings -- which is the whole of what laying a table out is.  The
    painter can, through ``ComputeStringBounds``, so the columns are sized to
    what they hold and the lettering to what will fit.
    """

    def __init__(self):
        self.header: tuple[str, ...] = ()
        self.rows: list[tuple[str, ...]] = []

    def Initialize(self, item) -> bool:
        return True

    @property
    def lines(self) -> list[tuple[str, ...]]:
        return [self.header, *self.rows]

    def Paint(self, item, painter) -> bool:
        if not self.rows:
            return True

        scene = item.GetScene()
        # The scene geometry, not the view: the first is this renderer's
        # viewport and the second is the whole window, and drawing to the
        # second puts the table across the charts.
        width, height = scene.GetSceneWidth(), scene.GetSceneHeight()
        if not width or not height:
            return True

        margin = height * BAND_MARGIN
        step = (height - 2 * margin) / (len(self.rows) + 1)

        text = painter.GetTextProp()
        text.SetColor(*FOREGROUND)
        text.SetJustificationToLeft()
        text.SetVerticalJustificationToBottom()

        size, offsets = self._layout(painter, text, width - 2 * margin, step)
        text.SetFontSize(size)

        left = margin
        # Painted from the top down, and VTK's y runs the other way.
        top = height - margin - step

        text.SetBold(True)
        for offset, title in zip(offsets, self.header):
            painter.DrawString(left + offset, top, title)
        text.SetBold(False)

        painter.ApplyPen(rule_pen())
        underline = top - step * 0.2
        painter.DrawLine(left, underline, width - margin, underline)

        for row, cells in enumerate(self.rows):
            baseline = top - step * (row + 1)
            for offset, value in zip(offsets, cells):
                painter.DrawString(left + offset, baseline, value)

        return True

    def _layout(self, painter, text, usable: float, step: float):
        """A font size the table fits in, and where each column starts.

        Sized to its contents rather than divided evenly: the structure names
        are several times the width of an ejection fraction, and even columns
        would either crowd the first or strand the rest.
        """
        size = max(MIN_FONT, min(MAX_FONT, int(step * FONT_OF_ROW)))
        text.SetFontSize(size)
        widths = self._column_widths(painter)

        total = sum(widths) + COLUMN_GAP * (len(widths) - 1)
        if total > usable:
            size = max(MIN_FONT, int(size * usable / total))
            text.SetFontSize(size)
            widths = self._column_widths(painter)

        offsets, running = [], 0.0
        for column in widths:
            offsets.append(running)
            running += column + COLUMN_GAP
        return size, offsets

    def _column_widths(self, painter) -> list[float]:
        """How wide each column has to be to hold its widest cell."""
        bounds = [0.0, 0.0, 0.0, 0.0]

        def measured(value: str) -> float:
            painter.ComputeStringBounds(value, bounds)
            return bounds[2]

        return [
            max(measured(line[column]) for line in self.lines if column < len(line))
            for column in range(len(self.header))
        ]


def rule_pen() -> vtk.vtkPen:
    pen = vtk.vtkPen()
    pen.SetColor(*RULE)
    pen.SetWidth(RULE_WIDTH)
    return pen


class VolumetryViews:
    """One offscreen render window: a grid of curves, and a table of numbers."""

    def __init__(self, background: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.background = background

        self._window = vtk.vtkRenderWindow()
        self._window.SetOffScreenRendering(True)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyle())
        self._window.SetInteractor(interactor)

        self._chart_renderer, self._chart_scene = self._add_layer()
        self._table_renderer, self._table_scene = self._add_layer()

        self._result: Result | None = None
        self._chart: vtk.vtkChartXY | None = None
        self._rule: vtk.vtkTable | None = None
        self._range: tuple[float, float] = (0.0, 1.0)
        self._page = 0
        self._frame = 0

        # The tables and plots are held because VTK keeps only a weak hold of
        # its own: a plot whose python table is collected draws nothing.
        self._held: list = []

        self._table = MetricsTable()
        self._item = vtk.vtkPythonItem()
        self._item.SetPythonObject(self._table)
        self._table_scene.AddItem(self._item)

        self.set_split(band_fraction(0))

    def _add_layer(self) -> tuple[vtk.vtkRenderer, vtk.vtkContextScene]:
        """A renderer with a 2D context scene drawn into it."""
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(*self.background)
        # Opaque, unlike every other viewport's. Those are cut-outs of a scene,
        # and a transparent ground is what lets one be composited; a chart is
        # the ground -- nine tenths of it is background, and captured with an
        # alpha of zero the lettering arrives as white on whatever the reader
        # happens to open it over.
        renderer.SetBackgroundAlpha(1.0)
        self._window.AddRenderer(renderer)

        scene = vtk.vtkContextScene()
        actor = vtk.vtkContextActor()
        actor.SetScene(scene)
        renderer.AddViewProp(actor)
        scene.SetRenderer(renderer)
        return renderer, scene

    @property
    def window(self) -> vtk.vtkRenderWindow:
        return self._window

    @property
    def chart(self) -> vtk.vtkChartXY | None:
        """The chart currently drawn, or None before anything has been shown."""
        return self._chart

    @property
    def pages(self) -> int:
        """How many structures there are to page through."""
        return len(self._result.measurements) if self._result is not None else 0

    @property
    def page(self) -> int:
        return self._page

    def set_split(self, fraction: float):
        """Give the table the bottom ``fraction`` of the window, the chart the rest."""
        self._table_renderer.SetViewport(0.0, 0.0, 1.0, fraction)
        self._chart_renderer.SetViewport(0.0, fraction, 1.0, 1.0)

    def clear(self):
        """Drop the page, leaving the window and its layers in place."""
        if self._chart is not None:
            self._chart_scene.RemoveItem(self._chart)
        self._chart = None
        self._rule = None
        self._held = []
        self._table.header = ()
        self._table.rows = []

    def show(self, result: Result):
        """Take a measurement to page through, and draw its first page."""
        self._result = result
        self.set_page(0)

    def set_page(self, page: int):
        """Draw one structure: its curve above, its metrics below.

        Rebuilt rather than hidden and shown, because a page is one chart and
        one table -- cheap enough that keeping every structure's built and
        switching between them would be more state than it saves.
        """
        self.clear()
        if self._result is None or not self._result.measurements:
            self._page = 0
            self.set_split(band_fraction(0))
            return

        self._page = max(0, min(int(page), self.pages - 1))
        measurement = self._result.measurements[self._page]

        self._build_chart(measurement)
        self._table.header, self._table.rows = page_table(measurement, self._result.bsa)
        self.set_split(band_fraction(len(self._table.rows)))
        self.set_frame(self._frame)

    def _build_chart(self, measurement):
        group = measurement.group
        chart = vtk.vtkChartXY()
        chart.SetTitle(group.name)
        chart.SetShowLegend(False)
        style(chart.GetTitleProperties(), TITLE_FONT)
        self._chart_scene.AddItem(chart)

        values = measurement.values
        # From zero, so that two pages are read against the same floor and a
        # curve that barely moves is seen not to; the script's `rangemode`.
        low = 0.0
        high = float(values.max()) * HEADROOM or 1.0
        self._range = (low, high)

        left = chart.GetAxis(vtk.vtkAxis.LEFT)
        left.SetTitle(
            f"{'Mass' if group.density is not None else 'Volume'} ({group.unit})"
        )
        left.SetBehavior(vtk.vtkAxis.FIXED)
        left.SetRange(low, high)
        bottom = chart.GetAxis(vtk.vtkAxis.BOTTOM)
        bottom.SetTitle("Frame")
        for axis in (left, bottom):
            style(axis.GetTitleProperties(), AXIS_FONT)
            style(axis.GetLabelProperties(), AXIS_FONT)
            axis.GetGridPen().SetColorF(*GRID)

        rgb = byte_color(group.rgb)

        curve = paired_table("frame", group.name)
        fill(curve, list(enumerate(float(value) for value in values)))
        plot = chart.AddPlot(vtk.vtkChart.LINE)
        plot.SetInputData(curve, 0, 1)
        plot.SetColor(*rgb, 255)
        plot.SetWidth(CURVE_WIDTH)
        self._held.append((curve, plot))

        for marker, frame in (
            (vtk.vtkPlotPoints.CROSS, measurement.metrics.maximum_frame),
            (vtk.vtkPlotPoints.DIAMOND, measurement.metrics.minimum_frame),
        ):
            extreme = paired_table()
            fill(extreme, [(frame, float(values[frame]))])
            points = chart.AddPlot(vtk.vtkChart.POINTS)
            points.SetInputData(extreme, 0, 1)
            points.SetColor(*rgb, 255)
            points.SetMarkerStyle(marker)
            points.SetMarkerSize(MARKER_SIZE)
            self._held.append((extreme, points))

        rule = paired_table()
        rule_plot = chart.AddPlot(vtk.vtkChart.LINE)
        rule_plot.SetInputData(rule, 0, 1)
        rule_plot.SetColor(*RULE)
        rule_plot.SetWidth(RULE_WIDTH)
        self._rule = rule
        self._held.append((rule, rule_plot))

        self._chart = chart

    def set_frame(self, frame: int):
        """Move the rule marking which frame the rest of the app is showing.

        Only the rule moves: a curve covers every frame, so what a frame change
        means here is a position and not a measurement.
        """
        self._frame = int(frame)
        if self._rule is not None:
            low, high = self._range
            fill(self._rule, [(self._frame, low), (self._frame, high)])


def style(properties, size: int | None = None):
    """Put a chart's lettering in the one colour everything here is drawn in."""
    properties.SetColor(*FOREGROUND)
    if size is not None:
        properties.SetFontSize(size)
