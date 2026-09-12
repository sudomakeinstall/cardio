"""The chart window draws what was measured, offscreen and on its own.

The gap this guards: a chart is only useful if it renders where the app renders
-- headless, through EGL, into a window the capture pipeline can read pixels
out of.  None of the app's other views could have told us that, because none of
them draws a 2D context scene, and several of the failures here are silent: a
python-backed item whose table is collected draws nothing at all, a page laid
out against the window rather than its own viewport loses all but two rows, and
a capture taken with the renderer's default alpha is a transparent cut-out.
"""

# Third Party
import numpy as np
import pytest
import vtk
from vtk.util import numpy_support as vtknp

# Internal
from cardio.utils import label_color
from cardio.volumetry import ABSENT, Result, StructureGroup, measure
from cardio.volumetry_views import VolumetryViews, band_fraction, byte_color

SIZE = (900, 700)


def group(name: str, label: int, **kwargs) -> StructureGroup:
    return StructureGroup(name=name, labels=[label], **kwargs)


def curve(peak: float, frames: int = 12) -> list[dict[int, int]]:
    """Frames whose single label rises and falls, in voxel counts."""
    return [
        {1: int(peak * (0.6 + 0.4 * np.sin(2 * np.pi * frame / frames)))}
        for frame in range(frames)
    ]


def measured(names=("LV",), bsa=None, **group_kwargs) -> Result:
    counts = curve(1000)
    groups = [group(name, 1, **group_kwargs) for name in names]
    return measure(counts, 0.001, groups, bsa=bsa)


@pytest.fixture
def views() -> VolumetryViews:
    view = VolumetryViews()
    view.window.SetSize(*SIZE)
    return view


def rendered(views: VolumetryViews) -> np.ndarray:
    """The window as pixels, top row first -- the path a capture takes."""
    views.window.Render()
    filter = vtk.vtkWindowToImageFilter()
    filter.SetInput(views.window)
    filter.SetInputBufferTypeToRGB()
    filter.Modified()
    filter.Update()

    image = filter.GetOutput()
    columns, rows, _ = image.GetDimensions()
    flat = vtknp.vtk_to_numpy(image.GetPointData().GetScalars())
    return np.flipud(flat.reshape(rows, columns, -1))


def drawn(pixels: np.ndarray) -> np.ndarray:
    """Which pixels are something other than the black the window opens on."""
    return pixels.sum(axis=2) > 0


def text_lines(pixels: np.ndarray, least: int = 4) -> int:
    """How many bands of lettering there are down ``pixels``.

    Runs shorter than ``least`` are not counted: a rule is two pixels and
    antialiasing leaves one under a row of digits, and neither is a line of
    text. Counting exact runs instead makes the count turn on the font.
    """
    inked = drawn(pixels).any(axis=1)
    edges = np.diff(np.concatenate(([0], inked.astype(int), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return int(((ends - starts) >= least).sum())


# --- paging ---


def test_a_page_is_built_for_the_first_structure_on_show(views):
    views.show(measured(names=("LV", "RV", "Myo")))

    assert views.pages == 3
    assert views.page == 0
    assert views.chart.GetTitle() == "LV"


def test_turning_the_page_draws_another_structure(views):
    views.show(measured(names=("LV", "RV", "Myo")))
    views.set_page(2)

    assert views.page == 2
    assert views.chart.GetTitle() == "Myo"


def test_a_page_replaces_the_one_before_it_rather_than_joining_it(views):
    """Both are drawn into one scene, so a page left behind would be drawn over
    the next one rather than simply wasted."""
    views.show(measured(names=("LV", "RV")))
    first = views.chart
    views.set_page(1)

    assert views.chart is not first
    assert views._chart_scene.GetNumberOfItems() == 1


def test_a_page_past_the_end_is_held_at_the_last_one(views):
    views.show(measured(names=("LV", "RV")))
    views.set_page(9)

    assert views.page == 1


def test_showing_a_second_measurement_starts_again_at_its_first_page(views):
    views.show(measured(names=("LV", "RV", "Myo")))
    views.set_page(2)
    views.show(measured(names=("LA",)))

    assert (views.pages, views.page) == (1, 0)
    assert views.chart.GetTitle() == "LA"


# --- the split ----------------------------------------------------------------


def test_the_table_is_given_more_of_the_window_as_it_gains_rows():
    assert band_fraction(2) < band_fraction(5)


def test_the_table_never_takes_the_window_from_the_curve():
    """A page with many metrics still has to show the curve they came from."""
    assert band_fraction(40) <= 0.4


def test_the_chart_and_the_table_get_a_viewport_each(views):
    views.show(measured(names=("LV", "RV")))
    chart = views._chart_renderer.GetViewport()
    table = views._table_renderer.GetViewport()

    assert table[1] == 0.0
    assert chart[3] == 1.0
    assert chart[1] == pytest.approx(table[3])


# --- what actually lands on the pixels ----------------------------------------


def test_the_window_renders_something_other_than_its_background(views):
    """Offscreen, through EGL, with a python-backed item in the scene.

    The same path ``WindowFrames`` takes, so a capture of this viewport is
    covered by the same assertion.
    """
    views.show(measured(names=("LV", "RV", "Myo")))
    assert drawn(rendered(views)).sum() > 1000


def test_the_numbers_are_drawn_below_the_curves(views):
    """The band is the table's, and the charts must not be laid over it."""
    views.show(measured(names=("LV", "RV")))
    pixels = rendered(views)
    rows = pixels.shape[0]
    band = round(rows * (1.0 - band_fraction(2)))

    assert drawn(pixels[band:]).sum() > 100
    assert drawn(pixels[:band]).sum() > 1000


def test_every_metric_gets_a_line_of_its_own_in_the_table(views):
    """A context scene is asked for its size two ways, and only one is right.

    ``GetViewWidth``/``GetViewHeight`` are the whole window; this renderer's
    own viewport is ``GetSceneWidth``/``GetSceneHeight``.  Laid out against the
    first, the rows are spaced over a band several times the height of the one
    they are drawn in, so all but the bottom two are clipped away and the rest
    are set in headlines -- which looks like a styling choice rather than a
    wrong accessor.  Counting the lines that actually land is what tells them
    apart.
    """
    views.show(measured(names=("LV",)))
    views._chart_scene.RemoveItem(views.chart)

    pixels = rendered(views)
    band = round(pixels.shape[0] * (1.0 - band_fraction(len(views._table.rows))))

    # The header, and one line per metric: EDV, ESV, SV, EF.
    assert text_lines(pixels[band:]) == 5


def test_the_charts_are_captured_against_a_ground_rather_than_a_hole(views):
    """A PNG capture carries an alpha channel, and a renderer's default is zero.

    Every other viewport is a cut-out of a scene, where that is what lets one
    be composited. A chart is nine tenths background: captured transparent, the
    white lettering arrives as white on whatever the reader opens it over, and
    the whole thing reads as blank.
    """
    views.show(measured(names=("LV", "RV")))
    views.window.Render()

    filter = vtk.vtkWindowToImageFilter()
    filter.SetInput(views.window)
    filter.SetInputBufferTypeToRGBA()
    filter.Modified()
    filter.Update()

    image = filter.GetOutput()
    alpha = vtknp.vtk_to_numpy(image.GetPointData().GetScalars())[:, 3]
    assert (alpha == 255).all()


def test_a_curve_is_drawn_in_its_structures_own_colour(views):
    """So the chart and the overlay of the same label agree."""
    views.show(measured(names=("LV",)))
    rgb = byte_color(label_color(1))
    pixels = rendered(views).reshape(-1, 3)
    assert (np.abs(pixels.astype(int) - np.array(rgb)).sum(axis=1) < 30).any()


def test_a_window_with_nothing_measured_still_renders(views):
    """A scene that configured no structures opens on the viewport like any
    other, and must not raise on its way to an empty chart."""
    views.show(Result(measurements=[], bsa=None, frames=0))
    assert drawn(rendered(views)).sum() == 0


# --- the frame rule -----------------------------------------------------------


def test_moving_the_frame_marker_leaves_the_curve_alone(views):
    """A curve covers every frame, so a frame change is a position, not a
    measurement -- and a cine must not rebuild a chart per beat."""
    views.show(measured(names=("LV", "RV")))
    before = views.chart

    views.set_frame(4)
    assert views.chart is before


def test_the_frame_marker_moves_where_it_is_put(views):
    views.show(measured(names=("LV",)))
    views.set_frame(3)
    assert views._rule.GetValue(0, 0).ToFloat() == pytest.approx(3.0)
    views.set_frame(9)
    assert views._rule.GetValue(0, 0).ToFloat() == pytest.approx(9.0)


def test_the_frame_marker_spans_the_whole_of_its_charts_axis(views):
    """Pinned axes are why the rule can be drawn as a two-point line at all."""
    views.show(measured(names=("LV",)))
    views.set_frame(2)
    low, high = views._range
    assert views._rule.GetValue(0, 1).ToFloat() == pytest.approx(low)
    assert views._rule.GetValue(1, 1).ToFloat() == pytest.approx(high)


def test_the_marker_stays_put_when_the_page_turns(views):
    """The frame is the app's, not the page's: turning to another structure
    does not move the reader to another moment in the cycle."""
    views.show(measured(names=("LV", "RV")))
    views.set_frame(5)
    views.set_page(1)

    assert views._rule.GetValue(0, 0).ToFloat() == pytest.approx(5.0)


# --- the table ----------------------------------------------------------------


def test_the_table_carries_a_row_for_every_metric(views):
    views.show(measured(names=("LV",)))

    assert [row[0] for row in views._table.rows] == ["EDV", "ESV", "SV", "EF"]


def test_a_value_carries_the_unit_its_metric_is_in(views):
    """The pivot is what makes this necessary and possible: a column of values
    down a page holds millilitres and a percentage, so the unit cannot sit in
    the heading the way it did when each row was a structure."""
    views.show(measured(names=("Myo",), density=1.05))
    values = dict(row[:2] for row in views._table.rows)

    assert values["EDV"].endswith(" g")
    assert values["EF"].endswith(" %")


def test_a_structure_that_does_not_pump_is_read_as_extremes_and_nothing_more(views):
    """Nothing here knows what phase anything was acquired at, so the cardiac
    vocabulary is a reader's claim and not the measurement's: an aorta or a
    lung has a maximum, and no stroke volume underneath it to wonder about."""
    views.show(measured(names=("Ao",), chamber=False))

    assert [row[0] for row in views._table.rows] == ["Maximum", "Minimum"]


def test_the_indexed_column_appears_only_once_a_body_surface_area_is_known(views):
    views.show(measured(names=("LV",)))
    assert len(views._table.header) == 2

    views.show(measured(names=("LV",), bsa=1.9))
    assert len(views._table.header) == 3


def test_an_indexed_value_is_the_value_over_the_area(views):
    views.show(measured(names=("LV",), bsa=2.0))
    plain, indexed = {row[0]: (row[1], row[2]) for row in views._table.rows}["EDV"]

    assert float(indexed.split()[0]) == pytest.approx(
        float(plain.split()[0]) / 2.0, abs=0.05
    )


def test_an_ejection_fraction_is_not_indexed(views):
    """It is already a ratio, so dividing it by an area says nothing."""
    views.show(measured(names=("LV",), bsa=2.0))

    assert {row[0]: row[2] for row in views._table.rows}["EF"] == ABSENT
