"""When the pages are measured, when they are re-marked, and what is written.

The gap this guards: a curve covers every frame, so the expensive half of this
feature has no business running per frame -- and the cheap half has to run
exactly when the frame moves, from here rather than from the frame path, which
renders before this controller has been told.  Both are invisible when wrong:
the charts look right either way, and what changes is whether a cine costs a
measurement a beat and whether the rule sits where the reader is looking.

The report is the other half: it is a page per structure rather than a capture
of what is on screen, so it has to render pages nobody asked to look at and put
the reader back where they were afterwards.
"""

# System
import csv
import itertools as it

# Third Party
import pytest

# Internal
from cardio.volumetry import StructureGroup, Volumetry
from tests.test_app_smoke import build_app, build_scene, connect

GROUPS = [
    StructureGroup(name="One", labels=[1]),
    StructureGroup(name="Two", labels=[2, 3], density=1.05, cycle=False),
]

_counter = it.count()


def built(tmp_path, layout="volumetry", **volumetry):
    """A real app on a scene whose segmentation carries three labels."""
    directory = tmp_path / f"scene{next(_counter)}"
    directory.mkdir()
    scene = build_scene(
        directory,
        # Under the test's own directory: left at its default this writes into
        # ./data, where one test's export is the next one's leftovers.
        serialization_directory=directory / "out",
        volumetry=Volumetry(groups=GROUPS, **volumetry),
        view={"layout": layout},
    )
    server, _, logic, _ = build_app(scene)
    connect(server)
    return server, scene, logic


@pytest.fixture
def app(tmp_path):
    return built(tmp_path)


# --- what is measured, and when -----------------------------------------------


def test_the_pages_are_drawn_from_the_labels_the_segmentation_carries(app):
    _, _, logic = app

    assert [m.group.name for m in logic.volumetry.result().measurements] == [
        "One",
        "Two",
    ]
    assert logic.volumetry.views.pages == 2
    assert logic.volumetry.views.chart.GetTitle() == "One"


def test_a_curve_carries_a_point_for_every_frame(app):
    _, scene, logic = app
    result = logic.volumetry.result()

    assert result.frames == len(scene.segmentations[0].actors)
    assert len(result.of("One").values) == result.frames


def test_nothing_is_measured_while_the_charts_are_off_screen(tmp_path):
    """A cine must not pay for a view nobody is showing.

    The whole of this controller returns on the layout, so the scene opening
    somewhere else is the case that proves it rather than an incidental one.
    """
    _, _, logic = built(tmp_path, layout="quad")

    assert logic.volumetry._result is None
    assert logic.volumetry.views.chart is None


def test_the_pages_are_drawn_when_the_layout_comes_round_to_them(tmp_path):
    server, _, logic = built(tmp_path, layout="quad")

    with server.state:
        server.state.maximized_view = "volumetry"

    assert logic.volumetry.views.chart is not None


def test_a_scene_that_configures_no_structures_measures_nothing(tmp_path):
    directory = tmp_path / "bare"
    directory.mkdir()
    scene = build_scene(directory, view={"layout": "volumetry"})
    server, _, logic, _ = build_app(scene)
    connect(server)

    assert logic.volumetry.result() is None
    assert logic.volumetry.views.chart is None


# --- what a frame change costs ------------------------------------------------


def test_a_frame_change_moves_the_rule_rather_than_the_curve(app):
    server, _, logic = app
    before = logic.volumetry.views.chart
    measured = logic.volumetry._result

    with server.state:
        server.state.frame = 1

    assert logic.volumetry.views.chart is before
    assert logic.volumetry._result is measured


def test_the_rule_lands_on_the_frame_the_rest_of_the_app_is_showing(app):
    server, _, logic = app

    with server.state:
        server.state.frame = 1

    assert logic.volumetry.views._rule.GetValue(0, 0).ToFloat() == pytest.approx(1.0)


def test_the_charts_are_redrawn_by_this_controller_and_not_by_the_frame_path(app):
    """Playback registers its listener first, so the render it asks for
    happens before the rule has moved.  Left to it, the marker would sit one
    frame behind whatever the slices are showing."""
    server, _, logic = app
    drawn = []
    server.controller.volumetry_update = lambda *a, **k: drawn.append(
        logic.volumetry.views._rule.GetValue(0, 0).ToFloat()
    )

    with server.state:
        server.state.frame = 1

    assert drawn[-1] == pytest.approx(1.0)


def test_a_frame_change_off_screen_measures_nothing_and_moves_nothing(tmp_path):
    """The frame path pushes to every render view whatever the layout is, so
    what says the charts are idle is that nothing here was measured and no
    rule exists to have been moved."""
    server, _, logic = built(tmp_path, layout="quad")

    with server.state:
        server.state.frame = 1

    assert logic.volumetry._result is None
    assert logic.volumetry.views._rule is None


# --- what invalidates a measurement -------------------------------------------


def test_choosing_another_segmentation_drops_what_was_measured_off_the_first(app):
    server, _, logic = app
    assert logic.volumetry._result is not None

    with server.state:
        server.state.volumetry_seg_label = "other"

    assert logic.volumetry._result is None


def test_choosing_another_structure_turns_to_its_page(app):
    server, _, logic = app

    with server.state:
        server.state.volumetry_structure = "Two"

    assert logic.volumetry.views.chart.GetTitle() == "Two"


def test_turning_the_page_does_not_measure_anything_again(app):
    """The measurement covers every structure; a page is a view of it."""
    server, _, logic = app
    measured = logic.volumetry._result

    with server.state:
        server.state.volumetry_structure = "Two"

    assert logic.volumetry._result is measured


def test_the_structures_are_offered_in_the_order_they_are_configured(app):
    server, _, _ = app

    assert server.state.volumetry_structures == ["One", "Two"]


def test_the_page_is_not_rebuilt_when_nothing_about_it_changed(app):
    """Refresh is a listener on the layout, which a maximized view rewrites."""
    server, _, logic = app
    before = logic.volumetry.views.chart

    with server.state:
        server.state.maximized_view = "volumetry"

    assert logic.volumetry.views.chart is before


# --- what the volumes are indexed by ------------------------------------------


def test_a_scene_that_knows_no_body_size_indexes_nothing(app):
    """The phantom carries no PatientSize, and a guessed one would be worse
    than none: an indexed volume against somebody else's body reads as a
    finding."""
    _, _, logic = app

    assert logic.volumetry.body_surface_area() is None
    assert logic.volumetry.result().bsa is None


def test_a_configured_height_and_weight_index_the_volumes(tmp_path):
    _, _, logic = built(tmp_path, patient_height_m=1.78, patient_weight_kg=74.0)

    assert logic.volumetry.result().bsa == pytest.approx(1.9128, abs=1e-3)


def test_indexing_can_be_switched_off_for_a_body_whose_size_is_known(tmp_path):
    server, _, logic = built(tmp_path, patient_height_m=1.78, patient_weight_kg=74.0)

    with server.state:
        server.state.volumetry_indexed = False

    assert logic.volumetry.result().bsa is None


# --- what the drawer is told --------------------------------------------------


def test_the_drawer_lists_the_shown_pages_metrics(app):
    """The page's rather than every structure's: the drawer sits beside one
    chart, and a table of all of them there would be the crowding the pages
    exist to undo."""
    server, _, _ = app

    assert [row["Metric"] for row in server.state.volumetry_rows] == [
        "EDV",
        "ESV",
        "SV",
        "EF",
    ]


def test_the_drawer_follows_the_page_it_is_beside(app):
    server, _, _ = app
    before = server.state.volumetry_rows

    with server.state:
        server.state.volumetry_structure = "Two"

    assert server.state.volumetry_rows != before


def test_the_rows_gain_their_indexed_column_with_a_body_surface_area(tmp_path):
    server, _, _ = built(tmp_path, patient_height_m=1.78, patient_weight_kg=74.0)

    assert "Indexed" in server.state.volumetry_rows[0]


# --- what an export writes ----------------------------------------------------


def read(path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def exported(logic, scene):
    """The one directory a save wrote into."""
    logic.dispatch("save_volumetry")
    return next(iter(scene.volumetry_directory.iterdir()))


def test_an_export_writes_the_tables_and_a_page_for_every_structure(app):
    _, scene, logic = app
    directory = exported(logic, scene)

    assert sorted(path.name for path in directory.iterdir()) == [
        "metrics.csv",
        "pages",
        "timeseries.csv",
    ]
    assert len(list((directory / "pages").iterdir())) == 2


def test_an_export_leaves_the_reader_on_the_page_they_were_looking_at(app):
    """It renders pages nobody asked to see, and has to put the view back."""
    server, scene, logic = app
    with server.state:
        server.state.volumetry_structure = "Two"

    exported(logic, scene)

    assert logic.volumetry.views.chart.GetTitle() == "Two"


def test_an_export_writes_its_pages_in_the_format_the_capture_is_set_to(app):
    """The report is a sequence, which is what the capture writers take: asking
    for dicom-rendered is what makes it a Secondary Capture series."""
    server, scene, logic = app
    with server.state:
        server.state.capture_format = "dicom-rendered"

    pages = sorted((exported(logic, scene) / "pages").iterdir())
    assert [path.suffix for path in pages] == [".dcm", ".dcm"]


def test_the_curves_are_written_one_row_per_frame(app):
    _, scene, logic = app
    header, rows = read(exported(logic, scene) / "timeseries.csv")

    assert header == ["frame", "One_mL", "Two_g"]
    assert len(rows) == logic.volumetry.result().frames


def test_the_metrics_are_written_one_row_per_structure(app):
    _, scene, logic = app
    header, rows = read(exported(logic, scene) / "metrics.csv")

    assert [row[0] for row in rows] == ["One", "Two"]
    assert header[:2] == ["structure", "unit"]


def test_a_number_that_would_mean_nothing_is_left_empty_rather_than_dashed(app):
    """A spreadsheet reading a dash in a column of numbers has a parse error,
    where the on-screen table has a reader who knows what a dash means."""
    _, scene, logic = app
    header, rows = read(exported(logic, scene) / "metrics.csv")

    outside = next(row for row in rows if row[0] == "Two")
    assert outside[header.index("SV")] == ""


def test_the_indexed_columns_are_written_only_once_there_is_a_body_to_index_by(
    tmp_path,
):
    _, scene, logic = built(tmp_path, patient_height_m=1.78, patient_weight_kg=74.0)
    header, _ = read(exported(logic, scene) / "metrics.csv")

    assert "EDVi" in header and "BSA_m2" in header


def test_an_export_says_what_it_wrote_and_when(app):
    server, scene, logic = app
    exported(logic, scene)

    assert server.state.volumetry_ok
    assert "2 pages" in server.state.volumetry_summary
    assert server.state.volumetry_saved_at


def test_a_scene_with_nothing_configured_writes_nothing_and_says_so(tmp_path):
    """Better than an empty pair of files, which read as a measurement of zero."""
    directory = tmp_path / "bare"
    directory.mkdir()
    scene = build_scene(
        directory,
        serialization_directory=directory / "out",
        view={"layout": "volumetry"},
    )
    server, _, logic, _ = build_app(scene)
    connect(server)

    logic.dispatch("save_volumetry")

    assert not server.state.volumetry_ok
    assert not scene.volumetry_directory.exists()


def test_a_report_can_be_written_before_the_charts_have_ever_been_laid_out(
    tmp_path,
):
    """The page sizes a viewport when it lays one out, and a hidden container
    is never laid out -- so the window is nothing by nothing until the reader
    first opens it, and a report asked for before that captured empty buffers.
    """
    _, scene, logic = built(tmp_path, layout="quad")
    assert not all(logic.volumetry.views.window.GetSize())

    directory = exported(logic, scene)

    assert logic.volumetry.views.window.GetSize() == tuple(scene.headless_size)
    assert len(list((directory / "pages").iterdir())) == 2


def test_an_export_does_not_need_the_charts_to_be_on_screen(tmp_path):
    """A headless session asks for the report without arranging to look at it,
    which is most of what a script does with this -- so the pages have to be
    drawn from the measurement rather than from whatever the layout drew."""
    _, scene, logic = built(tmp_path, layout="quad")

    directory = exported(logic, scene)

    assert len(list((directory / "pages").iterdir())) == 2
