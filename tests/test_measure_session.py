"""Tracing a region through the whole app, with no page to click on.

A session is the app without a browser, which is what makes the claim under
this feature testable: a click is a display position, and what it becomes is a
number of square millimetres. Here the clicks are put in by hand at positions
whose answer is known, so the area is checked against arithmetic rather than
against whatever the app last said.
"""

# System
import itertools
import pathlib as pl

# Third Party
import numpy as np
import pytest

# Internal
import cardio.planimetry as planimetry
from cardio.camera import cut_point, world_per_pixel
from cardio.measurement import MeasurementSet
from cardio.planimetry import ContourStyle
from cardio.session import Session
from cardio.tile import TileSource
from tests.test_app_smoke import build_scene, write_volume

_names = itertools.count()


def session_on(directory: pl.Path, **overrides) -> Session:
    """A session whose output lands under ``directory`` rather than in ./data.

    ``serialization_directory`` defaults to a relative path, so a test that
    saves without setting it writes into the working tree -- and then reads the
    files the test before it left there.
    """
    overrides.setdefault("serialization_directory", directory / "out")
    return Session(
        build_scene(directory, active_volume_label="vol", **overrides),
        server=f"measure-{next(_names)}",
    )


@pytest.fixture
def armed(tmp_path) -> Session:
    """A session with its windows built and sized, as a page would leave them."""
    session = session_on(tmp_path)
    session.ready()
    return session


def renderer(session: Session, view: str = "ul"):
    return session.scene.mpr_views.renderer(view)


def regions(session: Session):
    return session.logic.measurements.regions.measurements


def trace(session: Session, corners, view: str = "ul", close: bool = True):
    """Place a point at each display position, then close the region.

    The mode is entered only if it is not already on: closing a region leaves
    it on, so that several can be traced one after another without reaching
    for the key between them.
    """
    if not session.server.state.measuring:
        session.do("toggle_measuring")
    for x, y in corners:
        session.do("place_measurement_point", view_name=view, x=float(x), y=float(y))
    if close:
        session.do("close_measurement")
    return session


def box(session: Session, half: float, view: str = "ul"):
    """Four display positions ``2 * half`` pixels apart, about the view's centre."""
    width, height = renderer(session, view).GetSize()
    x, y = width / 2.0, height / 2.0
    return [
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
    ]


# --- what a trace comes to ----------------------------------------------------


def test_a_traced_square_measures_the_square_it_was_traced_as(armed):
    """The one number the whole feature exists to produce.

    A hundred pixels each way at the view's own scale is a square of that many
    world units, and its area is the side squared -- which is what the app has
    to say without being told the scale.
    """
    scale = world_per_pixel(renderer(armed))
    trace(armed, box(armed, 50.0))

    assert len(regions(armed)) == 1
    assert regions(armed)[0].area == pytest.approx((100.0 * scale) ** 2, rel=1e-3)


def test_the_points_land_where_the_clicks_were(armed):
    corners = box(armed, 40.0)
    trace(armed, corners)

    expected = [cut_point(renderer(armed), x, y) for x, y in corners]
    assert np.array(regions(armed)[0].points) == pytest.approx(
        np.array(expected), abs=1e-6
    )


def test_a_region_is_filed_against_the_cut_it_was_traced_on(armed):
    trace(armed, box(armed, 30.0))
    region = regions(armed)[0]

    assert region.view == "ul"
    assert region.cut == armed.logic.measurements.cut_of("ul")
    assert region.on(armed.logic.measurements.cut_of("ul"), frame=region.frame)


def test_the_traced_points_can_be_put_back_in_the_patient(armed):
    trace(armed, box(armed, 30.0))
    region = regions(armed)[0]

    assert region.lps().shape == (4, 3)
    # Round trip through the cut it was traced on, which is what a reader does.
    assert planimetry.from_lps(region.cut, region.lps()) == pytest.approx(
        np.array(region.points)
    )


def test_a_region_is_named_if_it_is_not_given_one(armed):
    trace(armed, box(armed, 30.0))
    assert regions(armed)[0].name == "Region 1"


def test_a_region_keeps_the_name_it_was_closed_with(armed):
    session = trace(armed, box(armed, 30.0), close=False)
    session.do("close_measurement", name="AVA")

    assert regions(session)[0].name == "AVA"


# --- the mode -----------------------------------------------------------------


def test_a_click_outside_the_mode_places_nothing(armed):
    armed.do("place_measurement_point", view_name="ul", x=100.0, y=100.0)

    assert armed.server.state.measurement_pending == 0
    assert regions(armed) == []


def test_leaving_the_mode_abandons_a_half_traced_region(armed):
    trace(armed, box(armed, 30.0), close=False)
    assert armed.server.state.measurement_pending == 4

    armed.do("toggle_measuring")

    assert armed.server.state.measurement_pending == 0
    assert regions(armed) == []


def test_a_region_of_two_points_is_refused(armed):
    armed.do("toggle_measuring")
    for x, y in box(armed, 30.0)[:2]:
        armed.do("place_measurement_point", view_name="ul", x=x, y=y)

    armed.do("close_measurement")

    assert regions(armed) == []
    assert armed.server.state.measurement_pending == 2, "still being traced"


def test_closing_with_nothing_traced_does_nothing(armed):
    armed.do("toggle_measuring")
    armed.do("close_measurement")

    assert regions(armed) == []


def test_undo_takes_back_one_point_at_a_time(armed):
    trace(armed, box(armed, 30.0), close=False)

    armed.do("undo_measurement_point")
    assert armed.server.state.measurement_pending == 3

    armed.do("undo_measurement_point")
    armed.do("undo_measurement_point")
    armed.do("undo_measurement_point")
    assert armed.server.state.measurement_pending == 0
    assert armed.server.state.measurement_view == ""


def test_cancel_drops_the_whole_trace(armed):
    trace(armed, box(armed, 30.0), close=False)

    armed.do("cancel_measurement")

    assert armed.server.state.measurement_pending == 0
    assert regions(armed) == []


# --- which cut it belongs on --------------------------------------------------


def test_a_region_leaves_the_screen_when_the_cut_scrolls_off_its_plane(armed):
    trace(armed, box(armed, 30.0))
    assert armed.server.state.measurement_on_plane == [True]

    armed.do("scroll_slice", view_name="ul", distance=5.0)

    assert armed.server.state.measurement_on_plane == [False]


def test_a_region_comes_back_when_the_cut_returns_to_its_plane(armed):
    trace(armed, box(armed, 30.0))
    armed.do("scroll_slice", view_name="ul", distance=5.0)
    armed.do("scroll_slice", view_name="ul", distance=-5.0)

    assert armed.server.state.measurement_on_plane == [True]


def test_a_region_stays_on_screen_when_the_cut_pans_within_its_plane(armed):
    """A pan moves the camera and the origin within the plane, not the plane."""
    trace(armed, box(armed, 30.0))

    armed.do("pan_view", view_name="ul", dx=12.0, dy=7.0)

    assert armed.server.state.measurement_on_plane == [True]


def test_a_pan_does_not_move_the_region_off_the_anatomy(armed):
    """The points are re-cut into the moved frame, so the patient sees no change."""
    trace(armed, box(armed, 30.0))
    before = regions(armed)[0].lps()

    armed.do("pan_view", view_name="ul", dx=12.0, dy=7.0)

    assert regions(armed)[0].lps() == pytest.approx(before)


def test_a_region_leaves_the_screen_when_the_cut_turns_out_of_its_plane(armed):
    trace(armed, box(armed, 30.0))

    armed.do("add_rotation", axis="X")
    armed.do("set_state", key="mpr_rotation_data.angles_list.0.angle", value=20.0)

    assert armed.server.state.measurement_on_plane == [False]


def test_a_region_traced_in_one_pane_is_not_drawn_in_another(armed):
    """The three cuts are mutually perpendicular, so only one holds the region."""
    trace(armed, box(armed, 30.0))
    measurements = armed.logic.measurements

    on = [
        view
        for view in ("ul", "ll", "lr")
        if regions(armed)[0].on(measurements.cut_of(view), frame=0)
    ]
    assert on == ["ul"]


# --- a trace that spans a change ----------------------------------------------


def test_a_point_placed_on_another_cut_is_refused(armed):
    """The points already down are measured in a cut this one is not."""
    armed.do("toggle_measuring")
    for x, y in box(armed, 30.0)[:3]:
        armed.do("place_measurement_point", view_name="ul", x=x, y=y)

    armed.do("place_measurement_point", view_name="ll", x=200.0, y=150.0)

    assert armed.server.state.measurement_pending == 3
    assert armed.server.state.measurement_view == "ul"


def test_a_pan_between_two_points_leaves_the_earlier_ones_on_the_anatomy(armed):
    """A pan mid-trace moves the cut within its own plane, and the points
    already placed stay where the patient is.

    Not where the *screen* is: a later click lands on whatever the view is now
    showing at that spot, which is the anatomy under the cursor rather than the
    anatomy that was under it before. What must not happen is the earlier
    points sliding, which is why they are kept in the traced cut's own
    millimetres rather than re-read from the moved one.
    """
    corners = box(armed, 50.0)

    armed.do("toggle_measuring")
    for x, y in corners[:2]:
        armed.do("place_measurement_point", view_name="ul", x=x, y=y)
    placed = planimetry.to_lps(
        armed.logic.measurements.cut_of("ul"), armed.logic.measurements._points
    )

    armed.do("pan_view", view_name="ul", dx=25.0, dy=15.0)
    for x, y in corners[2:]:
        armed.do("place_measurement_point", view_name="ul", x=x, y=y)
    armed.do("close_measurement")

    assert len(regions(armed)) == 1
    assert regions(armed)[0].lps()[:2] == pytest.approx(placed)


# --- staying on the view ------------------------------------------------------

# Every path that redraws a cut clears the renderer first, so a region has to be
# put back by each of them.  The two that matter are the ones that reach the
# renderers without going through a state listener: a frame of a cine, and a
# segmentation overlay being switched on.


def props(session: Session, view: str = "ul") -> int:
    return renderer(session, view).GetViewProps().GetNumberOfItems()


def contour_props(session: Session, view: str = "ul") -> int:
    """How many of the view's props belong to the traced regions."""
    holder = renderer(session, view).GetViewProps()
    drawn = [holder.GetItemAsObject(i) for i in range(holder.GetNumberOfItems())]
    return sum(1 for prop in drawn if not prop.GetUseBounds())


def test_a_region_is_drawn_on_the_view_it_was_traced_on(armed):
    trace(armed, box(armed, 30.0))

    assert contour_props(armed) == 3, "the curve, its points and its area"
    assert contour_props(armed, "ll") == 0


def test_a_region_survives_a_frame_of_a_cine(tmp_path):
    """The frame path calls the controllers directly rather than through state."""
    for frame in (1, 2):
        write_volume(tmp_path / f"vol{frame}.nii.gz")
    session = session_on(
        tmp_path,
        volumes=[
            {
                "label": "vol",
                "directory": tmp_path,
                "file_paths": ["vol0.nii.gz", "vol1.nii.gz", "vol2.nii.gz"],
            }
        ],
    )
    session.ready()
    trace(session, box(session, 30.0))

    session.do("increment_frame")

    assert session.server.state.frame == 1
    assert session.server.state.measurement_on_plane == [False], "another phase"

    session.do("decrement_frame")

    assert session.server.state.measurement_on_plane == [True]
    assert contour_props(session) == 3


def test_a_region_survives_a_segmentation_overlay_being_switched_on(armed):
    """That path clears the renderers without going through ``show``."""
    trace(armed, box(armed, 30.0))

    armed.do("set_state", key="mpr_segmentation_overlay_seg", value=True)

    assert contour_props(armed) == 3


def test_a_region_survives_the_layout_leaving_the_cuts_and_coming_back(armed):
    trace(armed, box(armed, 30.0))

    armed.do("toggle_maximized", view="volume")
    assert contour_props(armed) == 0, "nothing left on a view nobody is drawing"

    armed.do("toggle_maximized", view="volume")

    assert contour_props(armed) == 3


def test_a_camera_reset_does_not_frame_the_region_instead_of_the_cut(armed):
    """A contour that voted on the bounds would zoom the view onto itself."""
    trace(armed, box(armed, 30.0))
    before = world_per_pixel(renderer(armed))

    armed.scene.mpr_views.reset_cameras()

    assert world_per_pixel(renderer(armed)) == pytest.approx(before, rel=1e-6)


def test_a_region_off_its_plane_leaves_nothing_behind_on_the_view(armed):
    trace(armed, box(armed, 30.0))

    armed.do("scroll_slice", view_name="ul", distance=5.0)

    assert contour_props(armed) == 0


# --- the closure style --------------------------------------------------------


def test_a_region_is_closed_with_the_style_that_was_chosen(armed):
    armed.do("set_state", key="measurement_contour", value="spline")
    trace(armed, box(armed, 50.0))

    assert regions(armed)[0].contour is ContourStyle.SPLINE


def test_a_splined_region_encloses_more_than_the_points_it_was_traced_from(armed):
    trace(armed, box(armed, 50.0))
    straight = regions(armed)[0].area

    armed.do("set_state", key="measurement_contour", value="spline")
    trace(armed, box(armed, 50.0))

    assert regions(armed)[1].area > straight


def test_changing_the_style_leaves_a_region_already_closed_alone(armed):
    """The picker says how the *next* region is closed.

    A measurement records the curve it was measured as. If moving a picker
    restyled the regions already taken, every area recorded before it would
    change without anybody asking -- which for a study comparing measurements
    is the one thing that must not happen quietly. Changing one is done to that
    region, by name, and marks the set unsaved.
    """
    trace(armed, box(armed, 50.0))
    before = regions(armed)[0].area

    armed.do("set_state", key="measurement_contour", value="spline")

    assert regions(armed)[0].contour is ContourStyle.POLYGON
    assert regions(armed)[0].area == pytest.approx(before)


def test_closing_a_region_leaves_the_mode_on_for_the_next(armed):
    """Several regions in a row is the normal case, not the exception."""
    trace(armed, box(armed, 50.0))

    assert armed.server.state.measuring
    assert armed.server.state.measurement_pending == 0, "and nothing half traced"


# --- changing a region already taken ------------------------------------------


def test_restyling_a_region_measures_it_again(armed):
    trace(armed, box(armed, 50.0))
    straight = regions(armed)[0].area

    armed.do("restyle_measurement", index=0, contour="spline")

    assert regions(armed)[0].contour is ContourStyle.SPLINE
    assert regions(armed)[0].area > straight


def test_restyling_leaves_the_points_where_they_were_placed(armed):
    """The clicks are the measurement; the style is how they are joined up."""
    trace(armed, box(armed, 50.0))
    placed = regions(armed)[0].points

    armed.do("restyle_measurement", index=0, contour="spline")

    assert regions(armed)[0].points == placed


def test_restyling_does_not_change_the_default_for_the_next_region(armed):
    trace(armed, box(armed, 50.0))

    armed.do("restyle_measurement", index=0, contour="spline")
    trace(armed, box(armed, 30.0))

    assert regions(armed)[1].contour is ContourStyle.POLYGON


def test_a_region_can_be_renamed(armed):
    trace(armed, box(armed, 50.0))

    armed.do("rename_measurement", index=0, name="AVA systole")

    assert regions(armed)[0].name == "AVA systole"


def test_deleting_a_region_takes_its_outline_with_it(armed):
    trace(armed, box(armed, 50.0))
    trace(armed, box(armed, 30.0))
    assert contour_props(armed) == 6

    armed.do("delete_measurement", index=0)

    assert len(regions(armed)) == 1
    assert contour_props(armed) == 3, "one region, one outline"


def test_deleting_the_first_of_two_leaves_the_right_one_behind(armed):
    """The props are keyed by position, so a delete has to shuffle them."""
    trace(armed, box(armed, 50.0))
    armed.do("rename_measurement", index=1 - 1, name="first")
    trace(armed, box(armed, 30.0))
    armed.do("rename_measurement", index=1, name="second")

    armed.do("delete_measurement", index=0)

    assert [region.name for region in regions(armed)] == ["second"]
    assert regions(armed)[0].area == pytest.approx(
        (60.0 * world_per_pixel(renderer(armed))) ** 2, rel=1e-3
    )


def test_clearing_takes_every_region_and_every_outline(armed):
    trace(armed, box(armed, 50.0))
    trace(armed, box(armed, 30.0))

    armed.do("clear_measurements")

    assert regions(armed) == []
    assert contour_props(armed) == 0


@pytest.mark.parametrize("action", ["delete_measurement", "recall_measurement"])
def test_an_index_that_is_not_there_is_refused(armed, action):
    trace(armed, box(armed, 50.0))

    armed.do(action, index=7)

    assert len(regions(armed)) == 1


# --- correcting a region already taken ----------------------------------------


def edit(session: Session, index: int = 0):
    """Open a region for correction, and say where its points are on screen."""
    session.do("edit_measurement", index=index)
    return session


def corner(session: Session, half: float, which: int, view: str = "ul"):
    """The display position of one corner of ``box``."""
    return box(session, half, view)[which]


def drag(session: Session, start, end, view: str = "ul", moves: int = 1):
    """Press at ``start``, travel to ``end`` in ``moves`` steps, and let go."""
    grabbed = session.logic.dispatch(
        "grab_measurement_point", view_name=view, x=float(start[0]), y=float(start[1])
    )
    for step in range(1, moves + 1):
        session.do(
            "drag_measurement_point",
            view_name=view,
            x=float(start[0] + (end[0] - start[0]) * step / moves),
            y=float(start[1] + (end[1] - start[1]) * step / moves),
        )
    session.do("drop_measurement_point")
    return grabbed


def test_a_region_on_the_cut_in_view_can_be_corrected(armed):
    trace(armed, box(armed, 50.0))

    edit(armed)

    assert armed.server.state.measurement_editing == 0
    assert armed.server.state.measurement_selected == 0


def test_a_region_off_the_cut_in_view_cannot_be(armed):
    """There is nothing to press on, so the refusal points at Recall instead."""
    trace(armed, box(armed, 50.0))
    armed.do("scroll_slice", view_name="ul", distance=5.0)

    edit(armed)

    assert armed.server.state.measurement_editing is None


def test_a_press_on_a_point_takes_hold_of_it(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)

    assert armed.logic.dispatch(
        "grab_measurement_point",
        view_name="ul",
        x=float(corner(armed, 50.0, 0)[0]),
        y=float(corner(armed, 50.0, 0)[1]),
    )


def test_a_press_away_from_every_point_takes_hold_of_nothing(armed):
    """Which is what leaves that drag to the window and level it always was."""
    trace(armed, box(armed, 50.0))
    edit(armed)

    x, y = corner(armed, 50.0, 0)
    assert not armed.logic.dispatch(
        "grab_measurement_point", view_name="ul", x=float(x) + 40.0, y=float(y)
    )


def test_a_press_takes_hold_of_nothing_while_no_region_is_being_corrected(armed):
    trace(armed, box(armed, 50.0))

    x, y = corner(armed, 50.0, 0)
    assert not armed.logic.dispatch(
        "grab_measurement_point", view_name="ul", x=float(x), y=float(y)
    )


def test_dragging_a_point_moves_that_one_and_no_other(armed):
    trace(armed, box(armed, 50.0))
    before = list(regions(armed)[0].points)
    edit(armed)

    start = corner(armed, 50.0, 0)
    drag(armed, start, (start[0] - 20.0, start[1] - 20.0))

    after = regions(armed)[0].points
    assert after[1:] == before[1:]
    assert after[0] != before[0]


def test_a_dragged_point_lands_where_the_cursor_let_go(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)

    start = corner(armed, 50.0, 0)
    end = (start[0] - 20.0, start[1] - 25.0)
    drag(armed, start, end)

    assert regions(armed)[0].points[0] == pytest.approx(
        cut_point(renderer(armed), *end), abs=1e-6
    )


def test_dragging_a_corner_measures_the_region_again(armed):
    """A square a hundred pixels across, with one corner pulled out to a kite."""
    scale = world_per_pixel(renderer(armed))
    trace(armed, box(armed, 50.0))
    edit(armed)

    start = corner(armed, 50.0, 0)
    drag(armed, start, (start[0] - 100.0, start[1]))

    # The square plus the triangle the pulled corner sweeps out.
    assert regions(armed)[0].area == pytest.approx(
        (100.0 * scale) ** 2 + 100.0 * 100.0 * scale**2 / 2.0, rel=1e-3
    )


def test_one_drag_is_one_edit_however_many_frames_it_took(armed):
    """Sixty writes a second is not sixty things to undo, or to read in a log."""
    trace(armed, box(armed, 50.0))
    edit(armed)

    written = []
    armed.server.state.change("measurement_data")(
        lambda **kwargs: written.append(kwargs["measurement_data"])
    )

    start = corner(armed, 50.0, 0)
    drag(armed, start, (start[0] - 30.0, start[1] - 30.0), moves=20)

    assert len(written) == 1


def test_the_picture_follows_the_cursor_before_the_drag_is_written_down(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)
    placed = list(regions(armed)[0].points)

    start = corner(armed, 50.0, 0)
    armed.logic.dispatch(
        "grab_measurement_point", view_name="ul", x=float(start[0]), y=float(start[1])
    )
    armed.do(
        "drag_measurement_point", view_name="ul", x=start[0] - 30.0, y=start[1] - 30.0
    )

    drawn = armed.logic.measurements._shape_of(0, regions(armed)[0])
    assert regions(armed)[0].points == placed, "not published until it is let go"
    assert drawn[0] != placed[0], "but drawn where the cursor is"


def test_reverting_puts_the_points_back_where_the_correction_found_them(armed):
    trace(armed, box(armed, 50.0))
    placed = list(regions(armed)[0].points)
    area = regions(armed)[0].area
    edit(armed)

    start = corner(armed, 50.0, 0)
    drag(armed, start, (start[0] - 30.0, start[1] - 30.0))
    armed.do("revert_measurement_edit")

    assert regions(armed)[0].points == placed
    assert regions(armed)[0].area == pytest.approx(area)
    assert armed.server.state.measurement_editing is None


def test_finishing_keeps_what_the_correction_did(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)

    start = corner(armed, 50.0, 0)
    drag(armed, start, (start[0] - 30.0, start[1] - 30.0))
    moved = list(regions(armed)[0].points)
    armed.do("finish_measurement_edit")

    assert regions(armed)[0].points == moved
    assert armed.server.state.measurement_editing is None


def test_a_click_on_the_contour_adds_a_point_to_it(armed):
    scale = world_per_pixel(renderer(armed))
    trace(armed, box(armed, 50.0))
    edit(armed)

    first, second = box(armed, 50.0)[0], box(armed, 50.0)[1]
    middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    armed.do("insert_measurement_point", view_name="ul", x=middle[0], y=middle[1])

    points = regions(armed)[0].points
    assert len(points) == 5
    assert points[1] == pytest.approx(cut_point(renderer(armed), *middle), abs=1e-6)
    # On the edge, so the region it encloses is the one it already was.
    assert regions(armed)[0].area == pytest.approx((100.0 * scale) ** 2, rel=1e-3)


def test_a_point_added_lands_on_the_contour_rather_than_under_the_cursor(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)

    first, second = box(armed, 50.0)[0], box(armed, 50.0)[1]
    middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    armed.do("insert_measurement_point", view_name="ul", x=middle[0], y=middle[1] - 4.0)

    added = regions(armed)[0].points[1]
    assert added == pytest.approx(cut_point(renderer(armed), *middle), abs=1e-6)


def test_a_click_far_from_the_contour_adds_nothing(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)

    x, y = box(armed, 50.0)[0]
    armed.do("insert_measurement_point", view_name="ul", x=x, y=y - 60.0)

    assert len(regions(armed)[0].points) == 4


def test_a_click_on_a_point_takes_it_away(armed):
    """Added and then taken away again, which is also the pair undoing itself."""
    scale = world_per_pixel(renderer(armed))
    trace(armed, box(armed, 50.0))
    edit(armed)

    first, second = box(armed, 50.0)[0], box(armed, 50.0)[1]
    middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    armed.do("insert_measurement_point", view_name="ul", x=middle[0], y=middle[1])
    assert len(regions(armed)[0].points) == 5

    armed.do("delete_measurement_point", view_name="ul", x=middle[0], y=middle[1])

    assert len(regions(armed)[0].points) == 4
    assert regions(armed)[0].area == pytest.approx((100.0 * scale) ** 2, rel=1e-3)


def test_the_last_three_points_of_a_region_cannot_be_taken_away(armed):
    """Fewer than three enclose nothing, and a region that encloses nothing
    is a region that should have been deleted rather than thinned."""
    trace(armed, box(armed, 50.0))
    edit(armed)

    for which in (0, 1):
        x, y = box(armed, 50.0)[which]
        armed.do("delete_measurement_point", view_name="ul", x=x, y=y)

    assert len(regions(armed)[0].points) == 3


def test_a_click_away_from_every_point_takes_none_of_them(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)

    x, y = box(armed, 50.0)[0]
    armed.do("delete_measurement_point", view_name="ul", x=x + 40.0, y=y)

    assert len(regions(armed)[0].points) == 4


def test_a_correction_marks_the_set_unsaved(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)
    armed.server.state.measurements_saved_at = "12:00:00"
    armed.server.state.measurements_stale = False

    start = corner(armed, 50.0, 0)
    drag(armed, start, (start[0] - 20.0, start[1]))

    assert armed.server.state.measurements_stale


def test_a_cut_moved_off_the_region_ends_the_correction(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)

    armed.do("scroll_slice", view_name="ul", distance=5.0)

    assert armed.server.state.measurement_editing is None


def test_tracing_is_suspended_while_a_region_is_being_corrected(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)

    armed.do("toggle_measuring")
    for x, y in box(armed, 20.0):
        armed.do("place_measurement_point", view_name="ul", x=float(x), y=float(y))

    assert armed.server.state.measurement_pending == 0
    assert len(regions(armed)) == 1


def test_deleting_the_region_being_corrected_ends_the_correction(armed):
    trace(armed, box(armed, 50.0))
    edit(armed)

    armed.do("delete_measurement", index=0)

    assert armed.server.state.measurement_editing is None


def test_the_key_corrects_whichever_region_is_highlighted(armed):
    trace(armed, box(armed, 50.0))
    trace(armed, box(armed, 30.0))
    armed.server.state.measurement_selected = 1

    armed.do("toggle_measurement_edit")
    assert armed.server.state.measurement_editing == 1

    armed.do("toggle_measurement_edit")
    assert armed.server.state.measurement_editing is None


def test_a_correction_survives_being_saved_and_read_back(armed, tmp_path):
    trace(armed, box(armed, 50.0))
    edit(armed)

    start = corner(armed, 50.0, 0)
    drag(armed, start, (start[0] - 30.0, start[1] - 20.0))
    armed.do("finish_measurement_edit")
    armed.do("save_measurements")

    written = next((armed.scene.measurements_directory / "vol").glob("*.toml"))
    reopened = MeasurementSet.from_file(written).measurements[0]
    assert np.array(reopened.points) == pytest.approx(
        np.array(regions(armed)[0].points)
    )
    assert reopened.area == pytest.approx(regions(armed)[0].area)


# --- recall -------------------------------------------------------------------


def pose_of(session: Session):
    """The pose the cuts stand at, as a recall would have to restore it."""
    return (
        list(session.server.state.mpr_origin),
        session.server.state.frame,
        [
            (step.get("axis"), step.get("angle"))
            for step in session.server.state.mpr_rotation_data["angles_list"]
        ],
    )


def test_recall_puts_the_cuts_back_where_the_region_was_measured(armed):
    trace(armed, box(armed, 50.0))
    at = pose_of(armed)

    armed.do("scroll_slice", view_name="ul", distance=8.0)
    armed.do("add_rotation", axis="X")
    armed.do("set_state", key="mpr_rotation_data.angles_list.0.angle", value=25.0)
    assert pose_of(armed) != at

    armed.do("recall_measurement", index=0)

    assert pose_of(armed) == at


def test_a_recalled_region_is_on_the_cut_again(armed):
    trace(armed, box(armed, 50.0))
    armed.do("scroll_slice", view_name="ul", distance=8.0)
    assert armed.server.state.measurement_on_plane == [False]

    armed.do("recall_measurement", index=0)

    assert armed.server.state.measurement_on_plane == [True]
    assert contour_props(armed) == 3


def test_recall_returns_to_the_frame_it_was_measured_at(tmp_path):
    for frame in (1, 2):
        write_volume(tmp_path / f"vol{frame}.nii.gz")
    session = session_on(
        tmp_path,
        volumes=[
            {
                "label": "vol",
                "directory": tmp_path,
                "file_paths": ["vol0.nii.gz", "vol1.nii.gz", "vol2.nii.gz"],
            }
        ],
    )
    session.ready()
    session.do("increment_frame")
    trace(session, box(session, 50.0))
    assert regions(session)[0].frame == 1

    session.do("decrement_frame")
    assert session.server.state.measurement_on_plane == [False]

    session.do("recall_measurement", index=0)

    assert session.server.state.frame == 1
    assert session.server.state.measurement_on_plane == [True]


def test_recall_leaves_the_quad_view_alone(armed):
    """It already shows all three cuts, so there is nothing to maximize."""
    armed.do("toggle_maximized", view="ul")
    armed.do("toggle_maximized", view="ul")
    assert armed.server.state.maximized_view == ""

    trace(armed, box(armed, 50.0))
    armed.do("recall_measurement", index=0)

    assert armed.server.state.maximized_view == ""


def test_recall_brings_the_pane_back_when_another_one_is_maximized(armed):
    trace(armed, box(armed, 50.0))
    armed.do("toggle_maximized", view="ll")

    armed.do("recall_measurement", index=0)

    assert armed.server.state.maximized_view == "ul"


def test_recall_abandons_a_half_traced_region(armed):
    """It was being traced on the cut the recall is about to move away from."""
    trace(armed, box(armed, 50.0))
    armed.do("place_measurement_point", view_name="ul", x=10.0, y=10.0)
    assert armed.server.state.measurement_pending == 1

    armed.do("recall_measurement", index=0)

    assert armed.server.state.measurement_pending == 0


# --- saving -------------------------------------------------------------------


def saved_files(session: Session):
    directory = session.scene.measurements_directory / "vol"
    return sorted(directory.glob("*.toml")) if directory.exists() else []


def test_saving_writes_a_measurement_file_under_the_volume(armed):
    trace(armed, box(armed, 50.0))

    armed.do("save_measurements")

    written = saved_files(armed)
    assert len(written) == 1
    assert written[0].suffix == ".toml"


def test_a_saved_file_holds_the_regions_and_what_they_came_to(armed):
    trace(armed, box(armed, 50.0))
    armed.do("rename_measurement", index=0, name="AVA")
    expected = regions(armed)[0].area

    armed.do("save_measurements")

    reopened = MeasurementSet.from_file(saved_files(armed)[0])
    assert [region.name for region in reopened.measurements] == ["AVA"]
    assert reopened.measurements[0].area == pytest.approx(expected)
    assert reopened.metadata.volume_label == "vol"


def test_a_saved_file_reopens_the_plane_it_was_measured_in(armed, tmp_path):
    """The whole point of the file: the region comes back on its own cut.

    Traced somewhere the app does not open on, so that arriving on its plane is
    something the recall did rather than something the default pose gave away.
    """
    armed.do("scroll_slice", view_name="ul", distance=3.0)
    armed.do("add_rotation", axis="Z")
    armed.do("set_state", key="mpr_rotation_data.angles_list.0.angle", value=15.0)
    trace(armed, box(armed, 50.0))
    armed.do("save_measurements")
    path = saved_files(armed)[0]

    again = tmp_path / "again"
    again.mkdir()
    fresh = session_on(again, measurement_file=path)
    fresh.ready()
    assert fresh.server.state.measurement_on_plane == [False], (
        "the file says where somebody looked, and does not go and look there"
    )

    fresh.do("recall_measurement", index=0)

    assert fresh.server.state.measurement_on_plane == [True]
    assert regions(fresh)[0].area == pytest.approx(regions(armed)[0].area)
    assert fresh.server.state.mpr_rotation_data["angles_list"][0]["angle"] == 15.0


def test_saving_says_when_it_happened(armed):
    trace(armed, box(armed, 50.0))
    assert armed.server.state.measurements_saved_at is None

    armed.do("save_measurements")

    assert armed.server.state.measurements_saved_at
    assert not armed.server.state.measurements_stale


def test_an_edit_after_a_save_says_the_file_is_behind(armed):
    trace(armed, box(armed, 50.0))
    armed.do("save_measurements")

    armed.do("rename_measurement", index=0, name="AVA")

    assert armed.server.state.measurements_stale


def test_saving_nothing_writes_nothing(armed):
    armed.do("save_measurements")

    assert saved_files(armed) == []
    assert armed.server.state.measurements_saved_at is None


# --- the tile grid ------------------------------------------------------------

# A tile is a place a cut is shown rather than a thing a cut belongs to, so a
# region traced on one is filed against its plane like any other. What is
# particular to the grid is that a tile has no pose of its own -- it is computed
# from the panel's settings -- so those have to be carried for a recall to work.


@pytest.fixture
def gridded(tmp_path) -> Session:
    """A session showing a parallel stack of cuts, a fixed distance apart."""
    session = session_on(
        tmp_path,
        view={"layout": "tile"},
        tile={"rows": 1, "cols": 3, "source": "spacing", "spacing": 2.0},
    )
    session.ready()
    return session


def tile_renderer(session: Session, index: int):
    return session.scene.tile_views.renderers[index]


def tile_props(session: Session, index: int) -> int:
    holder = tile_renderer(session, index).GetViewProps()
    drawn = [holder.GetItemAsObject(i) for i in range(holder.GetNumberOfItems())]
    return sum(1 for prop in drawn if not prop.GetUseBounds())


def tile_box(session: Session, index: int, half: float):
    """Four display positions about the centre of one tile of the grid."""
    renderer = tile_renderer(session, index)
    left, bottom = renderer.GetOrigin()
    width, height = renderer.GetSize()
    x, y = left + width / 2.0, bottom + height / 2.0
    return [
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
    ]


def test_the_grid_says_which_tile_a_point_landed_in(gridded):
    views = gridded.scene.tile_views

    for index in range(len(views)):
        x, y = tile_box(gridded, index, 0.0)[0]
        assert views.tile_at(x, y) == index


def test_a_point_outside_the_grid_lands_in_no_tile(gridded):
    views = gridded.scene.tile_views
    width, height = views.window.GetSize()

    assert views.tile_at(-5.0, -5.0) is None
    assert views.tile_at(width + 50.0, height + 50.0) is None


def test_a_region_is_traced_on_the_tile_it_was_clicked_in(gridded):
    trace(gridded, tile_box(gridded, 1, 30.0), view="tile")

    assert len(regions(gridded)) == 1
    assert regions(gridded)[0].view == "tile"
    assert regions(gridded)[0].tile.index == 1


def test_a_tile_region_measures_the_box_it_was_traced_as(gridded):
    """Both sides measured through the tile the clicks went into.

    Not one side squared: a tile is a third of the window wide and all of it
    tall, and a third of a pixel count is not a whole number, so a tile's
    pixels are a shade off square. The region is a rectangle in the cut, and
    the area it encloses is the rectangle's.
    """
    corners = tile_box(gridded, 1, 40.0)
    renderer = tile_renderer(gridded, 1)
    placed = [cut_point(renderer, x, y) for x, y in corners]
    expected = abs(placed[1][0] - placed[0][0]) * abs(placed[2][1] - placed[1][1])

    trace(gridded, corners, view="tile")

    assert regions(gridded)[0].area == pytest.approx(expected, rel=1e-9)


def test_a_tile_region_is_drawn_on_its_own_tile_alone(gridded):
    """The tiles are parallel cuts a fixed distance apart, so only one holds it."""
    trace(gridded, tile_box(gridded, 1, 30.0), view="tile")

    assert [tile_props(gridded, i) for i in range(3)] == [0, 3, 0]


def test_a_tile_region_carries_the_grid_it_was_traced_on(gridded):
    trace(gridded, tile_box(gridded, 1, 30.0), view="tile")
    grid = regions(gridded)[0].tile.grid

    assert (grid.rows, grid.cols) == (1, 3)
    assert grid.source is TileSource.SPACING
    assert grid.spacing == pytest.approx(2.0)


def test_a_point_on_another_tile_is_refused_mid_trace(gridded):
    """The tiles are different planes, so the points already down are not on it."""
    gridded.do("toggle_measuring")
    for x, y in tile_box(gridded, 1, 30.0)[:3]:
        gridded.do("place_measurement_point", view_name="tile", x=x, y=y)

    x, y = tile_box(gridded, 2, 30.0)[0]
    gridded.do("place_measurement_point", view_name="tile", x=x, y=y)

    assert gridded.server.state.measurement_pending == 3


def test_a_click_outside_every_tile_places_nothing(gridded):
    gridded.do("toggle_measuring")

    gridded.do("place_measurement_point", view_name="tile", x=-20.0, y=-20.0)

    assert gridded.server.state.measurement_pending == 0


def test_a_tile_region_moves_to_whichever_tile_comes_to_rest_on_its_plane(gridded):
    """A tile is a place a cut is shown, not a thing a cut belongs to.

    Reversing the stack takes the same cuts in the other order, so the plane
    traced on tile 0 is the one tile 2 now shows -- and the region is drawn
    there, because that is where its anatomy is.
    """
    trace(gridded, tile_box(gridded, 0, 30.0), view="tile")
    assert [tile_props(gridded, i) for i in range(3)] == [3, 0, 0]

    gridded.do("set_state", key="tile_reverse", value=True)

    assert [tile_props(gridded, i) for i in range(3)] == [0, 0, 3]


def test_a_tile_region_leaves_the_grid_when_the_stack_steps_past_it(gridded):
    """Traced on an outer tile, which is the half of the stack that moves.

    The stack is centred on the MPR origin, so the middle tile of an odd grid
    sits on that origin whatever the spacing is and never leaves its own plane.
    """
    trace(gridded, tile_box(gridded, 0, 30.0), view="tile")
    assert gridded.server.state.measurement_on_plane == [True]

    gridded.do("set_state", key="tile_spacing", value=7.0)

    assert gridded.server.state.measurement_on_plane == [False]
    assert [tile_props(gridded, i) for i in range(3)] == [0, 0, 0]


def test_the_middle_of_an_odd_stack_stays_put_however_it_is_spaced(gridded):
    """The other half of that, said outright: it sits on the shared origin."""
    trace(gridded, tile_box(gridded, 1, 30.0), view="tile")

    gridded.do("set_state", key="tile_spacing", value=7.0)

    assert gridded.server.state.measurement_on_plane == [True]
    assert [tile_props(gridded, i) for i in range(3)] == [0, 3, 0]


def test_a_tile_region_comes_back_when_the_stack_returns_to_it(gridded):
    trace(gridded, tile_box(gridded, 0, 30.0), view="tile")
    gridded.do("set_state", key="tile_spacing", value=7.0)

    gridded.do("set_state", key="tile_spacing", value=2.0)

    assert gridded.server.state.measurement_on_plane == [True]


def test_recall_puts_the_grid_back_before_it_looks_for_the_tile(gridded):
    """A tile has no pose of its own; the grid computes one from these."""
    trace(gridded, tile_box(gridded, 0, 30.0), view="tile")

    gridded.do("set_state", key="tile_spacing", value=9.0)
    gridded.do("set_state", key="tile_cols", value=2)
    assert gridded.server.state.measurement_on_plane == [False]

    gridded.do("recall_measurement", index=0)

    assert gridded.server.state.tile_spacing == pytest.approx(2.0)
    assert gridded.server.state.tile_cols == 3
    assert gridded.server.state.measurement_on_plane == [True]


def test_recall_brings_the_grid_back_on_screen(gridded):
    trace(gridded, tile_box(gridded, 1, 30.0), view="tile")
    gridded.do("toggle_maximized", view="ul")

    gridded.do("recall_measurement", index=0)

    assert gridded.server.state.maximized_view == "tile"
    assert gridded.server.state.measurement_on_plane == [True]


def test_deleting_a_tile_region_takes_its_outline_off_the_grid(gridded):
    trace(gridded, tile_box(gridded, 1, 30.0), view="tile")

    gridded.do("delete_measurement", index=0)

    assert [tile_props(gridded, i) for i in range(3)] == [0, 0, 0]


def test_a_tile_region_survives_a_saved_file(gridded, tmp_path):
    trace(gridded, tile_box(gridded, 1, 30.0), view="tile")
    expected = regions(gridded)[0].area
    gridded.do("save_measurements")

    reopened = MeasurementSet.from_file(saved_files(gridded)[0]).measurements[0]

    assert reopened.view == "tile"
    assert reopened.tile.index == 1
    assert reopened.tile.grid.spacing == pytest.approx(2.0)
    assert reopened.area == pytest.approx(expected)


def test_a_region_traced_on_a_pane_carries_no_grid(armed):
    assert regions(trace(armed, box(armed, 30.0))) and regions(armed)[0].tile is None


def test_a_region_reads_as_off_plane_while_no_layout_draws_its_cut(armed):
    """The dot says "on the cut now showing", so with nothing showing it is off.

    The pose has not moved -- coming back to the cuts lights it again without
    anything being recalled -- but a dot claiming the region is on a cut nobody
    is drawing would be answering a question it was not asked.
    """
    trace(armed, box(armed, 50.0))
    assert armed.server.state.measurement_on_plane == [True]

    armed.do("toggle_maximized", view="volume")
    assert armed.server.state.measurement_on_plane == [False]

    armed.do("toggle_maximized", view="volume")
    assert armed.server.state.measurement_on_plane == [True]


def test_a_region_traced_on_a_tile_can_be_corrected_there(gridded):
    """The grid is one window of viewports, so a press has to find its tile
    before it can find a point -- which is the one thing correcting on a tile
    does that correcting on a pane does not."""
    corners = tile_box(gridded, 1, 30.0)
    trace(gridded, corners, view="tile")
    before = list(regions(gridded)[0].points)

    gridded.do("edit_measurement", index=0)
    assert gridded.server.state.measurement_editing == 0

    start = corners[0]
    drag(gridded, start, (start[0] - 15.0, start[1] - 15.0), view="tile")

    after = regions(gridded)[0].points
    assert after[1:] == before[1:]
    assert after[0] == pytest.approx(
        cut_point(tile_renderer(gridded, 1), start[0] - 15.0, start[1] - 15.0), abs=1e-6
    )
