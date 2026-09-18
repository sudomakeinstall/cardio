"""A traced region as a document: what it holds, and what survives being written.

The area is ``planimetry``'s and is tested there.  What is tested here is the
part a study rests on -- that a region written to a file and read back is the
same region, on the same plane, with the same number against it, and that the
pose it carries is enough to put the cuts back.
"""

# System
import datetime as dt

# Third Party
import numpy as np
import pydantic as pc
import pytest

# Internal
import cardio.planimetry as planimetry
from cardio.measurement import Measurement, MeasurementSet, TileCut
from cardio.orientation import (
    AngleUnits,
    EulerAxis,
    IndexOrder,
    euler_angle_to_rotation_matrix,
)
from cardio.planimetry import ContourStyle
from cardio.rotation import RotationMetadata, RotationSequence, RotationStep
from cardio.snap import Snap, SnapMode
from cardio.tile import Tile, TileSource

SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]


def turn(axis: str, degrees: float) -> np.ndarray:
    return euler_angle_to_rotation_matrix(EulerAxis(axis), degrees, AngleUnits.DEGREES)


def pose() -> RotationSequence:
    return RotationSequence(
        metadata=RotationMetadata(
            index_order=IndexOrder.ITK, angle_units=AngleUnits.DEGREES
        ),
        angles_list=[RotationStep(axis="Z", angle=31.0, name="Mouse S-I")],
        mpr_origin=[3.0, -2.0, 11.0],
    )


def traced(**overrides) -> Measurement:
    fields = {
        "name": "AVA",
        "view": "ul",
        "frame": 7,
        "points": SQUARE,
        "cut": planimetry.cut_from([3.0, -2.0, 11.0], turn("Z", 31.0), "ul"),
        "pose": pose(),
    }
    return Measurement(**(fields | overrides))


# --- what a measurement holds ---------------------------------------------


def test_the_area_is_of_the_curve_the_region_is_drawn_as():
    assert traced().area == pytest.approx(16.0)
    assert traced(contour=ContourStyle.SPLINE).area > 16.0


def test_the_traced_points_are_placed_in_the_patient_by_the_cut():
    region = traced()
    assert region.lps()[0] == pytest.approx(region.cut.origin)
    assert len(region.lps()) == len(SQUARE)


def test_a_region_is_on_the_cut_it_was_traced_on():
    region = traced()
    assert region.on(region.cut, frame=7)


def test_a_region_is_not_on_the_same_plane_at_another_phase():
    region = traced()
    assert not region.on(region.cut, frame=8)


def test_a_region_stays_on_its_plane_when_the_cut_moves_within_it():
    region = traced()
    panned = planimetry.cut_from(
        np.array(region.cut.origin) + np.array(region.cut.right) * 40.0,
        turn("Z", 31.0),
        "ul",
    )
    assert region.on(panned, frame=7)


def test_a_region_leaves_its_plane_when_the_cut_is_scrolled_off_it():
    region = traced()
    scrolled = planimetry.cut_from(
        np.array(region.cut.origin) + region.cut.normal * 2.0, turn("Z", 31.0), "ul"
    )
    assert not region.on(scrolled, frame=7)


def test_a_region_leaves_its_plane_when_the_cut_is_turned_out_of_it():
    region = traced()
    turned = planimetry.cut_from(
        region.cut.origin, turn("Z", 31.0) @ turn("X", 5.0), "ul"
    )
    assert not region.on(turned, frame=7)


def test_a_cut_refuses_a_key_it_does_not_know():
    """A ``Cut`` is machine-written, so a misspelling there is a bug."""
    with pytest.raises(pc.ValidationError):
        planimetry.Cut(
            origin=(0.0, 0.0, 0.0), right=(1.0, 0.0, 0.0), top=(0.0, 1.0, 0.0)
        )


# --- the round trip -------------------------------------------------------


def test_a_set_survives_being_written_down():
    before = MeasurementSet(measurements=[traced()])
    after = MeasurementSet.from_toml(before.to_toml())

    assert after.measurements[0].points == before.measurements[0].points
    assert after.measurements[0].cut == before.measurements[0].cut
    assert after.measurements[0].area == pytest.approx(before.measurements[0].area)
    assert after.measurements[0].name == "AVA"
    assert after.measurements[0].frame == 7


def test_the_file_carries_the_area_it_measured():
    """A reader outside this app gets the number without redoing the shoelace."""
    assert "area = 16.0" in MeasurementSet(measurements=[traced()]).to_toml()


def test_the_area_it_carries_is_ignored_on_the_way_back_in():
    """Which is what lets the app read a file the app wrote.

    ``area`` is computed, so it cannot be assigned; a model that forbade extras
    would refuse its own output.  What comes back is measured again from the
    points, so a file edited to disagree with itself is read as the points.
    """
    text = MeasurementSet(measurements=[traced()]).to_toml()
    tampered = text.replace("area = 16.0", "area = 999.0")

    assert MeasurementSet.from_toml(tampered).measurements[0].area == pytest.approx(
        16.0
    )


def test_a_set_reaches_a_file_and_comes_back(tmp_path):
    before = MeasurementSet(measurements=[traced(), traced(name="AVA-2", frame=3)])
    path = before.to_file(tmp_path / "vol" / "regions.toml")

    after = MeasurementSet.from_file(path)

    assert path.exists()
    assert [m.name for m in after.measurements] == ["AVA", "AVA-2"]
    assert [m.frame for m in after.measurements] == [7, 3]


def test_an_empty_set_is_a_file_like_any_other(tmp_path):
    path = MeasurementSet().to_file(tmp_path / "none.toml")
    assert MeasurementSet.from_file(path).measurements == []


def test_the_file_says_what_its_numbers_are_in():
    text = MeasurementSet().to_toml()
    assert 'coordinate_system = "LPS"' in text
    assert 'length_units = "mm"' in text


def test_a_set_carries_a_stamp_of_when_it_was_taken():
    stamped = MeasurementSet().metadata.timestamp
    assert dt.datetime.fromisoformat(stamped).tzinfo is not None


# --- the pose it carries --------------------------------------------------


def test_the_pose_is_the_whole_sequence_the_cuts_were_at():
    region = MeasurementSet.from_toml(
        MeasurementSet(measurements=[traced()]).to_toml()
    ).measurements[0]

    assert region.pose.mpr_origin == [3.0, -2.0, 11.0]
    assert [step.axis for step in region.pose.angles_list] == ["Z"]
    assert region.pose.angles_list[0].angle == pytest.approx(31.0)
    assert region.pose.metadata.angle_units is AngleUnits.DEGREES


def test_the_pose_carries_its_own_convention_so_the_file_needs_none():
    """A measurement file is LPS throughout, whatever order a session is in.

    Nobody types a cut, so there is no user convention for it to be in -- and
    the poses nested inside say for themselves what order they were written in.
    """
    roma = traced(pose=pose().with_index_order(IndexOrder.ROMA))
    after = MeasurementSet.from_toml(
        MeasurementSet(measurements=[roma]).to_toml()
    ).measurements[0]

    assert after.pose.metadata.index_order is IndexOrder.ROMA
    assert after.cut == roma.cut
    assert after.points == roma.points


def test_re_expressing_a_pose_leaves_the_region_it_measured_alone():
    """The points are the cut's own millimetres, which no index order touches."""
    region = traced()
    moved = region.model_copy(
        update={"pose": region.pose.with_index_order(IndexOrder.ROMA)}
    )

    assert moved.points == region.points
    assert moved.cut == region.cut
    assert moved.area == pytest.approx(region.area)


# --- a region traced on a tile --------------------------------------------


def tile_traced() -> Measurement:
    return traced(
        view="tile",
        tile=TileCut(
            index=4,
            grid=Tile(rows=2, cols=3, source=TileSource.SPACING, spacing=7.5),
            snap=Snap(mode=SnapMode.INTERFACE, labels_a=[1], labels_b=[2]),
        ),
    )


def test_a_tile_region_carries_the_grid_it_was_traced_on():
    after = MeasurementSet.from_toml(
        MeasurementSet(measurements=[tile_traced()]).to_toml()
    ).measurements[0]

    assert after.tile.index == 4
    assert after.tile.grid.rows == 2
    assert after.tile.grid.cols == 3
    assert after.tile.grid.spacing == pytest.approx(7.5)
    assert after.tile.grid.source is TileSource.SPACING
    assert after.tile.snap.labels_a == [1]


def test_a_region_traced_anywhere_else_carries_no_grid():
    assert traced().tile is None
    assert "tile" not in MeasurementSet(measurements=[traced()]).to_toml()
