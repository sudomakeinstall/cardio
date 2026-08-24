"""Test traversal between the two interface planes of three label groups."""

import itk
import numpy as np
import pytest

from cardio.logic import ALIGN_STEP_NAME
from cardio.orientation import (
    AngleUnits,
    axcode_transform_matrix,
    cumulative_rotation_matrix,
    minimal_rotation,
    slerp_rotation_matrices,
)
from cardio.segmentation import Segmentation, interpolate_planes, plane_basis
from tests.fakes import FakeApp, FakeScene


def tilted_plane(degrees: float) -> np.ndarray:
    """A left-handed plane basis whose normal is tilted from z toward x."""
    theta = np.radians(degrees)
    return plane_basis(np.array([np.sin(theta), 0.0, np.cos(theta)]))


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


# Interpolation math


def test_slerp_reaches_both_endpoints():
    start, end = np.eye(3), tilted_plane(0) @ tilted_plane(40).T
    assert slerp_rotation_matrices(start, end, 0.0) == pytest.approx(start, abs=1e-12)
    assert slerp_rotation_matrices(start, end, 1.0) == pytest.approx(end, abs=1e-12)


def test_interpolate_planes_reproduces_its_endpoints():
    start = (np.array([0.0, 0.0, 0.0]), tilted_plane(0), 0.01)
    end = (np.array([3.0, 4.0, 0.0]), tilted_plane(30), 0.02)

    for fraction, expected in ((0.0, start), (1.0, end)):
        centroid, axes, _ = interpolate_planes(start, end, fraction)
        assert centroid == pytest.approx(expected[0], abs=1e-12)
        assert axes == pytest.approx(expected[1], abs=1e-12)


def test_interpolate_planes_halves_the_tilt():
    start = (np.zeros(3), tilted_plane(0), 0.0)
    end = (np.zeros(3), tilted_plane(30), 0.0)
    _, axes, _ = interpolate_planes(start, end, 0.5)
    assert angle_between(axes[:, 2], start[1][:, 2]) == pytest.approx(15.0, abs=1e-6)
    assert angle_between(axes[:, 2], end[1][:, 2]) == pytest.approx(15.0, abs=1e-6)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_interpolated_basis_stays_left_handed(fraction):
    """_apply_alignment composes with an improper transform, so this must hold."""
    start = (np.zeros(3), tilted_plane(-20), 0.0)
    end = (np.zeros(3), tilted_plane(50), 0.0)
    _, axes, _ = interpolate_planes(start, end, fraction)
    assert axes.T @ axes == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(axes) == pytest.approx(-1.0, abs=1e-9)


def test_interpolated_centroid_is_a_straight_blend():
    start = (np.array([0.0, 0.0, 0.0]), tilted_plane(0), 0.0)
    end = (np.array([10.0, -4.0, 2.0]), tilted_plane(30), 0.0)
    centroid, _, _ = interpolate_planes(start, end, 0.25)
    assert centroid == pytest.approx([2.5, -1.0, 0.5], abs=1e-12)


def test_flatness_is_the_worse_of_the_two():
    start = (np.zeros(3), tilted_plane(0), 0.4)
    end = (np.zeros(3), tilted_plane(10), 0.02)
    assert interpolate_planes(start, end, 0.5)[2] == pytest.approx(0.4)


@pytest.mark.parametrize("fraction,clamped", [(-0.5, 0.0), (1.5, 1.0)])
def test_fraction_is_clamped(fraction, clamped):
    start = (np.array([0.0, 0.0, 0.0]), tilted_plane(0), 0.0)
    end = (np.array([6.0, 0.0, 0.0]), tilted_plane(30), 0.0)
    centroid, axes, _ = interpolate_planes(start, end, fraction)
    expected_centroid, expected_axes, _ = interpolate_planes(start, end, clamped)
    assert centroid == pytest.approx(expected_centroid, abs=1e-12)
    assert axes == pytest.approx(expected_axes, abs=1e-12)


# Phantoms

M = 32


def stacked_array(tilt_degrees: float, offset: int = 0) -> np.ndarray:
    """Three slabs along x: 1 | 2 | 3.

    The 1|2 interface is normal to x. The 2|3 interface is tilted by
    ``tilt_degrees`` toward y, so travelling between them is a pure tilt.
    ``offset`` displaces the whole stack along x, ends included, so the two
    interface patches translate rigidly rather than being clipped differently.
    """
    array = np.zeros((M,) * 3, dtype=np.uint8)
    kk, jj, ii = np.meshgrid(*[np.arange(M)] * 3, indexing="ij")
    body = (
        (((jj - 16.0) ** 2 + (kk - 16.0) ** 2) < 49.0)
        & ((ii - offset) >= 6)
        & ((ii - offset) < 26)
    )

    theta = np.radians(tilt_degrees)
    nx, ny = np.cos(theta), np.sin(theta)
    far = (ii - offset) * nx + (jj - 16.0) * ny >= 20.0 * nx

    array[body & ((ii - offset) < 12)] = 1
    array[body & ((ii - offset) >= 12) & ~far] = 2
    array[body & far] = 3
    return array


def stacked_segmentation(tmp_path, tilt_degrees: float = 30.0) -> Segmentation:
    name = "stack.nii.gz"
    itk.imwrite(itk.image_from_array(stacked_array(tilt_degrees)), str(tmp_path / name))
    return Segmentation(label="s", directory=tmp_path, file_paths=[name])


MOVING_FRAMES = 3


def moving_stack_segmentation(tmp_path) -> Segmentation:
    """The same stack, displaced two voxels along x per frame."""
    names = []
    for frame in range(MOVING_FRAMES):
        name = f"move{frame}.nii.gz"
        itk.imwrite(
            itk.image_from_array(stacked_array(30.0, offset=2 * frame)),
            str(tmp_path / name),
        )
        names.append(name)
    return Segmentation(label="s", directory=tmp_path, file_paths=names)


def make_logic(segmentation, **overrides) -> FakeApp:
    state = dict(
        snap_seg_label="s",
        snap_mode="traverse",
        snap_labels_a=[1],
        snap_labels_b=[2],
        snap_labels_c=[3],
        snap_traverse=0,
        snap_locked=False,
        snap_orientation_locked=False,
        snap_no_interface=False,
        interface_flatness=0.0,
        frame=0,
        mpr_origin=[0.0, 0.0, 0.0],
        mpr_rotation_data={"angles_list": []},
    )
    state.update(overrides)
    return FakeApp(FakeScene([segmentation]), **state)


def axial_normal(logic: FakeApp) -> np.ndarray:
    """The axial slice normal the reslice pipeline will actually use."""
    sequence, angles = logic.rotations.visible_rotation_data()
    cumulative = cumulative_rotation_matrix(sequence, angles, AngleUnits.DEGREES)
    return (cumulative @ axcode_transform_matrix("LPS", "LAS"))[:, 2]


def align_at(logic: FakeApp, traverse: int):
    logic.server.state.snap_traverse = traverse
    logic.snap.align_to_interface()


# Selection


def test_traverse_requires_all_three_groups(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path))
    assert logic.snap._snap_selection() is not None
    logic.server.state.snap_labels_c = []
    assert logic.snap._snap_selection() is None


def test_interface_mode_ignores_group_c(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path), snap_mode="interface")
    logic.server.state.snap_labels_c = []
    assert logic.snap._snap_selection() is not None


def test_incomplete_traverse_does_not_move_the_views(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path), snap_labels_c=[])
    logic.snap.align_to_interface()
    logic.snap.snap_to_centroid()
    assert logic.server.state.mpr_origin == [0.0, 0.0, 0.0]
    assert logic.server.state.mpr_rotation_data["angles_list"] == []


# Travelling


def test_endpoints_match_the_two_interfaces(tmp_path):
    """0% aligns to A|B and 100% to B|C, as aligning to each directly would."""
    seg = stacked_segmentation(tmp_path)
    logic = make_logic(seg)

    align_at(logic, 0)
    start_normal = axial_normal(logic)
    align_at(logic, 100)
    end_normal = axial_normal(logic)

    assert start_normal == pytest.approx(
        logic.snap._interface_plane(0)[1][:, 2], abs=1e-9
    )
    assert angle_between(start_normal, [1.0, 0.0, 0.0]) < 2.0
    assert angle_between(start_normal, end_normal) == pytest.approx(30.0, abs=2.0)


def test_midpoint_is_halfway_between_the_interfaces(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path))

    align_at(logic, 0)
    start_normal = axial_normal(logic)
    align_at(logic, 100)
    end_normal = axial_normal(logic)
    align_at(logic, 50)
    middle_normal = axial_normal(logic)

    tilt = angle_between(start_normal, end_normal)
    assert angle_between(middle_normal, start_normal) == pytest.approx(
        tilt / 2, abs=1e-6
    )
    assert angle_between(middle_normal, end_normal) == pytest.approx(tilt / 2, abs=1e-6)


def test_origin_walks_between_the_two_interface_centroids(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path))

    align_at(logic, 0)
    start = np.array(logic.server.state.mpr_origin)
    align_at(logic, 100)
    end = np.array(logic.server.state.mpr_origin)
    align_at(logic, 25)
    quarter = np.array(logic.server.state.mpr_origin)

    assert end[0] > start[0] + 4.0
    assert quarter == pytest.approx(0.75 * start + 0.25 * end, abs=1e-9)


def test_traversal_is_continuous(tmp_path):
    """No jump anywhere along the slider, in either the origin or the normal."""
    logic = make_logic(stacked_segmentation(tmp_path))

    poses = []
    for traverse in range(0, 101, 10):
        align_at(logic, traverse)
        poses.append((np.array(logic.server.state.mpr_origin), axial_normal(logic)))

    steps = [
        (
            float(np.linalg.norm(b[0] - a[0])),
            angle_between(a[1], b[1]),
        )
        for a, b in zip(poses, poses[1:])
    ]
    for distance, turn in steps:
        assert distance == pytest.approx(steps[0][0], abs=1e-6)
        assert turn == pytest.approx(steps[0][1], abs=1e-6)


def test_alignment_replaces_rather_than_stacks_steps(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path))
    for traverse in (0, 40, 80):
        align_at(logic, traverse)
    steps = logic.server.state.mpr_rotation_data["angles_list"]
    assert [step["name"] for step in steps] == [ALIGN_STEP_NAME]


def test_dragging_the_slider_realigns(tmp_path):
    """The listener follows the slider, so the user need not press Align again."""
    logic = make_logic(stacked_segmentation(tmp_path))
    align_at(logic, 0)
    start = np.array(logic.server.state.mpr_origin)

    logic.server.state.snap_traverse = 100
    logic.snap._on_snap_traverse_changed()
    assert np.array(logic.server.state.mpr_origin)[0] > start[0] + 4.0


def test_slider_does_nothing_outside_traverse_mode(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path), snap_mode="interface")
    logic.server.state.snap_traverse = 100
    logic.snap._on_snap_traverse_changed()
    assert logic.server.state.mpr_rotation_data["angles_list"] == []


# Anchoring


def test_the_two_planes_differ_by_the_minimal_rotation(tmp_path):
    """B|C is anchored on A|B, so travelling tilts the view without spinning it."""
    logic = make_logic(stacked_segmentation(tmp_path))
    start = logic.snap._interface_plane(0)[1]
    end = logic.snap._interface_plane(0, 1)[1]

    carried = minimal_rotation(start[:, 2], end[:, 2]) @ start
    assert end == pytest.approx(carried, abs=1e-6)


def test_both_normals_point_downstream(tmp_path):
    """A->B and B->C are the same direction of travel, so the normals agree."""
    logic = make_logic(stacked_segmentation(tmp_path))
    start = logic.snap._interface_plane(0)[1][:, 2]
    end = logic.snap._interface_plane(0, 1)[1][:, 2]
    assert start @ end > 0.5


def test_planes_are_memoized_per_interface(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path))
    seg = logic.scene.segmentations[0]
    calls = []
    original = seg.interface_plane

    def counted(labels_a, labels_b, frame=0, anchor=None):
        calls.append((tuple(labels_a), tuple(labels_b), frame))
        return original(labels_a, labels_b, frame, anchor)

    object.__setattr__(seg, "interface_plane", counted)
    for _ in range(3):
        logic.snap._traverse_plane(0)
    assert calls == [((1,), (2,), 0), ((2,), (3,), 0)]


def test_selection_change_drops_both_planes(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path))
    logic.snap._traverse_plane(0)
    assert set(logic.snap._lock_planes) == {(0, 0), (0, 1)}
    logic.snap._on_snap_selection_changed()
    assert logic.snap._lock_planes == {}


def test_slider_reuses_the_cached_planes(tmp_path):
    """Dragging pays for the interpolation only, not two more plane fits."""
    logic = make_logic(stacked_segmentation(tmp_path))
    align_at(logic, 0)
    cached = dict(logic.snap._lock_planes)
    align_at(logic, 70)
    assert logic.snap._lock_planes == cached


# Swapping


def test_swap_reverses_the_direction_of_travel(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path))
    align_at(logic, 0)
    start = np.array(logic.server.state.mpr_origin)
    align_at(logic, 100)
    end = np.array(logic.server.state.mpr_origin)

    align_at(logic, 0)
    logic.snap.swap_groups()
    assert logic.server.state.snap_labels_a == [3]
    assert logic.server.state.snap_labels_b == [2]
    assert logic.server.state.snap_labels_c == [1]
    assert logic.server.state.snap_traverse == 100

    logic.snap.align_to_interface()
    assert np.array(logic.server.state.mpr_origin) == pytest.approx(start, abs=1e-6)

    align_at(logic, 0)
    assert np.array(logic.server.state.mpr_origin) == pytest.approx(end, abs=1e-6)


def test_swap_reverses_the_normal(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path))
    align_at(logic, 50)
    before = axial_normal(logic)

    logic.snap.swap_groups()
    logic.snap.align_to_interface()
    assert axial_normal(logic) == pytest.approx(-before, abs=1e-6)


# Locking


def test_locked_traverse_tracks_the_frame(tmp_path):
    """Both interfaces are refitted per frame and re-interpolated at the slider."""
    logic = make_logic(
        moving_stack_segmentation(tmp_path),
        snap_locked=True,
        snap_orientation_locked=True,
        snap_traverse=50,
    )

    origins = []
    for frame in range(MOVING_FRAMES):
        logic.server.state.frame = frame
        logic.snap.apply_frame_lock(frame)
        origins.append(np.array(logic.server.state.mpr_origin))

    assert not logic.server.state.snap_no_interface
    for earlier, later in zip(origins, origins[1:]):
        assert later[0] - earlier[0] == pytest.approx(2.0, abs=0.3)


def test_locked_traverse_honours_a_slider_move(tmp_path):
    logic = make_logic(
        moving_stack_segmentation(tmp_path),
        snap_locked=True,
        snap_orientation_locked=True,
    )
    logic.snap.apply_frame_lock(0)
    start = np.array(logic.server.state.mpr_origin)

    logic.server.state.snap_traverse = 100
    logic.snap.apply_frame_lock(0)
    assert np.array(logic.server.state.mpr_origin)[0] > start[0] + 4.0


def test_orientation_lock_applies_in_traverse_mode(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path), snap_orientation_locked=True)
    logic.snap.apply_frame_lock(0)
    steps = logic.server.state.mpr_rotation_data["angles_list"]
    assert [step["name"] for step in steps] == [ALIGN_STEP_NAME]


def test_flatness_reports_the_worse_interface(tmp_path):
    logic = make_logic(stacked_segmentation(tmp_path))
    flatnesses = [logic.snap._interface_plane(0, pair)[2] for pair in (0, 1)]
    align_at(logic, 50)
    assert logic.server.state.interface_flatness == pytest.approx(max(flatnesses))
