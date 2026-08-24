"""Test interface plane fitting and MPR alignment."""

import itk
import numpy as np
import pytest

from cardio.logic import ALIGN_STEP_NAME
from cardio.orientation import (
    AngleUnits,
    IndexOrder,
    axcode_transform_matrix,
    cumulative_rotation_matrix,
)
from cardio.segmentation import Segmentation, plane_basis, principal_axes
from tests.fakes import FakeApp, FakeScene

N = 24


def split_array(kind: str) -> np.ndarray:
    """A block divided into labels 1 and 2 by a surface of the given shape."""
    array = np.zeros((N,) * 3, dtype=np.uint8)
    kk, jj, ii = np.meshgrid(*[np.arange(N)] * 3, indexing="ij")
    block = (ii >= 4) & (ii < 20) & (jj >= 4) & (jj < 20) & (kk >= 4) & (kk < 20)
    match kind:
        case "plane_x":
            split = ii < 12
        case "oblique":
            split = (ii + jj) < 24
        case "oblique3d":
            split = (ii + jj + kk) < 36
        case "curved":
            split = ((ii - 12.0) ** 2 + (jj - 12.0) ** 2 + (kk - 12.0) ** 2) < 49.0
    array[block & split] = 1
    array[block & ~split] = 2
    return array


def make_segmentation(tmp_path, kind: str) -> Segmentation:
    itk.imwrite(itk.image_from_array(split_array(kind)), str(tmp_path / "s.nii.gz"))
    return Segmentation(label="s", directory=tmp_path, file_paths=["s.nii.gz"])


TILT_FRAMES = 4


def tilting_segmentation(tmp_path) -> Segmentation:
    """A series whose interface plane rotates 12 degrees per frame."""
    names = []
    kk, jj, ii = np.meshgrid(*[np.arange(N)] * 3, indexing="ij")
    block = (ii >= 4) & (ii < 20) & (jj >= 4) & (jj < 20) & (kk >= 4) & (kk < 20)
    for frame in range(TILT_FRAMES):
        theta = np.radians(12.0 * frame)
        nx, ny = np.cos(theta), np.sin(theta)
        split = (ii * nx + jj * ny) < 12.0 * (nx + ny)
        array = np.zeros((N,) * 3, dtype=np.uint8)
        array[block & split] = 1
        array[block & ~split] = 2
        name = f"tilt{frame}.nii.gz"
        itk.imwrite(itk.image_from_array(array), str(tmp_path / name))
        names.append(name)
    return Segmentation(label="s", directory=tmp_path, file_paths=names)


def make_logic(segmentation, index_order=IndexOrder.ITK) -> FakeApp:
    obj = FakeApp(
        FakeScene([segmentation], index_order=index_order),
        snap_seg_label="s",
        snap_mode="interface",
        snap_labels_a=[1],
        snap_labels_b=[2],
        snap_locked=False,
        snap_no_interface=False,
        interface_flatness=0.0,
        frame=0,
        mpr_origin=[0.0, 0.0, 0.0],
        mpr_rotation_data={"angles_list": []},
    )
    return obj


def axial_normal(logic: FakeApp) -> np.ndarray:
    """The axial slice normal the reslice pipeline will actually use."""
    sequence, angles = logic.rotations.visible_rotation_data()
    cumulative = cumulative_rotation_matrix(sequence, angles, AngleUnits.DEGREES)
    return (cumulative @ axcode_transform_matrix("LPS", "LAS"))[:, 2]


def test_principal_axes_of_a_flat_cloud():
    """A cloud spread in x and y has its normal along z."""
    rng = np.random.default_rng(0)
    points = np.column_stack(
        [rng.normal(0, 10, 500), rng.normal(0, 5, 500), np.zeros(500)]
    )
    centroid, axes, extents = principal_axes(points)
    assert centroid == pytest.approx([0, 0, 0], abs=1.5)
    assert abs(axes[:, 0] @ [1, 0, 0]) == pytest.approx(1.0, abs=1e-3)
    assert abs(axes[:, 2] @ [0, 0, 1]) == pytest.approx(1.0, abs=1e-3)
    assert extents[0] > extents[1] > extents[2]


def test_principal_axes_are_orthonormal():
    rng = np.random.default_rng(1)
    _, axes, _ = principal_axes(rng.normal(0, 1, (200, 3)))
    assert axes.T @ axes == pytest.approx(np.eye(3), abs=1e-9)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("plane_x", [1.0, 0.0, 0.0]),
        ("oblique", [1.0, 1.0, 0.0]),
        ("oblique3d", [1.0, 1.0, 1.0]),
    ],
)
def test_interface_normal_matches_geometry(tmp_path, kind, expected):
    seg = make_segmentation(tmp_path, kind)
    _, axes, _ = seg.interface_plane([1], [2], 0)
    expected = np.array(expected) / np.linalg.norm(expected)
    assert axes[:, 2] == pytest.approx(expected, abs=1e-2)


def test_interface_normal_points_from_a_to_b(tmp_path):
    """Swapping the groups reverses the normal."""
    seg = make_segmentation(tmp_path, "plane_x")
    _, forward, _ = seg.interface_plane([1], [2], 0)
    _, reverse, _ = seg.interface_plane([2], [1], 0)
    assert forward[:, 2] == pytest.approx(-reverse[:, 2], abs=1e-6)


def test_interface_basis_is_left_handed(tmp_path):
    """Matches the handedness of the LPS->LAS view transform, so R stays proper."""
    seg = make_segmentation(tmp_path, "oblique3d")
    _, axes, _ = seg.interface_plane([1], [2], 0)
    assert np.linalg.det(axes) == pytest.approx(-1.0, abs=1e-9)


def test_planar_interface_is_flat(tmp_path):
    seg = make_segmentation(tmp_path, "oblique3d")
    _, _, flatness = seg.interface_plane([1], [2], 0)
    assert flatness < 0.05


def test_curved_interface_is_not_flat(tmp_path):
    """A spherical interface has no dominant plane."""
    seg = make_segmentation(tmp_path, "curved")
    _, _, flatness = seg.interface_plane([1], [2], 0)
    assert flatness > 0.5


def test_interface_plane_requires_both_groups(tmp_path):
    seg = make_segmentation(tmp_path, "plane_x")
    assert seg.interface_plane([1], [], 0) is None
    assert seg.interface_plane([1], [7], 0) is None


def test_align_puts_axial_view_in_the_interface_plane(tmp_path):
    seg = make_segmentation(tmp_path, "oblique3d")
    logic = make_logic(seg)
    _, axes, _ = seg.interface_plane([1], [2], 0)

    logic.snap.align_to_interface()
    assert axial_normal(logic) == pytest.approx(axes[:, 2], abs=1e-6)


def test_align_also_centres_on_the_interface(tmp_path):
    seg = make_segmentation(tmp_path, "oblique3d")
    logic = make_logic(seg)
    logic.snap.align_to_interface()
    assert logic.server.state.mpr_origin == pytest.approx(
        seg.interface_centroid([1], [2], 0)
    )


def test_align_is_idempotent(tmp_path):
    """Re-aligning replaces the previous step instead of stacking."""
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.snap.align_to_interface()
    first = axial_normal(logic)
    logic.snap.align_to_interface()

    steps = logic.server.state.mpr_rotation_data["angles_list"]
    assert [s["name"] for s in steps] == [ALIGN_STEP_NAME]
    assert axial_normal(logic) == pytest.approx(first, abs=1e-9)


def user_rotation(axis: str, angle: float) -> dict:
    return {
        "axis": axis,
        "angle": angle,
        "visible": True,
        "name": "user",
        "name_editable": True,
        "deletable": True,
    }


def test_align_is_prepended_before_user_rotations(tmp_path):
    """Alignment is the base; the user's own rotations sit on top of it."""
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.server.state.mpr_rotation_data = {"angles_list": [user_rotation("X", 37.0)]}

    logic.snap.align_to_interface()
    steps = logic.server.state.mpr_rotation_data["angles_list"]
    assert [s["name"] for s in steps] == [ALIGN_STEP_NAME, "user"]


def test_in_plane_rotation_stays_in_the_plane(tmp_path):
    """A Z rotation on top of the alignment spins within the interface plane."""
    seg = make_segmentation(tmp_path, "oblique3d")
    logic = make_logic(seg)
    logic.snap.align_to_interface()
    aligned = axial_normal(logic)
    _, axes, _ = seg.interface_plane([1], [2], 0)

    steps = logic.server.state.mpr_rotation_data["angles_list"]
    logic.server.state.mpr_rotation_data = {
        "angles_list": steps + [user_rotation("Z", 30.0)]
    }
    assert axial_normal(logic) == pytest.approx(aligned, abs=1e-9)
    assert axial_normal(logic) == pytest.approx(axes[:, 2], abs=1e-6)


@pytest.mark.parametrize("axis", ["X", "Y"])
@pytest.mark.parametrize("angle", [30.0, 60.0])
def test_out_of_plane_rotation_tilts_by_that_angle(tmp_path, axis, angle):
    """X and Y rotations tilt off the plane by exactly the angle dialled in."""
    seg = make_segmentation(tmp_path, "oblique3d")
    logic = make_logic(seg)
    logic.snap.align_to_interface()
    _, axes, _ = seg.interface_plane([1], [2], 0)

    steps = logic.server.state.mpr_rotation_data["angles_list"]
    logic.server.state.mpr_rotation_data = {
        "angles_list": steps + [user_rotation(axis, angle)]
    }
    tilt = np.degrees(
        np.arccos(abs(np.clip(axial_normal(logic) @ axes[:, 2], -1.0, 1.0)))
    )
    assert tilt == pytest.approx(angle, abs=1e-6)


def test_align_round_trips_through_roma_convention(tmp_path):
    seg = make_segmentation(tmp_path, "oblique3d")
    logic = make_logic(seg, index_order=IndexOrder.ROMA)
    _, axes, _ = seg.interface_plane([1], [2], 0)

    logic.snap.align_to_interface()
    assert axial_normal(logic) == pytest.approx(axes[:, 2], abs=1e-6)


def test_align_records_flatness(tmp_path):
    logic = make_logic(make_segmentation(tmp_path, "curved"))
    logic.snap.align_to_interface()
    assert logic.server.state.interface_flatness > 0.5


def test_align_ignored_outside_interface_mode(tmp_path):
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.server.state.snap_mode = "label"
    logic.snap.align_to_interface()
    assert logic.server.state.mpr_rotation_data["angles_list"] == []


def test_align_ignored_without_both_groups(tmp_path):
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.server.state.snap_labels_b = []
    logic.snap.align_to_interface()
    assert logic.server.state.mpr_rotation_data["angles_list"] == []


def locked_logic(tmp_path, kind="oblique3d") -> FakeApp:
    """Aligned and orientation-locked, so the view follows the interface."""
    logic = make_logic(make_segmentation(tmp_path, kind))
    logic.server.state.snap_orientation_locked = True
    logic.snap.align_to_interface()
    return logic


def test_swap_exchanges_the_two_groups(tmp_path):
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.snap.swap_groups()
    assert logic.server.state.snap_labels_a == [2]
    assert logic.server.state.snap_labels_b == [1]


def test_swap_leaves_an_unlocked_view_alone(tmp_path):
    """Swapping is an edit to the selection, not a command to move the views."""
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.snap.align_to_interface()
    before = axial_normal(logic)

    logic.snap.swap_groups()
    assert axial_normal(logic) == pytest.approx(before, abs=1e-9)


def test_align_after_a_swap_flips_the_axial_normal(tmp_path):
    """The flip lands on the next align, viewing the interface from behind."""
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.snap.align_to_interface()
    before = axial_normal(logic)

    logic.snap.swap_groups()
    logic.snap.align_to_interface()
    assert axial_normal(logic) == pytest.approx(-before, abs=1e-6)


def test_swap_does_not_align_a_view_that_was_not_aligned(tmp_path):
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.snap.swap_groups()
    assert logic.server.state.mpr_rotation_data["angles_list"] == []


def test_swap_flips_a_locked_view_immediately(tmp_path):
    logic = locked_logic(tmp_path)
    before = axial_normal(logic)

    logic.snap.swap_groups()
    assert axial_normal(logic) == pytest.approx(-before, abs=1e-6)


def test_swap_leaves_the_origin_where_it_was(tmp_path):
    """The interface centroid is symmetric in A and B; only orientation flips."""
    logic = locked_logic(tmp_path)
    origin = list(logic.server.state.mpr_origin)

    logic.snap.swap_groups()
    assert logic.server.state.mpr_origin == pytest.approx(origin)


def test_swap_does_not_stack_alignment_steps(tmp_path):
    logic = locked_logic(tmp_path)
    logic.snap.swap_groups()
    logic.snap.swap_groups()

    steps = logic.server.state.mpr_rotation_data["angles_list"]
    assert [s["name"] for s in steps] == [ALIGN_STEP_NAME]


def test_swap_round_trips(tmp_path):
    logic = locked_logic(tmp_path)
    before = axial_normal(logic)

    logic.snap.swap_groups()
    logic.snap.swap_groups()
    assert logic.server.state.snap_labels_a == [1]
    assert axial_normal(logic) == pytest.approx(before, abs=1e-6)


def test_swap_keeps_user_rotations(tmp_path):
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.server.state.snap_orientation_locked = True
    logic.server.state.mpr_rotation_data = {"angles_list": [user_rotation("X", 37.0)]}
    logic.snap.align_to_interface()

    logic.snap.swap_groups()
    steps = logic.server.state.mpr_rotation_data["angles_list"]
    assert [s["name"] for s in steps] == [ALIGN_STEP_NAME, "user"]


def test_swap_ignored_outside_interface_mode(tmp_path):
    logic = make_logic(make_segmentation(tmp_path, "oblique3d"))
    logic.server.state.snap_mode = "label"

    logic.snap.swap_groups()
    assert logic.server.state.snap_labels_a == [1]
    assert logic.server.state.snap_labels_b == [2]


def test_orientation_lock_tracks_a_tilting_interface(tmp_path):
    """The view follows the plane frame to frame, unlike a one-shot align."""
    seg = tilting_segmentation(tmp_path)
    logic = make_logic(seg)
    logic.server.state.snap_orientation_locked = True

    for frame in range(TILT_FRAMES):
        logic.server.state.frame = frame
        logic.snap.apply_frame_lock(frame)
        _, axes, _ = seg.interface_plane([1], [2], frame)
        assert axial_normal(logic) == pytest.approx(axes[:, 2], abs=1e-6)


def test_one_shot_align_drifts_as_the_interface_tilts(tmp_path):
    """Contrast: aligning once leaves the view behind a moving plane."""
    seg = tilting_segmentation(tmp_path)
    logic = make_logic(seg)
    logic.snap.align_to_interface()
    fixed = axial_normal(logic)

    _, axes, _ = seg.interface_plane([1], [2], TILT_FRAMES - 1)
    assert fixed != pytest.approx(axes[:, 2], abs=1e-2)


def test_orientation_lock_is_independent_of_position_lock(tmp_path):
    """Orientation may be locked while the origin stays put."""
    seg = tilting_segmentation(tmp_path)
    logic = make_logic(seg)
    logic.server.state.snap_orientation_locked = True
    logic.server.state.snap_locked = False
    logic.server.state.mpr_origin = [9.0, 9.0, 9.0]

    logic.snap.apply_frame_lock(2)
    assert logic.server.state.mpr_origin == [9.0, 9.0, 9.0]
    _, axes, _ = seg.interface_plane([1], [2], 2)
    assert axial_normal(logic) == pytest.approx(axes[:, 2], abs=1e-6)


def test_position_lock_alone_leaves_orientation_alone(tmp_path):
    seg = tilting_segmentation(tmp_path)
    logic = make_logic(seg)
    logic.server.state.snap_locked = True
    logic.server.state.snap_orientation_locked = False

    logic.snap.apply_frame_lock(2)
    assert logic.server.state.mpr_rotation_data["angles_list"] == []
    assert logic.server.state.mpr_origin == pytest.approx(
        seg.interface_centroid([1], [2], 2)
    )


def test_both_locks_together(tmp_path):
    seg = tilting_segmentation(tmp_path)
    logic = make_logic(seg)
    logic.server.state.snap_locked = True
    logic.server.state.snap_orientation_locked = True

    for frame in range(TILT_FRAMES):
        logic.snap.apply_frame_lock(frame)
        _, axes, _ = seg.interface_plane([1], [2], frame)
        assert axial_normal(logic) == pytest.approx(axes[:, 2], abs=1e-6)
        assert logic.server.state.mpr_origin == pytest.approx(
            seg.interface_centroid([1], [2], frame)
        )


def test_orientation_lock_does_not_stack_steps(tmp_path):
    logic = make_logic(tilting_segmentation(tmp_path))
    logic.server.state.snap_orientation_locked = True
    for frame in range(TILT_FRAMES):
        logic.snap.apply_frame_lock(frame)
    steps = logic.server.state.mpr_rotation_data["angles_list"]
    assert [s["name"] for s in steps] == [ALIGN_STEP_NAME]


def test_orientation_lock_preserves_user_rotations_each_frame(tmp_path):
    """The user's offset is kept, not cancelled, as the plane is re-tracked."""
    seg = tilting_segmentation(tmp_path)
    logic = make_logic(seg)
    logic.server.state.mpr_rotation_data = {"angles_list": [user_rotation("Y", 20.0)]}
    logic.server.state.snap_orientation_locked = True

    for frame in range(TILT_FRAMES):
        logic.snap.apply_frame_lock(frame)
        _, axes, _ = seg.interface_plane([1], [2], frame)
        tilt = np.degrees(
            np.arccos(abs(np.clip(axial_normal(logic) @ axes[:, 2], -1.0, 1.0)))
        )
        # the plane is re-tracked every frame, with the user's 20 degrees kept
        assert tilt == pytest.approx(20.0, abs=1e-6)

    steps = logic.server.state.mpr_rotation_data["angles_list"]
    assert [s["name"] for s in steps] == [ALIGN_STEP_NAME, "user"]


def test_orientation_lock_ignored_in_label_mode(tmp_path):
    logic = make_logic(tilting_segmentation(tmp_path))
    logic.server.state.snap_orientation_locked = True
    logic.server.state.snap_mode = "label"
    logic.snap.apply_frame_lock(1)
    assert logic.server.state.mpr_rotation_data["angles_list"] == []


def test_interface_planes_are_memoized(tmp_path):
    logic = make_logic(tilting_segmentation(tmp_path))
    calls = []
    seg = logic.scene.segmentations[0]
    original = seg.interface_plane

    def counted(a, b, frame, anchor=None):
        calls.append(frame)
        return original(a, b, frame, anchor)

    object.__setattr__(seg, "interface_plane", counted)
    for _ in range(3):
        logic.snap._interface_plane(1)
    assert calls == [1]


def test_selection_change_drops_the_plane_cache(tmp_path):
    logic = make_logic(tilting_segmentation(tmp_path))
    logic.snap._interface_plane(0)
    assert logic.snap._lock_planes
    logic.snap._on_snap_selection_changed()
    assert logic.snap._lock_planes == {}


def precessing_normals(steps: int, tilt_degrees: float = 35.0) -> list[np.ndarray]:
    """Normals sweeping a cone, passing every anatomical reference direction."""
    tilt = np.radians(tilt_degrees)
    return [
        np.array([np.cos(tilt), np.sin(tilt) * np.cos(p), np.sin(tilt) * np.sin(p)])
        for p in np.linspace(0, 2 * np.pi, steps, endpoint=False)
    ]


def angle_between(a, b) -> float:
    return np.degrees(np.arccos(abs(np.clip(a @ b, -1.0, 1.0))))


def test_plane_basis_is_a_valid_view_basis():
    """Orthonormal, left-handed, and its third column is the normal."""
    anchor = None
    for normal in precessing_normals(60):
        basis = plane_basis(normal, anchor)
        anchor = anchor if anchor is not None else basis
        assert basis.T @ basis == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(basis) == pytest.approx(-1.0, abs=1e-12)
        assert basis[:, 2] == pytest.approx(normal, abs=1e-12)


def test_anchored_basis_is_continuous_under_refinement():
    """In-plane motion shrinks with the normal's, so there is no discontinuity."""
    for steps in (24, 48, 96):
        normals = precessing_normals(steps)
        anchor = plane_basis(normals[0])
        bases = [plane_basis(n, anchor) for n in normals]
        worst_normal = max(
            angle_between(normals[i], normals[i - 1]) for i in range(1, steps)
        )
        worst_in_plane = max(
            angle_between(bases[i][:, 0], bases[i - 1][:, 0]) for i in range(1, steps)
        )
        assert worst_in_plane < 3 * worst_normal


def test_anchor_removes_the_reference_singularity():
    """Sweeping the normal through the anatomical reference must not jump."""
    steps = 120
    normals = [
        np.array([np.cos(t), -np.sin(t), 0.0])
        for t in np.linspace(-np.pi / 2, np.pi / 2, steps)
    ]
    anchor = plane_basis(normals[0])
    bases = [plane_basis(n, anchor) for n in normals]
    step = angle_between(normals[1], normals[0])
    worst = max(
        angle_between(bases[i][:, 0], bases[i - 1][:, 0]) for i in range(1, steps)
    )
    assert worst == pytest.approx(step, abs=1e-6)


def circular_segmentation(tmp_path) -> Segmentation:
    """A near-circular (annulus-like) interface that tilts each frame.

    Its in-plane extents are nearly equal, so PCA cannot order the in-plane
    axes stably -- this is the case that made the view spin during playback.
    """
    names = []
    kk, jj, ii = np.meshgrid(*[np.arange(N)] * 3, indexing="ij")
    body = (((jj - 12.0) ** 2 + (kk - 12.0) ** 2) < 64.0) & (ii >= 4) & (ii < 20)
    for frame in range(TILT_FRAMES):
        theta = np.radians(9.0 * frame)
        nx, ny = np.cos(theta), np.sin(theta)
        split = (ii * nx + jj * ny) < 12.0 * (nx + ny)
        array = np.zeros((N,) * 3, dtype=np.uint8)
        array[body & split] = 1
        array[body & ~split] = 2
        name = f"round{frame}.nii.gz"
        itk.imwrite(itk.image_from_array(array), str(tmp_path / name))
        names.append(name)
    return Segmentation(label="s", directory=tmp_path, file_paths=names)


def test_in_plane_rotation_is_stable_for_a_circular_interface(tmp_path):
    """Regression: the in-plane axes must not jump between locked frames."""
    logic = make_logic(circular_segmentation(tmp_path))
    logic.server.state.snap_orientation_locked = True

    bases = [logic.snap._interface_plane(f)[1] for f in range(TILT_FRAMES)]
    for i in range(1, TILT_FRAMES):
        tilt = angle_between(bases[i][:, 2], bases[i - 1][:, 2])
        in_plane = angle_between(bases[i][:, 0], bases[i - 1][:, 0])
        assert in_plane < tilt + 5.0


WANDERING_FRAMES = 9


def wandering_group_segmentation(tmp_path) -> Segmentation:
    """A series where group B's centroid crosses to A's side of the interface.

    B is disconnected: a piece in front of A plus a growing piece behind it.
    The A/B interface never moves, but the whole-group centroid comparison
    reverses partway through.
    """
    names = []
    kk, jj, ii = np.meshgrid(*[np.arange(40)] * 3, indexing="ij")
    core = (jj >= 10) & (jj < 30) & (kk >= 10) & (kk < 30)
    for frame in range(WANDERING_FRAMES):
        array = np.zeros((40,) * 3, dtype=np.uint8)
        array[core & (ii >= 14) & (ii < 20)] = 1
        array[core & (ii >= 20) & (ii < 24)] = 2
        behind = 2 * frame
        if behind:
            array[core & (ii >= 14 - behind) & (ii < 14)] = 2
        name = f"wander{frame}.nii.gz"
        itk.imwrite(itk.image_from_array(array), str(tmp_path / name))
        names.append(name)
    return Segmentation(label="s", directory=tmp_path, file_paths=names)


def test_unanchored_normal_sign_can_flip(tmp_path):
    """Characterises why the sign is anchored: per-frame geometry is not stable."""
    seg = wandering_group_segmentation(tmp_path)
    normals = [
        seg.interface_plane([1], [2], f)[1][:, 2] for f in range(WANDERING_FRAMES)
    ]
    assert any(n @ normals[0] < 0 for n in normals)


def test_anchored_normal_sign_never_flips(tmp_path):
    """The same series, with the sign taken from the anchor, stays consistent."""
    seg = wandering_group_segmentation(tmp_path)
    anchor = seg.interface_plane([1], [2], 0)[1]
    for frame in range(WANDERING_FRAMES):
        plane = seg.interface_plane([1], [2], frame, anchor=anchor)
        assert plane[1][:, 2] @ anchor[:, 2] > 0


def test_anchored_sign_keeps_the_view_stable(tmp_path):
    """A flipped normal would reverse the in-plane axes; anchoring prevents it."""
    logic = make_logic(wandering_group_segmentation(tmp_path))
    logic.server.state.snap_orientation_locked = True

    bases = [logic.snap._interface_plane(f)[1] for f in range(WANDERING_FRAMES)]
    for basis in bases[1:]:
        assert basis[:, 0] @ bases[0][:, 0] > 0.99
        assert basis[:, 1] @ bases[0][:, 1] > 0.99


def test_anchor_frame_still_orients_from_a_to_b(tmp_path):
    """Without an anchor the geometric rule still decides, so swapping reverses."""
    seg = make_segmentation(tmp_path, "plane_x")
    forward = seg.interface_plane([1], [2], 0)[1][:, 2]
    reverse = seg.interface_plane([2], [1], 0)[1][:, 2]
    assert forward == pytest.approx(-reverse, abs=1e-6)


def test_anchored_sign_skips_the_centroid_passes(tmp_path):
    """The anchor path must not pay for the two group centroid computations."""
    seg = make_segmentation(tmp_path, "oblique3d")
    anchor = seg.interface_plane([1], [2], 0)[1]

    calls = []
    original = seg.label_centroid

    def counted(labels, frame=0):
        calls.append(tuple(labels))
        return original(labels, frame)

    object.__setattr__(seg, "label_centroid", counted)
    seg.interface_plane([1], [2], 0, anchor=anchor)
    assert calls == []


def make_logic_with_metadata(segmentation, index_order=IndexOrder.ITK) -> FakeApp:
    """As make_logic, but with the metadata sync_index_order rewrites."""
    logic = make_logic(segmentation, index_order=index_order)
    logic.server.state.mpr_rotation_data = {
        "angles_list": [],
        "metadata": {"index_order": str(index_order), "angle_units": "degrees"},
    }
    return logic


@pytest.mark.parametrize(
    "start,switch_to",
    [(IndexOrder.ITK, "roma"), (IndexOrder.ROMA, "itk")],
)
def test_index_order_switch_preserves_alignment(tmp_path, start, switch_to):
    """Switching convention must not move the view.

    The stored quaternion has to be converted along with the Euler steps; if it
    is left alone it keeps its numbers while changing meaning.
    """
    seg = make_segmentation(tmp_path, "oblique3d")
    logic = make_logic_with_metadata(seg, index_order=start)
    logic.snap.align_to_interface()
    before = axial_normal(logic)

    logic.rotations.sync_index_order(switch_to)
    assert axial_normal(logic) == pytest.approx(before, abs=1e-9)


def test_index_order_switch_preserves_alignment_with_user_rotation(tmp_path):
    """The same, with a Euler step present that converts by a different rule."""
    seg = make_segmentation(tmp_path, "oblique3d")
    logic = make_logic_with_metadata(seg)
    logic.snap.align_to_interface()
    steps = logic.server.state.mpr_rotation_data["angles_list"]
    logic.server.state.mpr_rotation_data = {
        "angles_list": steps + [user_rotation("Y", 25.0)],
        "metadata": {"index_order": "itk", "angle_units": "degrees"},
    }
    before = axial_normal(logic)

    logic.rotations.sync_index_order("roma")
    assert axial_normal(logic) == pytest.approx(before, abs=1e-9)


def test_index_order_switch_round_trips(tmp_path):
    seg = make_segmentation(tmp_path, "oblique3d")
    logic = make_logic_with_metadata(seg)
    logic.snap.align_to_interface()
    steps = logic.server.state.mpr_rotation_data["angles_list"]
    original = np.array([s["quaternion"] for s in steps])

    logic.rotations.sync_index_order("roma")
    logic.rotations.sync_index_order("itk")
    steps = logic.server.state.mpr_rotation_data["angles_list"]
    restored = np.array([s["quaternion"] for s in steps])
    assert restored == pytest.approx(original, abs=1e-12)
