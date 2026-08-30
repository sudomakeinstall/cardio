"""Test snap/lock centroid selection and per-frame locking in Logic."""

import itk
import numpy as np
import pytest

from cardio.logic import ALIGN_STEP_NAME
from cardio.orientation import IndexOrder
from cardio.segmentation import Segmentation
from tests.fakes import FakeApp, FakeScene

BLOCK_CENTER = 4.5
INTERFACE_X = 4.5


def shifting_label_array(offset: int) -> np.ndarray:
    """Two adjacent blocks whose shared plane is displaced by ``offset``."""
    array = np.zeros((16, 16, 16), dtype=np.uint8)
    array[2:8, 2:8, 2 + offset : 5 + offset] = 1
    array[2:8, 2:8, 5 + offset : 8 + offset] = 2
    return array


@pytest.fixture
def moving_segmentation(tmp_path) -> Segmentation:
    """A 3-frame segmentation whose interface moves one voxel per frame."""
    names = []
    for frame, offset in enumerate((0, 1, 2)):
        name = f"seg{frame}.nii.gz"
        itk.imwrite(
            itk.image_from_array(shifting_label_array(offset)), str(tmp_path / name)
        )
        names.append(name)
    return Segmentation(label="moving", directory=tmp_path, file_paths=names)


@pytest.fixture
def logic(moving_segmentation) -> FakeApp:
    """Real controllers wired to fakes, as Logic wires them."""
    obj = FakeApp(
        FakeScene([moving_segmentation]),
        snap_seg_label="moving",
        snap_mode="interface",
        snap_labels_a=[1],
        snap_labels_b=[2],
        snap_labels_c=[],
        snap_locked=False,
        snap_orientation_locked=False,
        snap_traverse=0,
        snap_no_interface=False,
        interface_flatness=0.0,
        frame=0,
        active_volume_label="",
        mpr_origin=[0.0, 0.0, 0.0],
        mpr_rotation_data={"angles_list": []},
    )
    return obj


def test_selection_requires_both_groups_in_interface_mode(logic):
    assert logic.snap._snap_selection() is not None
    logic.server.state.snap_labels_b = []
    assert logic.snap._snap_selection() is None


def test_selection_requires_labels_in_label_mode(logic):
    logic.server.state.snap_mode = "label"
    logic.server.state.snap_labels_a = []
    assert logic.snap._snap_selection() is None


def test_selection_requires_known_segmentation(logic):
    logic.server.state.snap_seg_label = "absent"
    assert logic.snap._snap_selection() is None


def test_snap_centroid_tracks_frame(logic):
    """The interface moves one voxel per frame, so the centroid follows it."""
    x0 = logic.snap._snap_centroid(0)[0]
    x1 = logic.snap._snap_centroid(1)[0]
    x2 = logic.snap._snap_centroid(2)[0]
    assert x0 == pytest.approx(INTERFACE_X, abs=1e-6)
    assert x1 == pytest.approx(INTERFACE_X + 1.0, abs=1e-6)
    assert x2 == pytest.approx(INTERFACE_X + 2.0, abs=1e-6)


def test_lock_disabled_is_a_noop(logic):
    logic.server.state.mpr_origin = [9.0, 9.0, 9.0]
    logic.snap.apply_frame_lock(2)
    assert logic.server.state.mpr_origin == [9.0, 9.0, 9.0]


def test_lock_recentres_on_each_frame(logic):
    logic.server.state.snap_locked = True
    for frame, expected_x in (
        (0, INTERFACE_X),
        (1, INTERFACE_X + 1),
        (2, INTERFACE_X + 2),
    ):
        logic.snap.apply_frame_lock(frame)
        assert logic.server.state.mpr_origin[0] == pytest.approx(expected_x, abs=1e-6)


def test_lock_matches_manual_snap(logic):
    """Locking to a frame gives the same origin as pressing Snap on it."""
    logic.server.state.frame = 2
    logic.snap.snap_to_centroid()
    manual = list(logic.server.state.mpr_origin)

    logic.server.state.mpr_origin = [0.0, 0.0, 0.0]
    logic.server.state.snap_locked = True
    logic.snap.apply_frame_lock(2)
    assert logic.server.state.mpr_origin == pytest.approx(manual)


def test_lock_reports_missing_interface(logic):
    logic.server.state.snap_locked = True
    logic.server.state.snap_labels_b = [3]
    logic.snap.apply_frame_lock(0)
    assert logic.server.state.snap_no_interface is True


def test_lock_ignores_an_empty_selection(logic):
    """Resetting the origin is a button now, not a mode that suspends the lock."""
    logic.server.state.snap_locked = True
    logic.server.state.snap_labels_a = []
    logic.server.state.mpr_origin = [9.0, 9.0, 9.0]
    logic.snap.apply_frame_lock(1)
    assert logic.server.state.mpr_origin == [9.0, 9.0, 9.0]


def test_snap_centroid_is_memoized(logic):
    calls = []
    seg = logic.scene.segmentations[0]
    original = seg.interface_centroid

    def counted(labels_a, labels_b, frame=0):
        calls.append(frame)
        return original(labels_a, labels_b, frame)

    object.__setattr__(seg, "interface_centroid", counted)
    for _ in range(3):
        logic.snap._snap_centroid(1)
    assert calls == [1]


def test_cache_is_dropped_when_selection_changes(logic):
    logic.snap._snap_centroid(0)
    assert logic.snap._lock_centroids
    logic.snap._on_snap_selection_changed()
    assert logic.snap._lock_centroids == {}


def test_enabling_lock_snaps_immediately(logic):
    logic.server.state.frame = 1
    logic.server.state.snap_locked = True
    logic.snap._on_snap_lock_changed(snap_locked=True)
    assert logic.server.state.mpr_origin[0] == pytest.approx(INTERFACE_X + 1, abs=1e-6)


def test_roma_index_order_swaps_axes(moving_segmentation):
    """ROMA scenes receive the centroid with X and Z exchanged."""
    obj = FakeApp(
        FakeScene([moving_segmentation], index_order=IndexOrder.ROMA),
        snap_seg_label="moving",
        snap_mode="interface",
        snap_labels_a=[1],
        snap_labels_b=[2],
        snap_locked=True,
        snap_no_interface=False,
        frame=0,
        mpr_origin=[0.0, 0.0, 0.0],
    )

    itk_centroid = obj.snap._snap_centroid(0)
    obj.snap.apply_frame_lock(0)
    assert obj.server.state.mpr_origin == pytest.approx(itk_centroid[::-1])


# --- Reset puts the panel back the way it started -----------------------------


def aligned_and_locked(logic) -> None:
    """Everything Reset has to undo, all switched on at once."""
    state = logic.server.state
    state.snap_mode = "traverse"
    state.snap_labels_a = [1]
    state.snap_labels_b = [2]
    state.snap_labels_c = [3]
    state.snap_traverse = 60
    state.snap_locked = True
    state.snap_orientation_locked = True
    state.snap_no_interface = True
    state.interface_flatness = 0.9


def step_names(logic) -> list[str]:
    return [s["name"] for s in logic.server.state.mpr_rotation_data["angles_list"]]


def test_reset_clears_the_groups_and_the_mode(logic):
    aligned_and_locked(logic)

    logic.snap.reset()

    state = logic.server.state
    assert state.snap_mode == "label"
    assert state.snap_labels_a == []
    assert state.snap_labels_b == []
    assert state.snap_labels_c == []
    assert state.snap_traverse == 0


def test_reset_releases_both_locks(logic):
    aligned_and_locked(logic)

    logic.snap.reset()

    assert logic.server.state.snap_locked is False
    assert logic.server.state.snap_orientation_locked is False


def test_reset_clears_the_warnings(logic):
    aligned_and_locked(logic)

    logic.snap.reset()

    assert logic.server.state.snap_no_interface is False
    assert logic.server.state.interface_flatness == 0.0


def test_reset_drops_the_alignment_step(logic):
    logic.server.state.snap_labels_a = [1]
    logic.server.state.snap_labels_b = [2]
    logic.snap.align_to_interface()
    assert ALIGN_STEP_NAME in step_names(logic)

    logic.snap.reset()

    assert step_names(logic) == []


def test_reset_keeps_the_rotations_the_user_added(logic):
    """Those are the rotations panel's to delete, not this button's."""
    logic.server.state.snap_labels_a = [1]
    logic.server.state.snap_labels_b = [2]
    logic.snap.align_to_interface()
    logic.rotations.add_mpr_rotation("Z")

    logic.snap.reset()

    steps = logic.server.state.mpr_rotation_data["angles_list"]
    assert [step["axis"] for step in steps] == ["Z"]
    assert ALIGN_STEP_NAME not in step_names(logic)


def test_reset_drops_the_memoized_planes(logic):
    logic.server.state.snap_labels_a = [1]
    logic.server.state.snap_labels_b = [2]
    logic.snap._snap_centroid(0)
    assert logic.snap._lock_centroids

    logic.snap.reset()

    assert logic.snap._lock_centroids == {}
    assert logic.snap._lock_planes == {}
