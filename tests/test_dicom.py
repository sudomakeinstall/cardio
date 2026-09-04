"""Reading a cine DICOM series back into the frames it was written from.

The phantom writes a known value into every (slice, phase) cell, so each test
can state which image should have landed where rather than compare pictures.
"""

# System

# Third Party
import itk
import numpy as np
import pydicom as pd
import pytest

# Internal
from cardio import dicom
from cardio.mesh import Mesh
from cardio.scene import Scene
from cardio.volume import Volume
from tests.phantoms import cine_voxel_value, write_cine_series


def oblique_direction(yaw_degrees: float = 35.0, pitch_degrees: float = 20.0):
    yaw, pitch = np.radians(yaw_degrees), np.radians(pitch_degrees)
    about_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    about_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    return about_z @ about_x


def slice_values(frame) -> list[int]:
    """The value carried by each slice of a frame, ordered along the stack."""
    array = itk.array_from_image(frame)
    return [int(array[z, 0, 0]) for z in range(array.shape[0])]


# --- shape of the result ------------------------------------------------------


def test_reads_one_frame_per_phase(tmp_path):
    write_cine_series(tmp_path, slices=4, phases=3)

    frames = dicom.read_series(tmp_path)

    assert len(frames) == 3
    for frame in frames:
        assert frame.GetImageDimension() == 3
        assert list(frame.GetLargestPossibleRegion().GetSize()) == [10, 12, 4]


def test_every_image_lands_in_its_own_cell(tmp_path):
    write_cine_series(tmp_path, slices=4, phases=3)

    frames = dicom.read_series(tmp_path)

    for phase, frame in enumerate(frames):
        assert slice_values(frame) == [cine_voxel_value(z, phase) for z in range(4)]


def test_geometry_comes_from_the_headers(tmp_path):
    write_cine_series(tmp_path, slices=4, phases=2, origin=(-7.0, 5.0, -3.0))

    frame = dicom.read_series(tmp_path)[0]

    # PixelSpacing is (row, column); ITK spacing is (column, row, slice).
    np.testing.assert_allclose(list(frame.GetSpacing()), [2.0, 1.5, 4.0])
    np.testing.assert_allclose(list(frame.GetOrigin()), [-7.0, 5.0, -3.0])


# --- grouping does not depend on the numbering --------------------------------


@pytest.mark.parametrize("order", ["slice_major", "phase_major", "shuffled"])
def test_grouping_survives_any_instance_order(tmp_path, order):
    """InstanceNumber may run either way round, or not usefully at all.

    Slice identity is taken from position along the acquisition normal, so the
    frames come back the same however the files happen to be numbered.
    """
    write_cine_series(tmp_path, slices=4, phases=3, instance_order=order)

    frames = dicom.read_series(tmp_path)

    assert len(frames) == 3
    for phase, frame in enumerate(frames):
        assert slice_values(frame) == [cine_voxel_value(z, phase) for z in range(4)]


def test_phases_ordered_by_trigger_time_not_instance_number(tmp_path):
    write_cine_series(tmp_path, slices=2, phases=4, instance_order="shuffled")

    frames = dicom.read_series(tmp_path)

    assert [slice_values(frame)[0] for frame in frames] == [
        cine_voxel_value(0, phase) for phase in range(4)
    ]


def test_falls_back_to_instance_number_without_timing(tmp_path):
    write_cine_series(tmp_path, slices=2, phases=3, with_trigger_time=False)

    frames = dicom.read_series(tmp_path)

    assert len(frames) == 3
    for phase, frame in enumerate(frames):
        assert slice_values(frame) == [cine_voxel_value(z, phase) for z in range(2)]


# --- oblique acquisition ------------------------------------------------------


def test_oblique_series_keeps_its_direction(tmp_path):
    direction = oblique_direction()
    write_cine_series(tmp_path, slices=4, phases=2, direction=direction)

    frames = dicom.read_series(tmp_path)

    for frame in frames:
        np.testing.assert_allclose(
            itk.array_from_matrix(frame.GetDirection()), direction, atol=1e-12
        )


def test_oblique_series_stacks_in_the_right_order(tmp_path):
    """The stack runs along the normal, which for an oblique series is not an axis."""
    write_cine_series(tmp_path, slices=4, phases=2, direction=oblique_direction())

    frames = dicom.read_series(tmp_path)

    for phase, frame in enumerate(frames):
        assert slice_values(frame) == [cine_voxel_value(z, phase) for z in range(4)]


# --- degenerate shapes --------------------------------------------------------


def test_single_phase_series_is_one_volume(tmp_path):
    write_cine_series(tmp_path, slices=5, phases=1)

    frames = dicom.read_series(tmp_path)

    assert len(frames) == 1
    assert list(frames[0].GetLargestPossibleRegion().GetSize()) == [10, 12, 5]


def test_single_slice_cine_is_a_stack_of_one(tmp_path):
    """A cine of one plane is ordinary in cardiac MR, and still 3D per frame."""
    write_cine_series(tmp_path, slices=1, phases=4)

    frames = dicom.read_series(tmp_path)

    assert len(frames) == 4
    for phase, frame in enumerate(frames):
        assert frame.GetImageDimension() == 3
        assert list(frame.GetLargestPossibleRegion().GetSize()) == [10, 12, 1]
        assert slice_values(frame) == [cine_voxel_value(0, phase)]


def test_single_slice_cine_keeps_its_geometry(tmp_path):
    direction = oblique_direction()
    write_cine_series(tmp_path, slices=1, phases=2, direction=direction)

    frame = dicom.read_series(tmp_path)[0]

    np.testing.assert_allclose(
        itk.array_from_matrix(frame.GetDirection()), direction, atol=1e-12
    )
    np.testing.assert_allclose(list(frame.GetSpacing()), [2.0, 1.5, 4.0])


# --- what it refuses to guess at ----------------------------------------------


def test_ragged_series_is_reported(tmp_path):
    """A missing image would otherwise reshape every frame after it."""
    write_cine_series(tmp_path, slices=4, phases=3, drop=(2, 1))

    with pytest.raises(ValueError, match="same number of images at every slice"):
        dicom.read_series(tmp_path)


def test_several_series_need_choosing_between(tmp_path):
    write_cine_series(tmp_path / "a", slices=2, phases=2, series_description="short")
    write_cine_series(tmp_path / "b", slices=2, phases=2, series_description="long")

    with pytest.raises(ValueError, match="2 DICOM series"):
        dicom.read_series(tmp_path)


def test_series_uid_chooses_between_them(tmp_path):
    wanted = write_cine_series(tmp_path / "a", slices=2, phases=3)
    write_cine_series(tmp_path / "b", slices=3, phases=2)

    frames = dicom.read_series(tmp_path, wanted)

    assert len(frames) == 3
    assert list(frames[0].GetLargestPossibleRegion().GetSize()) == [10, 12, 2]


def test_unknown_series_uid_lists_what_is_there(tmp_path):
    known = write_cine_series(tmp_path, slices=2, phases=2)

    with pytest.raises(ValueError, match="No series") as error:
        dicom.read_series(tmp_path, "1.2.3.4")

    assert known in str(error.value)


def test_empty_directory_is_reported(tmp_path):
    with pytest.raises(ValueError, match="No DICOM images found"):
        dicom.read_series(tmp_path)


def test_enhanced_multiframe_is_refused_clearly(tmp_path):
    write_cine_series(tmp_path, slices=1, phases=1)
    path = next(tmp_path.glob("*.dcm"))
    dataset = pd.dcmread(path)
    dataset.NumberOfFrames = 8
    dataset.save_as(path)

    with pytest.raises(ValueError, match="enhanced multi-frame"):
        dicom.read_series(tmp_path)


# --- living alongside other files ---------------------------------------------


def test_non_dicom_files_are_skipped(tmp_path):
    write_cine_series(tmp_path, slices=2, phases=2)
    (tmp_path / "NOTES.txt").write_text("not a dicom")
    (tmp_path / "DICOMDIR").write_bytes(b"\x00\x01\x02")

    frames = dicom.read_series(tmp_path)

    assert len(frames) == 2


def test_scan_finds_images_in_subdirectories(tmp_path):
    write_cine_series(tmp_path / "nested" / "deeper", slices=2, phases=2)

    assert len(dicom.scan(tmp_path)) == 4


# --- reaching the app through the ordinary config -----------------------------


def test_volume_loads_from_a_dicom_directory(tmp_path):
    write_cine_series(tmp_path, slices=3, phases=4)

    volume = Volume(label="cine", directory=tmp_path)

    assert volume.path_list == [tmp_path]
    assert len(volume.actors) == 4


def test_volume_passes_the_series_uid_through(tmp_path):
    write_cine_series(tmp_path / "a", slices=2, phases=2)
    wanted = write_cine_series(tmp_path / "b", slices=2, phases=5)

    volume = Volume(label="cine", directory=tmp_path, series_uid=wanted)

    assert len(volume.actors) == 5


def test_scene_opens_on_a_dicom_volume(tmp_path):
    write_cine_series(tmp_path, slices=3, phases=4)

    scene = Scene(volumes=[{"label": "cine", "directory": str(tmp_path)}])

    assert scene.nframes == 4


def test_a_directory_with_neither_frames_nor_dicom_is_reported(tmp_path):
    (tmp_path / "NOTES.txt").write_text("not a dicom")

    with pytest.raises(Exception, match="No DICOM images found"):
        Volume(label="empty", directory=tmp_path)


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(Exception, match="No files matching"):
        Volume(label="absent", directory=tmp_path / "nope")


def test_a_mesh_directory_is_not_read_as_dicom(tmp_path):
    """Only the image objects fall back to a series; a mesh says what is missing."""
    write_cine_series(tmp_path, slices=2, phases=2)

    with pytest.raises(Exception, match="No files matching"):
        Mesh(label="mesh", directory=tmp_path)
