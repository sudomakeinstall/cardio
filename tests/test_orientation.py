import itk
import numpy as np
import pytest

from cardio.orientation import (
    AngleUnits,
    EulerAxis,
    IndexOrder,
    axcode_transform_matrix,
    create_vtk_reslice_matrix,
    euler_angle_to_rotation_matrix,
    is_axis_aligned,
    is_righthanded_axcode,
    is_valid_axcode,
    minimal_rotation,
    quaternion_to_rotation_matrix,
    read_frames,
    read_source,
    temporal_frames,
)


def test_valid_axcodes():
    assert is_valid_axcode("LPS") is True
    assert is_valid_axcode("RAS") is True
    assert is_valid_axcode("PIL") is True


def test_invalid_axcodes():
    assert is_valid_axcode("lps") is False  # lowercase
    assert is_valid_axcode("LLL") is False  # duplicates
    assert is_valid_axcode("XYZ") is False  # invalid chars
    assert is_valid_axcode("LP") is False  # too short
    assert is_valid_axcode("LPSA") is False  # too long
    assert is_valid_axcode("LPA") is False  # missing S/I axis


def test_righthanded_axcodes():
    assert is_righthanded_axcode("LPS") is True  # L×P = S
    assert is_righthanded_axcode("RAS") is True  # R×A = S
    assert is_righthanded_axcode("PSL") is True  # P×S = L
    assert is_righthanded_axcode("LAS") is False  # L×A = -S ≠ S
    assert is_righthanded_axcode("RPS") is False  # R×P = -S ≠ S


def test_righthanded_invalid_input():
    with pytest.raises(ValueError):
        is_righthanded_axcode("invalid")


def test_axcode_transform_matrix():
    # Identity transformation
    T = axcode_transform_matrix("LPS", "LPS")
    np.testing.assert_array_equal(T, np.eye(3))

    # LPS to LAS transformation
    T = axcode_transform_matrix("LPS", "LAS")
    expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(T, expected)

    # Test coordinate transformation
    origin = np.array([1, 2, 3])
    transformed = T @ origin
    expected_origin = np.array([1, -2, 3])  # P flips to A
    np.testing.assert_array_equal(transformed, expected_origin)


def test_axcode_transform_matrix_invalid_input():
    with pytest.raises(ValueError):
        axcode_transform_matrix("invalid", "LPS")
    with pytest.raises(ValueError):
        axcode_transform_matrix("LPS", "invalid")


def test_euler_angle_to_rotation_matrix():
    # Test 90 degree rotation around X axis
    R = euler_angle_to_rotation_matrix(EulerAxis.X, 90)
    expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(R, expected, atol=1e-15)

    # Test with radians
    R = euler_angle_to_rotation_matrix(EulerAxis.Z, np.pi / 2, AngleUnits.RADIANS)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(R, expected, atol=1e-15)

    # Test identity (0 degrees)
    R = euler_angle_to_rotation_matrix(EulerAxis.Y, 0)
    np.testing.assert_array_equal(R, np.eye(3))


def test_create_vtk_reslice_matrix():
    # Create a simple transform and origin
    transform = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Cyclic permutation
    origin = [10.0, 20.0, 30.0]

    matrix = create_vtk_reslice_matrix(transform, origin)

    # Verify it's a VTK matrix
    import vtk

    assert isinstance(matrix, vtk.vtkMatrix4x4)

    # Verify the transform portion (upper 3x3)
    for i in range(3):
        for j in range(3):
            assert matrix.GetElement(i, j) == transform[i, j]

    # Verify the origin (4th column, first 3 rows)
    for i in range(3):
        assert matrix.GetElement(i, 3) == origin[i]

    # Verify the bottom row is [0, 0, 0, 1]
    assert matrix.GetElement(3, 0) == 0.0
    assert matrix.GetElement(3, 1) == 0.0
    assert matrix.GetElement(3, 2) == 0.0
    assert matrix.GetElement(3, 3) == 1.0


def test_is_axis_aligned():
    # Create axis-aligned image (identity direction)
    image_type = itk.Image[itk.F, 3]
    axis_aligned_image = image_type.New()
    axis_aligned_image.SetRegions(itk.Size[3]([10, 10, 10]))
    axis_aligned_image.Allocate()

    # Default direction is identity (axis-aligned)
    assert is_axis_aligned(axis_aligned_image) is True

    # Create non-axis-aligned image (rotated direction)
    non_aligned_image = image_type.New()
    non_aligned_image.SetRegions(itk.Size[3]([10, 10, 10]))
    non_aligned_image.Allocate()

    # Set a rotated direction matrix
    rotated_matrix = np.array(
        [[0.707, -0.707, 0], [0.707, 0.707, 0], [0, 0, 1]], dtype=np.float64
    )
    direction = itk.matrix_from_array(rotated_matrix)
    non_aligned_image.SetDirection(direction)

    assert is_axis_aligned(non_aligned_image) is False

    # Test axis-aligned with sign flips
    flipped_image = image_type.New()
    flipped_image.SetRegions(itk.Size[3]([10, 10, 10]))
    flipped_image.Allocate()

    flipped_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    direction = itk.matrix_from_array(flipped_matrix)
    flipped_image.SetDirection(direction)

    assert is_axis_aligned(flipped_image) is True

    # Test permuted axes (still axis-aligned)
    permuted_image = image_type.New()
    permuted_image.SetRegions(itk.Size[3]([10, 10, 10]))
    permuted_image.Allocate()

    permuted_matrix = np.array(
        [
            [0, 1, 0],  # X -> Y
            [0, 0, 1],  # Y -> Z
            [1, 0, 0],  # Z -> X
        ],
        dtype=np.float64,
    )
    direction = itk.matrix_from_array(permuted_matrix)
    permuted_image.SetDirection(direction)

    assert is_axis_aligned(permuted_image) is True


def make_4d_image(size=(10, 20, 30, 4), spatial_direction=None):
    """Build a 4D ITK image whose voxels encode their temporal index."""
    if spatial_direction is None:
        spatial_direction = np.eye(3)
    image = itk.Image[itk.F, 4].New()
    image.SetRegions(itk.Size[4](list(size)))
    image.Allocate()
    image.SetOrigin([5.0, 10.0, 15.0, 0.0])
    image.SetSpacing([1.0, 2.0, 3.0, 0.5])

    direction = np.eye(4)
    direction[:3, :3] = spatial_direction
    image.SetDirection(itk.matrix_from_array(direction))

    pixel_array = itk.array_view_from_image(image)
    for frame in range(size[3]):
        pixel_array[frame].fill(frame)

    return image


def test_temporal_frames_passes_through_3d():
    image = itk.Image[itk.F, 3].New()
    image.SetRegions(itk.Size[3]([10, 20, 30]))
    image.Allocate()

    frames = temporal_frames(image)

    assert len(frames) == 1
    assert frames[0] is image


def test_temporal_frames_splits_4d():
    image = make_4d_image(size=(10, 20, 30, 4))

    frames = temporal_frames(image)

    assert len(frames) == 4
    for index, frame in enumerate(frames):
        assert frame.GetImageDimension() == 3
        assert list(frame.GetLargestPossibleRegion().GetSize()) == [10, 20, 30]
        np.testing.assert_array_equal(list(frame.GetOrigin()), [5.0, 10.0, 15.0])
        np.testing.assert_array_equal(list(frame.GetSpacing()), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            itk.array_from_image(frame), np.full((30, 20, 10), index, dtype=np.float32)
        )


def test_temporal_frames_preserves_spatial_direction():
    flipped = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    image = make_4d_image(spatial_direction=flipped)

    for frame in temporal_frames(image):
        np.testing.assert_array_equal(
            itk.array_from_matrix(frame.GetDirection()), flipped
        )


def test_temporal_frames_rejects_other_dimensions():
    image = itk.Image[itk.F, 2].New()
    image.SetRegions(itk.Size[2]([10, 20]))
    image.Allocate()

    with pytest.raises(ValueError):
        temporal_frames(image)


def test_read_frames_splits_4d_file(tmp_path):
    flipped = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    path = tmp_path / "4d.nii.gz"
    itk.imwrite(
        make_4d_image(size=(10, 20, 30, 4), spatial_direction=flipped), str(path)
    )

    frames = read_frames(path)

    assert len(frames) == 4
    for index, frame in enumerate(frames):
        assert frame.GetImageDimension() == 3
        np.testing.assert_array_equal(
            itk.array_from_matrix(frame.GetDirection()), flipped
        )
        np.testing.assert_array_equal(
            itk.array_from_image(frame), np.full((30, 20, 10), index, dtype=np.float32)
        )


def test_read_frames_single_frame_from_3d_file(tmp_path):
    image = itk.Image[itk.F, 3].New()
    image.SetRegions(itk.Size[3]([10, 20, 30]))
    image.Allocate()
    image.SetSpacing([1.0, 2.0, 3.0])
    itk.array_view_from_image(image).fill(7)

    path = tmp_path / "3d.nii.gz"
    itk.imwrite(image, str(path))

    frames = read_frames(path)

    assert len(frames) == 1
    np.testing.assert_array_equal(
        itk.array_from_image(frames[0]), np.full((30, 20, 10), 7, dtype=np.float32)
    )


def test_read_source_returns_the_frames_read_frames_does(tmp_path):
    path = tmp_path / "4d.nii.gz"
    itk.imwrite(make_4d_image(size=(10, 20, 30, 4)), str(path))

    frames, source = read_source(path)

    assert len(frames) == len(read_frames(path))
    assert source.frame_count == 4
    assert source.format == "NIfTI"


def test_a_nifti_file_keeps_the_header_the_rebuild_would_drop(tmp_path):
    """Splitting a 4D file rebuilds each frame, which empties its dictionary."""
    path = tmp_path / "4d.nii.gz"
    itk.imwrite(make_4d_image(size=(10, 20, 30, 4)), str(path))

    _, source = read_source(path)

    assert source.header
    assert "qform_code" in source.header


def test_a_left_handed_file_records_that_it_was_corrected(tmp_path):
    left_handed = np.diag([1.0, 1.0, -1.0])
    path = tmp_path / "flipped.nii.gz"
    itk.imwrite(
        make_4d_image(size=(10, 20, 30, 2), spatial_direction=left_handed), str(path)
    )

    _, source = read_source(path)

    assert source.right_handed_correction is True


def test_a_right_handed_file_records_no_correction(tmp_path):
    path = tmp_path / "plain.nii.gz"
    itk.imwrite(make_4d_image(size=(10, 20, 30, 2)), str(path))

    _, source = read_source(path)

    assert source.right_handed_correction is False


def test_axis_convention_enum():
    assert IndexOrder.ITK.value == "itk"
    assert IndexOrder.ROMA.value == "roma"


def test_quaternion_roma_to_itk_conversion():
    """Roma→ITK quaternion conversion: [-z, -y, -x, w].

    90° around Roma Z (Left axis) should equal -90° around ITK X (Left axis).
    Both represent the same physical rotation, just in different frames.
    """
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)

    # +90° around Roma Z (Left) in Roma frame
    q_roma = [0.0, 0.0, s, c]
    r_roma = quaternion_to_rotation_matrix(q_roma)

    # Convert to ITK: [-z, -y, -x, w]
    q_itk = [-q_roma[2], -q_roma[1], -q_roma[0], q_roma[3]]
    r_itk_from_quat = quaternion_to_rotation_matrix(q_itk)

    # -90° around ITK X (Left) using Euler
    r_itk_from_euler = euler_angle_to_rotation_matrix(
        EulerAxis.X, -90.0, AngleUnits.DEGREES
    )

    assert np.allclose(r_itk_from_quat, r_itk_from_euler, atol=1e-6)

    # Also verify basis-change formula: R_itk = P @ R_roma @ P
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    r_itk_from_basis_change = P @ r_roma @ P
    assert np.allclose(r_itk_from_quat, r_itk_from_basis_change, atol=1e-6)


def test_quaternion_to_rotation_matrix_identity():
    mat = quaternion_to_rotation_matrix([0.0, 0.0, 0.0, 1.0])
    assert mat.shape == (3, 3)
    assert np.allclose(mat, np.eye(3))


def test_quaternion_to_rotation_matrix_90z():
    """90° around Z: quaternion [0, 0, sin(45°), cos(45°)] matches euler."""
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    mat = quaternion_to_rotation_matrix([0.0, 0.0, s, c])
    expected = euler_angle_to_rotation_matrix(EulerAxis.Z, 90.0, AngleUnits.DEGREES)
    assert np.allclose(mat, expected, atol=1e-6)


def test_quaternion_to_rotation_matrix_is_rotation():
    """Result must be a proper rotation matrix (orthogonal, det=1)."""
    # -30° around Y: [0, sin(-15°), 0, cos(15°)]
    s, c = np.sin(np.radians(-15)), np.cos(np.radians(15))
    q = [0.0, s, 0.0, c]
    mat = quaternion_to_rotation_matrix(q)
    assert np.allclose(mat @ mat.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-6)


def test_minimal_rotation_identity():
    axis = np.array([0.0, 0.0, 1.0])
    assert minimal_rotation(axis, axis) == pytest.approx(np.eye(3), abs=1e-12)


def test_minimal_rotation_maps_source_onto_target():
    source = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
    rotation = minimal_rotation(source, target)
    assert rotation @ source == pytest.approx(target, abs=1e-12)
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_minimal_rotation_handles_opposed_vectors():
    source = np.array([0.0, 0.0, 1.0])
    rotation = minimal_rotation(source, -source)
    assert rotation @ source == pytest.approx(-source, abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_minimal_rotation_is_the_smallest_one():
    """No rotation angle smaller than the angle between the vectors."""
    source = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    rotation = minimal_rotation(source, target)
    angle = np.degrees(np.arccos((np.trace(rotation) - 1.0) / 2.0))
    assert angle == pytest.approx(90.0, abs=1e-9)
