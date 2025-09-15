# System
import pytest

# Third Party
import numpy as np
import itk

# Internal
from cardio.orientation import (
    AngleUnits,
    EulerAxis,
    axcode_transform_matrix,
    create_vtk_reslice_matrix,
    euler_angle_to_rotation_matrix,
    is_axis_aligned,
    is_righthanded_axcode,
    is_valid_axcode,
    reset_direction,
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


def test_reset_direction():
    image_type = itk.Image[itk.F, 3]

    # Test with flipped image
    flipped_image = image_type.New()
    flipped_image.SetRegions(itk.Size[3]([10, 20, 30]))
    flipped_image.Allocate()
    flipped_image.SetOrigin([5.0, 10.0, 15.0])
    flipped_image.SetSpacing([1.0, 2.0, 3.0])

    flipped_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    direction = itk.matrix_from_array(flipped_matrix)
    flipped_image.SetDirection(direction)

    # Fill with test data
    pixel_array = itk.array_from_image(flipped_image)
    pixel_array.fill(42)

    reset_image = reset_direction(flipped_image)

    # Verify output has identity direction
    assert is_axis_aligned(reset_image)
    output_direction = itk.array_from_matrix(reset_image.GetDirection())
    np.testing.assert_array_equal(output_direction, np.eye(3))

    # Test with permuted image
    permuted_image = image_type.New()
    permuted_image.SetRegions(itk.Size[3]([10, 20, 30]))
    permuted_image.Allocate()
    permuted_image.SetOrigin([0.0, 0.0, 0.0])
    permuted_image.SetSpacing([1.0, 1.0, 1.0])

    permuted_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
    direction = itk.matrix_from_array(permuted_matrix)
    permuted_image.SetDirection(direction)

    reset_image = reset_direction(permuted_image)

    # Verify output has identity direction
    assert is_axis_aligned(reset_image)
    output_direction = itk.array_from_matrix(reset_image.GetDirection())
    np.testing.assert_array_equal(output_direction, np.eye(3))
