import enum

import itk
import numpy as np


# DICOM LPS canonical orientation vector mappings
class EulerAxis(enum.StrEnum):
    X = "X"
    Y = "Y"
    Z = "Z"


class IndexOrder(enum.StrEnum):
    ITK = "itk"  # X=Left, Y=Posterior, Z=Superior
    ROMA = "roma"  # X=Superior, Y=Posterior, Z=Left


class AngleUnits(enum.StrEnum):
    DEGREES = "degrees"
    RADIANS = "radians"


AXCODE_VECTORS = {
    "L": (1, 0, 0),
    "R": (-1, 0, 0),
    "P": (0, 1, 0),
    "A": (0, -1, 0),
    "S": (0, 0, 1),
    "I": (0, 0, -1),
}


def is_valid_axcode(axcode: str) -> bool:
    """Validate medical imaging axcode string.

    Valid axcode must have exactly 3 uppercase characters with:
    - One of L or R (Left/Right)
    - One of A or P (Anterior/Posterior)
    - One of S or I (Superior/Inferior)
    - No repeated characters
    """
    if len(axcode) != 3:
        return False

    if len(set(axcode)) != 3:
        return False

    has_lr = any(c in axcode for c in "LR")
    has_ap = any(c in axcode for c in "AP")
    has_si = any(c in axcode for c in "SI")

    valid_chars = set("LRAPSI")
    has_only_valid = all(c in valid_chars for c in axcode)

    return has_lr and has_ap and has_si and has_only_valid


def is_righthanded_axcode(axcode: str) -> bool:
    """Check if axcode represents a right-handed coordinate system.

    Right-handed when cross product of first two axes equals third axis.
    Uses DICOM LPS canonical orientation.
    """
    if not is_valid_axcode(axcode):
        raise ValueError(f"Invalid axcode: {axcode}")

    v1 = np.array(AXCODE_VECTORS[axcode[0]])
    v2 = np.array(AXCODE_VECTORS[axcode[1]])
    v3 = np.array(AXCODE_VECTORS[axcode[2]])

    cross = np.cross(v1, v2)

    return np.array_equal(cross, v3)


def axcode_transform_matrix(from_axcode: str, to_axcode: str) -> np.ndarray:
    """Calculate transformation matrix between two coordinate spaces.

    Returns matrix T such that: new_coords = T @ old_coords
    Uses DICOM LPS canonical orientation for vector mappings.
    """
    if not is_valid_axcode(from_axcode):
        raise ValueError(f"Invalid source axcode: {from_axcode}")
    if not is_valid_axcode(to_axcode):
        raise ValueError(f"Invalid target axcode: {to_axcode}")

    # Create basis matrices (each column is a basis vector)
    from_basis = np.array([AXCODE_VECTORS[c] for c in from_axcode]).T
    to_basis = np.array([AXCODE_VECTORS[c] for c in to_axcode]).T

    # Transformation matrix: T = to_basis @ from_basis^(-1)
    return to_basis @ np.linalg.inv(from_basis)


def euler_angle_to_rotation_matrix(
    axis: EulerAxis, angle: float, units: AngleUnits = AngleUnits.DEGREES
) -> np.ndarray:
    """Create rotation matrix for given axis and angle.

    Args:
        axis: Rotation axis (X, Y, or Z)
        angle: Rotation angle
        units: Angle units (degrees or radians)

    Returns:
        3x3 rotation matrix
    """
    match units:
        case AngleUnits.DEGREES:
            angle_rad = np.radians(angle)
        case AngleUnits.RADIANS:
            angle_rad = angle

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    match axis:
        case EulerAxis.X:
            return np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        case EulerAxis.Y:
            return np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        case EulerAxis.Z:
            return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])


def is_axis_aligned(image) -> bool:
    """Check if ITK image orientation is axis-aligned.

    An axis-aligned image has a direction matrix where:
    - Each column has exactly one non-zero entry
    - Non-zero entries are ±1

    Args:
        image: ITK image object

    Returns:
        True if image is axis-aligned, False otherwise
    """
    direction = itk.array_from_matrix(image.GetDirection())

    # Check each column has exactly one non-zero entry
    for col in range(direction.shape[1]):
        non_zero_count = np.count_nonzero(direction[:, col])
        if non_zero_count != 1:
            return False

    # Check non-zero entries are ±1
    non_zero_values = direction[direction != 0]
    if not np.allclose(np.abs(non_zero_values), 1.0):
        return False

    return True


def reset_direction(image):
    """Reset image direction to identity matrix, preserving physical extent."""
    assert is_axis_aligned(image), "Input image must be axis-aligned"

    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = itk.array_from_matrix(image.GetDirection())
    size = np.array(image.GetLargestPossibleRegion().GetSize())
    pixel_array = itk.array_from_image(image)

    permutation = []
    flips = []

    for col in range(3):
        row = np.nonzero(direction[:, col])[0][0]
        permutation.append(row)
        flips.append(direction[row, col] < 0)

    array_permutation = [2 - p for p in reversed(permutation)]
    pixel_array = np.transpose(pixel_array, array_permutation)

    for i, should_flip in enumerate(reversed(flips)):
        if should_flip:
            pixel_array = np.flip(pixel_array, axis=i)

    new_spacing = spacing[permutation]

    adjusted_origin = origin.copy()
    for i, should_flip in enumerate(flips):
        if should_flip:
            image_axis = permutation[i]
            extent_vector = (
                direction[:, image_axis] * (size[image_axis] - 1) * spacing[image_axis]
            )
            adjusted_origin += extent_vector

    new_origin = adjusted_origin[permutation]

    output = itk.image_from_array(pixel_array)
    output.SetOrigin(new_origin)
    output.SetSpacing(new_spacing)

    return output


def create_vtk_reslice_matrix(transform_3x3, origin):
    """Create 4x4 VTK reslice matrix from 3x3 transform and origin.

    Args:
        transform_3x3: 3x3 coordinate transformation matrix
        origin: 3-element origin position

    Returns:
        vtk.vtkMatrix4x4 for use with VTK reslice operations
    """
    import vtk

    matrix = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            matrix.SetElement(i, j, transform_3x3[i, j])
        matrix.SetElement(i, 3, origin[i])
    matrix.SetElement(3, 3, 1.0)
    return matrix
