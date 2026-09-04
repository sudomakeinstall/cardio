import dataclasses as dc
import enum
import pathlib as pl

import itk
import numpy as np
import vtk

from . import dicom


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


def axcode_from_direction(direction: np.ndarray) -> str:
    """The letters the image's index axes point toward, in LPS.

    Each column of an ITK direction matrix is where one index axis points in
    world space; the letter reported for it is the canonical direction that
    column lies closest to, so an oblique acquisition is described by the axes
    it most nearly runs along rather than refused.
    """
    direction = np.asarray(direction, dtype=np.float64)

    letters = []
    for column in range(3):
        axis = direction[:, column]
        letters.append(
            max(
                AXCODE_VECTORS,
                key=lambda code: float(np.array(AXCODE_VECTORS[code]) @ axis),
            )
        )
    return "".join(letters)


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


def angle_from_degrees(angle: float, units: AngleUnits) -> float:
    """An angle in degrees, expressed in ``units``."""
    match units:
        case AngleUnits.RADIANS:
            return float(np.radians(angle))
        case AngleUnits.DEGREES:
            return float(angle)


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


def quaternion_to_rotation_matrix(q: list[float]) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix (normalizes input)."""
    # Deferred: torch costs about half a second to import, and only the
    # quaternion paths need it.
    import roma
    import torch as t

    tensor = t.tensor(q, dtype=t.float64)
    tensor = tensor / tensor.norm()
    return roma.unitquat_to_rotmat(tensor).numpy()


def cumulative_rotation_matrix(
    rotation_sequence, rotation_angles=None, units: AngleUnits = AngleUnits.DEGREES
) -> np.ndarray:
    """Compose a rotation sequence into a single 3x3 matrix.

    Steps may be dicts or RotationStep objects, and are applied in order.
    """
    cumulative = np.eye(3)
    for i, step in enumerate(rotation_sequence or []):
        quat = step.get("quaternion") if isinstance(step, dict) else step.quaternion
        if quat is not None:
            rotation_matrix = quaternion_to_rotation_matrix(quat)
        else:
            axis = step.get("axis") if isinstance(step, dict) else step.axis
            angle = rotation_angles.get(i, 0) if rotation_angles else 0
            rotation_matrix = euler_angle_to_rotation_matrix(
                EulerAxis(axis), angle, units
            )
        cumulative = cumulative @ rotation_matrix
    return cumulative


def minimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest rotation carrying unit vector ``source`` onto ``target``.

    Undefined for exactly opposed vectors; an arbitrary perpendicular axis is
    used in that case.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)

    axis = np.cross(source, target)
    cosine = float(source @ target)
    sine = float(np.linalg.norm(axis))

    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        perpendicular = np.array([1.0, 0.0, 0.0])
        if abs(source @ perpendicular) > 0.9:
            perpendicular = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, perpendicular)
        axis = axis / np.linalg.norm(axis)
        cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return np.eye(3) + 2.0 * cross @ cross

    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + cross + cross @ cross * ((1.0 - cosine) / (sine**2))


def slerp_rotation_matrices(
    start: np.ndarray, end: np.ndarray, fraction: float
) -> np.ndarray:
    """Rotation ``fraction`` of the way along the geodesic from ``start`` to ``end``.

    Takes the shortest arc, so opposed rotations interpolate the short way round.
    """
    # Deferred; see quaternion_to_rotation_matrix.
    import roma
    import torch as t

    interpolated = roma.rotmat_slerp(
        t.tensor(np.asarray(start), dtype=t.float64),
        t.tensor(np.asarray(end), dtype=t.float64),
        t.tensor([float(fraction)], dtype=t.float64),
    )
    return interpolated[0].numpy()


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to a quaternion [x, y, z, w]."""
    # Deferred; see quaternion_to_rotation_matrix.
    import roma
    import torch as t

    quat = roma.rotmat_to_unitquat(t.tensor(np.asarray(matrix), dtype=t.float64))
    return [float(v) for v in quat]


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
    return bool(np.allclose(np.abs(non_zero_values), 1.0))


def temporal_frames(image) -> list:
    """Split a 4D ITK image into its 3D temporal frames.

    A 3D image is returned unchanged as a single frame, so callers can treat
    per-frame files and a single 4D file identically.
    """
    dimension = image.GetImageDimension()

    if dimension == 3:
        return [image]

    if dimension != 4:
        raise ValueError(f"Expected a 3D or 4D image, got {dimension}D")

    origin = np.array(image.GetOrigin())[:3]
    spacing = np.array(image.GetSpacing())[:3]
    direction = itk.matrix_from_array(
        itk.array_from_matrix(image.GetDirection())[:3, :3]
    )
    pixel_array = itk.array_from_image(image)

    frames = []
    for frame_array in pixel_array:
        frame = itk.image_from_array(np.ascontiguousarray(frame_array))
        frame.SetOrigin(origin)
        frame.SetSpacing(spacing)
        frame.SetDirection(direction)
        frames.append(frame)

    return frames


def ensure_right_handed(image):
    """An equivalent image whose direction matrix is right-handed.

    A left-handed direction is perfectly legal -- NIfTI writes one whenever
    qfac is negative, and a DICOM series does whenever its slices run against
    the normal -- but ``vtkGPUVolumeRayCastMapper`` draws nothing for one.

    Reversing the slice axis and negating the matching direction column trades
    the handedness for a flip, which is exact: it renumbers voxels rather than
    resampling them, and every voxel keeps the world position it had.
    """
    direction = itk.array_from_matrix(image.GetDirection())
    if np.linalg.det(direction) > 0:
        return image

    size = np.array(image.GetLargestPossibleRegion().GetSize())
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())

    # The far end of the slice axis becomes the near end.
    flipped_axis = 2
    new_origin = origin + direction[:, flipped_axis] * (
        (size[flipped_axis] - 1) * spacing[flipped_axis]
    )
    new_direction = direction.copy()
    new_direction[:, flipped_axis] *= -1.0

    # ITK axis 2 is the slowest-varying numpy axis.
    pixel_array = np.flip(itk.array_from_image(image), axis=0)

    output = itk.image_from_array(np.ascontiguousarray(pixel_array))
    output.SetOrigin(new_origin)
    output.SetSpacing(spacing)
    output.SetDirection(itk.matrix_from_array(new_direction))
    return output


def is_right_handed(image) -> bool:
    """Whether the image's direction matrix has a positive determinant."""
    return bool(np.linalg.det(itk.array_from_matrix(image.GetDirection())) > 0)


def file_format(path: pl.Path) -> str:
    """What to call the file's format on the metadata sheet."""
    suffixes = "".join(path.suffixes[-2:]).lower()
    if suffixes.endswith((".nii", ".nii.gz")):
        return "NIfTI"
    if suffixes.endswith((".mha", ".mhd")):
        return "MetaImage"
    if suffixes.endswith((".nrrd", ".nhdr")):
        return "NRRD"
    return path.suffix.lstrip(".").upper() or "image"


def image_header(image) -> dict[str, str]:
    """An ITK image's metadata dictionary, as strings.

    This is where a format's own header survives: NIfTI's ``descrip``,
    ``qform_code`` and ``srow_*`` all arrive here.  It has to be read from the
    image ``itk.imread`` returned, because splitting a 4D file or correcting
    the handedness rebuilds the image and leaves the dictionary behind.

    Entries whose type has no python conversion raise rather than convert, and
    are skipped: a header nobody can read is not worth failing a load over.
    """
    dictionary = image.GetMetaDataDictionary()

    fields = {}
    for key in dictionary.GetKeys():
        try:
            fields[key] = str(dictionary[key])
        except (RuntimeError, TypeError, KeyError):
            continue
    return fields


@dc.dataclass(frozen=True)
class Geometry:
    """Where an image sits and how finely it is sampled, as it is rendered."""

    size: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: np.ndarray
    voxel_type: str
    intensity_range: tuple[float, float]

    @property
    def axcode(self) -> str:
        return axcode_from_direction(self.direction)

    @property
    def extent(self) -> tuple[float, float, float]:
        """The image's span in millimetres along each index axis."""
        return tuple(float(n * s) for n, s in zip(self.size, self.spacing))


def geometry_of(image) -> Geometry:
    """One ITK image's geometry, read out into plain python."""
    array = itk.array_view_from_image(image)
    return Geometry(
        size=tuple(int(n) for n in image.GetLargestPossibleRegion().GetSize()),
        spacing=tuple(float(s) for s in image.GetSpacing()),
        origin=tuple(float(o) for o in image.GetOrigin()),
        direction=itk.array_from_matrix(image.GetDirection()),
        voxel_type=str(array.dtype),
        intensity_range=(float(array.min()), float(array.max())),
    )


@dc.dataclass(frozen=True)
class Source:
    """Where an object's images came from, and what they came with.

    The header is the only record of how the image was written -- both readers
    discard it on the way to VTK -- so it is captured at the one moment it
    exists rather than recovered later.  The frames themselves are not kept:
    the caller converts them to VTK straight away, and holding both would
    double the memory a series costs.
    """

    format: str
    header: dict[str, str]
    right_handed_correction: bool
    frame_count: int
    geometry: Geometry


def read_source(path, series_uid: str | None = None) -> tuple[list, Source]:
    """Read a path as 3D frames, with a description of where they came from.

    A 4D file is split along time, so a directory of DICOM, a file per frame
    and a single 4D file all reach the caller as a list of 3D frames.
    """
    if path.is_dir():
        instances = dicom.select_instances(path, series_uid)
        frames = dicom.read_instances(instances)
        source_format = "DICOM series"
        fields = dicom.header(instances[0])
    else:
        image = itk.imread(path)
        frames = temporal_frames(image)
        source_format = file_format(pl.Path(path))
        fields = image_header(image)

    corrected = any(not is_right_handed(frame) for frame in frames)
    frames = [ensure_right_handed(frame) for frame in frames]

    source = Source(
        format=source_format,
        header=fields,
        right_handed_correction=corrected,
        frame_count=len(frames),
        geometry=geometry_of(frames[0]),
    )
    return frames, source


def read_frames(path, series_uid: str | None = None) -> list:
    """Read a path as 3D frames: a DICOM series directory, or an image file."""
    return read_source(path, series_uid)[0]


def create_vtk_reslice_matrix(transform_3x3, origin):
    """Create 4x4 VTK reslice matrix from 3x3 transform and origin.

    Args:
        transform_3x3: 3x3 coordinate transformation matrix
        origin: 3-element origin position

    Returns:
        vtk.vtkMatrix4x4 for use with VTK reslice operations
    """
    matrix = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            matrix.SetElement(i, j, transform_3x3[i, j])
        matrix.SetElement(i, 3, origin[i])
    matrix.SetElement(3, 3, 1.0)
    return matrix
