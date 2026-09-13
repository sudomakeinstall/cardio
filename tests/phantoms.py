"""Synthetic images and segmentations the tests measure against.

Every phantom is built from an array whose geometry the test can state in
one sentence, so an expectation can be written down rather than recorded.
"""

# System
import pathlib as pl

# Third Party
import itk
import numpy as np
import pydicom as pd
import vtk

# Internal
from cardio.segmentation import Segmentation


def make_image(
    dims=(8, 10, 12), spacing=(1.0, 2.0, 3.0), direction=None
) -> vtk.vtkImageData:
    """A small scalar image with non-uniform spacing, so axes cannot be confused.

    ``direction`` gives it the acquisition axes of an oblique series, which is
    what makes a reslice's auto-cropped extent land off the whole millimetre.
    """
    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    image.SetSpacing(*spacing)
    image.SetOrigin(0.0, 0.0, 0.0)
    if direction is not None:
        matrix = vtk.vtkMatrix3x3()
        for row in range(3):
            for column in range(3):
                matrix.SetElement(row, column, float(direction[row][column]))
        image.SetDirectionMatrix(matrix)
    image.AllocateScalars(vtk.VTK_SHORT, 1)
    return image


def rotation_about_z(degrees: float) -> np.ndarray:
    """A proper rotation, for an image that was not acquired axis-aligned."""
    angle = np.radians(degrees)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def write_segmentation(
    directory,
    arrays,
    label="s",
    stem="seg",
    spacing=None,
    origin=None,
    direction=None,
) -> Segmentation:
    """One Segmentation over a series of label arrays, written out as frames.

    ``spacing``, ``origin`` and ``direction`` are in ITK order. Left out, the
    image is a unit grid at the world origin, which is all most phantoms need;
    given, they are what tells a centroid read in index space apart from one
    read in world space.
    """
    names = []
    for frame, array in enumerate(arrays):
        name = f"{stem}{frame}.nii.gz"
        image = itk.image_from_array(np.ascontiguousarray(array))
        if spacing is not None:
            image.SetSpacing([float(value) for value in spacing])
        if origin is not None:
            image.SetOrigin([float(value) for value in origin])
        if direction is not None:
            image.SetDirection(
                itk.matrix_from_array(np.asarray(direction, dtype=np.float64))
            )
        itk.imwrite(image, str(directory / name))
        names.append(name)
    return Segmentation(label=label, directory=directory, file_paths=names)


def to_world(index, spacing=None, origin=None, direction=None) -> np.ndarray:
    """An ITK-order index carried into world coordinates by hand.

    The long way round on purpose: a test that wants to know where a voxel
    landed should not ask the same transform the code under test asked.
    """
    index = np.asarray(index, dtype=np.float64)
    spacing = np.ones(3) if spacing is None else np.asarray(spacing, dtype=np.float64)
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    direction = np.eye(3) if direction is None else np.asarray(direction, np.float64)
    return origin + direction @ (spacing * index)


# --- an L, for a centroid that is not the centre of anything ------------------

# Two axis-aligned blocks meeting at a corner, as half-open (i, j, k) index
# ranges. Every block is a whole number of voxels, so the solid centroid is the
# volume-weighted mean of the two block centres exactly, with nothing to
# discretise -- and it sits well away from the surface centroid, which is what
# makes the shape worth having.
L_BLOCKS = (((2, 18), (2, 8), (2, 8)), ((2, 8), (2, 8), (8, 20)))
L_SHAPE = (24, 12, 22)


def l_block_array() -> np.ndarray:
    """An L-shaped label 1, as a (k, j, i) array."""
    array = np.zeros(L_SHAPE[::-1], dtype=np.uint8)
    for (i0, i1), (j0, j1), (k0, k1) in L_BLOCKS:
        array[k0:k1, j0:j1, i0:i1] = 1
    return array


def l_block_centroid(spacing=None, origin=None, direction=None) -> np.ndarray:
    """Where ``l_block_array``'s solid centroid falls, in world coordinates."""
    centres, weights = [], []
    for block in L_BLOCKS:
        centres.append([(low + high - 1) / 2.0 for low, high in block])
        weights.append(float(np.prod([high - low for low, high in block])))
    index = np.average(np.array(centres), axis=0, weights=np.array(weights))
    return to_world(index, spacing, origin, direction)


# --- an interface of two facets, for a centroid that has to be area-weighted ---

# A box cut by a surface that is flat up to ``FACET_KNEE`` and then rises at 45
# degrees until it leaves the box at ``FACET_CEILING``. The two facets have the
# same footprint per unit of x but not the same area, and SurfaceNets puts about
# the same number of vertices on each -- so an unweighted mean of the vertices
# lands somewhere the interface's own geometry never puts it.
FACET_EXTENT = (32.0, 8.0, 32.0)
FACET_KNEE = 12.0
FACET_FLOOR = 16.0
FACET_CEILING = 30.0


def facet_interface_array(spacing) -> np.ndarray:
    """Labels 1 and 2 either side of the two-facet surface, as a (k, j, i) array."""
    spacing = np.asarray(spacing, dtype=np.float64)
    counts = [round(e / s) + 1 for e, s in zip(FACET_EXTENT, spacing)]
    i, _, k = np.meshgrid(*[np.arange(n) for n in counts], indexing="ij")
    x, z = i * spacing[0], k * spacing[2]

    surface = np.where(x < FACET_KNEE, FACET_FLOOR, FACET_FLOOR + (x - FACET_KNEE))
    array = np.ones(counts, dtype=np.uint8)
    array[z > surface] = 2
    array[z > FACET_CEILING] = 0
    return np.ascontiguousarray(array.transpose(2, 1, 0))


def facet_interface_centroid() -> np.ndarray:
    """The area-weighted centre of that surface, worked out from its two facets."""
    width, depth = FACET_EXTENT[0], FACET_EXTENT[1]
    knee_end = FACET_KNEE + (FACET_CEILING - FACET_FLOOR)

    flat_area = FACET_KNEE * depth
    tilted_area = (knee_end - FACET_KNEE) * depth * np.sqrt(2.0)
    flat = np.array([FACET_KNEE / 2.0, depth / 2.0, FACET_FLOOR])
    tilted = np.array(
        [
            (FACET_KNEE + knee_end) / 2.0,
            depth / 2.0,
            (FACET_FLOOR + FACET_CEILING) / 2.0,
        ]
    )
    assert knee_end <= width
    return (flat_area * flat + tilted_area * tilted) / (flat_area + tilted_area)


M = 32
MOVING_FRAMES = 3


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


def stacked_segmentation(directory, tilt_degrees: float = 30.0) -> Segmentation:
    return write_segmentation(directory, [stacked_array(tilt_degrees)], stem="stack")


def moving_stack_segmentation(directory) -> Segmentation:
    """The same stack, displaced two voxels along x per frame."""
    return write_segmentation(
        directory,
        [stacked_array(30.0, offset=2 * frame) for frame in range(MOVING_FRAMES)],
        stem="move",
    )


# --- DICOM cine series --------------------------------------------------------

CINE_ROWS, CINE_COLUMNS = 12, 10
CINE_PIXEL_SPACING = (1.5, 2.0)
CINE_SLICE_SPACING = 4.0


def cine_voxel_value(slice_index: int, phase: int) -> int:
    """A value naming the (slice, phase) cell it belongs to.

    Every voxel of a frame's slice carries this, so a test can say which image
    landed where without depending on the pixel content itself.
    """
    return 100 * (slice_index + 1) + phase


def write_cine_series(
    directory,
    slices: int = 4,
    phases: int = 3,
    direction: np.ndarray | None = None,
    origin=(-7.0, 5.0, -3.0),
    series_uid: str | None = None,
    series_description: str = "cine",
    instance_order: str = "slice_major",
    drop: tuple[int, int] | None = None,
    with_trigger_time: bool = True,
):
    """A classic single-frame cine series: one file per slice per phase.

    ``direction`` supplies the acquisition axes, so an oblique stack can be
    written as easily as an axis-aligned one.  ``instance_order`` chooses how
    InstanceNumber runs, and ``drop`` removes one (slice, phase) file, so the
    grouping can be shown not to depend on either.
    """
    directory = pl.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if direction is None:
        direction = np.eye(3)
    direction = np.asarray(direction, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)

    uid = series_uid or pd.uid.generate_uid()
    study_uid = pd.uid.generate_uid()
    frame_of_reference_uid = pd.uid.generate_uid()
    normal = direction[:, 2]

    cells = [(s, p) for s in range(slices) for p in range(phases)]
    if instance_order == "phase_major":
        cells = [(s, p) for p in range(phases) for s in range(slices)]
    elif instance_order == "shuffled":
        rng = np.random.default_rng(0)
        cells = [cells[i] for i in rng.permutation(len(cells))]

    for number, (slice_index, phase) in enumerate(cells, start=1):
        if drop is not None and (slice_index, phase) == drop:
            continue

        dataset = pd.dataset.Dataset()
        dataset.file_meta = pd.dataset.FileMetaDataset()
        dataset.file_meta.MediaStorageSOPClassUID = pd.uid.MRImageStorage
        dataset.file_meta.MediaStorageSOPInstanceUID = pd.uid.generate_uid()
        dataset.file_meta.TransferSyntaxUID = pd.uid.ExplicitVRLittleEndian

        dataset.SOPClassUID = pd.uid.MRImageStorage
        dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
        dataset.PatientName = "Phantom^Cine"
        dataset.PatientID = "PHANTOM-1"
        dataset.Modality = "MR"
        dataset.StudyDate = "20260101"
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = uid
        dataset.FrameOfReferenceUID = frame_of_reference_uid
        dataset.SeriesDescription = series_description
        dataset.SeriesNumber = 1
        dataset.InstanceNumber = number

        dataset.Rows = CINE_ROWS
        dataset.Columns = CINE_COLUMNS
        dataset.PixelSpacing = list(CINE_PIXEL_SPACING)
        dataset.SliceThickness = CINE_SLICE_SPACING
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0

        dataset.ImageOrientationPatient = [float(v) for v in direction[:, :2].T.ravel()]
        dataset.ImagePositionPatient = [
            float(v) for v in origin + normal * (slice_index * CINE_SLICE_SPACING)
        ]
        if with_trigger_time:
            dataset.TriggerTime = float(phase * 50)

        array = np.full(
            (CINE_ROWS, CINE_COLUMNS), cine_voxel_value(slice_index, phase), np.uint16
        )
        dataset.PixelData = array.tobytes()

        dataset.save_as(directory / f"{number:04d}.dcm", enforce_file_format=True)

    return uid
