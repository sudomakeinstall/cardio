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


def make_image(dims=(8, 10, 12), spacing=(1.0, 2.0, 3.0)) -> vtk.vtkImageData:
    """A small scalar image with non-uniform spacing, so axes cannot be confused."""
    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    image.SetSpacing(*spacing)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.AllocateScalars(vtk.VTK_SHORT, 1)
    return image


def write_segmentation(directory, arrays, label="s", stem="seg") -> Segmentation:
    """One Segmentation over a series of label arrays, written out as frames."""
    names = []
    for frame, array in enumerate(arrays):
        name = f"{stem}{frame}.nii.gz"
        itk.imwrite(itk.image_from_array(array), str(directory / name))
        names.append(name)
    return Segmentation(label=label, directory=directory, file_paths=names)


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
