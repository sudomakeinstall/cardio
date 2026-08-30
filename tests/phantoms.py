"""Synthetic images and segmentations the tests measure against.

Every phantom is built from an array whose geometry the test can state in
one sentence, so an expectation can be written down rather than recorded.
"""

# Third Party
import itk
import numpy as np
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
