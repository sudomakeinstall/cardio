"""Where a reslice's pixels sit in the patient.

``ResliceSet.set_pose`` builds the reslice axes in LPS and hands them straight
to VTK, which is the one convention the app ever uses, so the matrix VTK holds
is world-LPS and DICOM's geometry can be read off it rather than reconstructed.
"""

# System
import math

# Third Party
import numpy as np
import vtk
from vtk.util import numpy_support as vtknp

# Internal
from ..orientation import axcode_from_direction
from ..reslice import configure_reslice
from .base import Location, Plane


def reslice_axes(reslice) -> np.ndarray:
    """The 4x4 pose the reslice was given, as an array.

    VTK maps output coordinates to input with it: a point of the cut at output
    ``(u, v, 0)`` is at ``axes[:3, :3] @ (u, v, 0) + axes[:3, 3]`` in the patient.
    """
    matrix = reslice.GetResliceAxes()
    return np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])


def scalars_2d(image) -> np.ndarray:
    """A 2D image's scalars as ``(rows, columns)``, in the type it holds them.

    Row zero is the low end of the output's y, which is the bottom of the view:
    a reslice is not a framebuffer, so there is nothing to undo here.  Turning
    the cut the way a viewer draws it is the caller's, once it is in a position
    to say so in the geometry as well.
    """
    columns, rows, _ = image.GetDimensions()
    return vtknp.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(rows, columns)


def _last_index(low: float, high: float, spacing: float) -> int:
    """The last index of a grid at ``spacing`` that covers ``low`` to ``high``."""
    return max(0, math.ceil((high - low) / spacing))


def square_pixels(reslice, rectangle=None) -> vtk.vtkImageReslice:
    """The same cut again, resampled onto square pixels, cropped to what is shown.

    An autocropped oblique cut comes out with a spacing of its own on each
    axis: a long axis view of a 0.59 by 0.59 by 2mm study lands on 0.93 by
    1.66mm pixels.  ``PixelSpacing`` says so, but a Secondary Capture carries
    no Image Plane module for a viewer to expect one in, and a viewer that
    draws the pixels square draws the anatomy squashed.  So the capture is
    written square, at the finest spacing the volume itself holds, which is
    what the mosaic already resamples its tiles onto.

    ``rectangle`` is the part of the cut to keep, in the cut's own coordinates
    -- what the view is showing, so that the data and the picture taken of the
    same view are framed alike.  Without one the cut is written whole, which is
    what the autocrop gives.

    A pipeline of its own rather than a change to the one on screen: the views
    a person posed are not resampled because a capture was taken of them.
    """
    image_data = reslice.GetInput()
    spacing = min(image_data.GetSpacing())

    square = configure_reslice(image_data, "linear", reslice.GetBackgroundLevel())
    square.SetResliceAxes(reslice.GetResliceAxes())
    square.SetOutputSpacing(spacing, spacing, spacing)

    if rectangle is not None:
        (low_x, high_x), (low_y, high_y) = rectangle
        square.AutoCropOutputOff()
        square.SetOutputOrigin(low_x, low_y, 0.0)
        square.SetOutputExtent(
            0,
            _last_index(low_x, high_x, spacing),
            0,
            _last_index(low_y, high_y, spacing),
            0,
            0,
        )

    square.Update()
    return square


def location_of(image, axes: np.ndarray) -> Location:
    """The two direction cosines and the corner position DICOM asks for.

    Of the row written first, which is the top of the view: the cut is turned
    over on the way out, so the column direction runs against the output's y
    and the first pixel is the one at the far end of it.
    """
    rows = image.GetDimensions()[1]
    spacing = image.GetSpacing()
    origin = np.array(image.GetOrigin(), dtype=np.float64)
    first = origin + np.array([0.0, (rows - 1) * spacing[1], 0.0])

    return Location(
        # DICOM names the row direction first -- the way an index along a row
        # advances, which is the output's own x.
        orientation=tuple(axes[:3, 0]) + tuple(-axes[:3, 1]),
        position=tuple(axes[:3, :3] @ first + axes[:3, 3]),
    )


def plane_from_reslice(reslice, rectangle=None) -> Plane:
    """One reslice's output as pixels, plus where they sit in the patient.

    Flipped to put the top of the view first, which is the row a viewer draws
    at the top: a cut written in the output's own order arrives upside down
    beside the rendered capture of the same view.  What that costs is a column
    direction running the other way, which ``location_of`` says.

    ``rectangle`` is passed on to ``square_pixels``, which is what crops the
    cut to the part of it the view is showing.
    """
    square = square_pixels(reslice, rectangle)
    image = square.GetOutput()
    spacing = image.GetSpacing()

    return Plane(
        scalars=np.flipud(scalars_2d(image)),
        # DICOM measures a pixel down a column first, then along a row.
        pixel_spacing=(spacing[1], spacing[0]),
        thickness=spacing[2],
        location=location_of(image, reslice_axes(square)),
    )


# DICOM names the two ends of the patient's long axis after the head and the
# feet; the app's own axis codes name them superior and inferior.
_PATIENT_LETTERS = {"S": "H", "I": "F"}


def patient_orientation(location: Location) -> tuple[str, str]:
    """Which way a cut's rows and columns run, in anatomical letters.

    The same two vectors ``ImageOrientationPatient`` carries, named rather than
    measured: Patient Orientation is what a viewer letters the edges of an
    image from, and it is the one thing Secondary Capture asks for that
    highdicom will not build an instance without.

    The letters come from ``axcode_from_direction``, so an oblique cut is
    described by the axes it most nearly runs along rather than refused, and
    the reading agrees with the orientation the metadata sheet shows.
    """
    cosines = np.asarray(location.orientation, dtype=np.float64).reshape(2, 3).T
    normal = np.cross(cosines[:, 0], cosines[:, 1])
    axcodes = axcode_from_direction(np.column_stack([cosines, normal]))
    return tuple(_PATIENT_LETTERS.get(code, code) for code in axcodes[:2])
