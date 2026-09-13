"""Where a reslice's pixels sit in the patient.

``ResliceSet.set_pose`` builds the reslice axes in LPS and hands them straight
to VTK, which is the one convention the app ever uses, so the matrix VTK holds
is world-LPS and DICOM's geometry can be read off it rather than reconstructed.
"""

# Third Party
import numpy as np
from vtk.util import numpy_support as vtknp

# Internal
from ..orientation import axcode_from_direction
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

    No vertical flip, unlike a window capture: that one is a framebuffer, which
    VTK fills bottom-up.  Here row zero is simply the low end of the output's y,
    which is what the row direction cosine is then told.
    """
    columns, rows, _ = image.GetDimensions()
    return vtknp.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(rows, columns)


def location_of(image, axes: np.ndarray) -> Location:
    """The two direction cosines and the corner position DICOM asks for."""
    origin = np.array(image.GetOrigin(), dtype=np.float64)
    return Location(
        # DICOM names the column direction first -- the way an index along a
        # row advances, which is the output's own x.
        orientation=tuple(axes[:3, 0]) + tuple(axes[:3, 1]),
        position=tuple(axes[:3, :3] @ origin + axes[:3, 3]),
    )


def plane_from_reslice(reslice) -> Plane:
    """One reslice's output as pixels, plus where they sit in the patient."""
    reslice.Update()
    image = reslice.GetOutput()
    spacing = image.GetSpacing()

    return Plane(
        scalars=scalars_2d(image),
        # DICOM measures a pixel down a column first, then along a row.
        pixel_spacing=(spacing[1], spacing[0]),
        thickness=spacing[2],
        location=location_of(image, reslice_axes(reslice)),
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
