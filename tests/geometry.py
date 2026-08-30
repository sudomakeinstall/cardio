"""Small geometric probes the tests assert with."""

# Third Party
import numpy as np
import vtk

# Internal
from cardio.segmentation import plane_basis


def tilted_plane(degrees: float) -> np.ndarray:
    """A left-handed plane basis whose normal is tilted from z toward x."""
    theta = np.radians(degrees)
    return plane_basis(np.array([np.sin(theta), 0.0, np.cos(theta)]))


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def axis_angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """As ``angle_between``, but between undirected axes.

    A basis column carries no preferred sign, so a flip is not a half turn:
    antiparallel counts as zero rather than 180 degrees.
    """
    return float(np.degrees(np.arccos(abs(np.clip(u @ v, -1.0, 1.0)))))


def matrix_array(matrix: vtk.vtkMatrix4x4) -> np.ndarray:
    """The 4x4 VTK is actually holding, as a numpy array."""
    return np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
