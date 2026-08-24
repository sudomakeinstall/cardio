"""The three MPR reslice pipelines belonging to a single frame."""

# System
import typing as ty

# Third Party
import numpy as np
import vtk

# Internal
from .orientation import (
    AngleUnits,
    axcode_transform_matrix,
    create_vtk_reslice_matrix,
    cumulative_rotation_matrix,
)

VIEW_AXCODES = {
    "axial": "LAS",  # Left-Anterior-Superior
    "sagittal": "ASL",  # Anterior-Superior-Left
    "coronal": "LSA",  # Left-Superior-Anterior
}

VIEWS = tuple(VIEW_AXCODES)

VIEW_TRANSFORMS = {
    view: axcode_transform_matrix("LPS", axcode)
    for view, axcode in VIEW_AXCODES.items()
}

OutputFilter = ty.Callable[
    [vtk.vtkImageReslice], tuple[vtk.vtkAlgorithm, dict[str, ty.Any]]
]


class ResliceSet:
    """One frame's axial, sagittal and coronal reslice pipelines.

    All three views share an origin and a rotation; ``set_pose`` is the only way
    to move them, so the views cannot drift out of step.

    Volumes and segmentations differ only in what sits between the reslice and
    the actor -- a segmentation maps labels through a lookup table -- which is
    what ``output_filter`` supplies.
    """

    def __init__(
        self,
        image_data,
        interpolation: str,
        background_level: float,
        output_filter: OutputFilter | None = None,
    ):
        self.image_data = image_data
        self.views: dict[str, dict[str, ty.Any]] = {}

        for view in VIEWS:
            reslice = vtk.vtkImageReslice()
            reslice.SetInputData(image_data)
            reslice.SetOutputDimensionality(2)
            match interpolation:
                case "linear":
                    reslice.SetInterpolationModeToLinear()
                case "nearest":
                    reslice.SetInterpolationModeToNearestNeighbor()
                case _:
                    raise ValueError(f"Unknown interpolation mode: {interpolation}")
            reslice.SetBackgroundLevel(background_level)
            reslice.AutoCropOutputOn()

            if output_filter is None:
                source, extra = reslice, {}
            else:
                source, extra = output_filter(reslice)

            actor = vtk.vtkImageActor()
            actor.GetMapper().SetInputConnection(source.GetOutputPort())
            actor.SetVisibility(False)

            self.views[view] = {"reslice": reslice, "actor": actor} | extra

        self.center_on_image()

    def __getitem__(self, view: str) -> dict[str, ty.Any]:
        return self.views[view]

    def __contains__(self, view: str) -> bool:
        return view in self.views

    def values(self):
        return self.views.values()

    def set_pose(self, origin: list[float], rotation: np.ndarray | None = None):
        """Aim all three views at ``origin``, rotated by ``rotation``.

        ``origin`` and ``rotation`` are both in LPS (ITK) coordinates, which is
        the only convention VTK is ever given.
        """
        if rotation is None:
            rotation = np.eye(3)

        for view, parts in self.views.items():
            matrix = create_vtk_reslice_matrix(rotation @ VIEW_TRANSFORMS[view], origin)
            parts["reslice"].SetResliceAxes(matrix)
            parts["reslice"].Update()

    def set_pose_from_sequence(
        self,
        origin: list[float],
        rotation_sequence=None,
        rotation_angles=None,
        angle_units: AngleUnits | None = None,
    ):
        """``set_pose`` with the rotation composed from a rotation sequence."""
        rotation = cumulative_rotation_matrix(
            rotation_sequence, rotation_angles, angle_units or AngleUnits.DEGREES
        )
        self.set_pose(origin, rotation)

    def center_on_image(self):
        self.set_pose(list(self.image_data.GetCenter()))

    def set_window_level(self, window: float, level: float):
        for parts in self.views.values():
            image_property = parts["actor"].GetProperty()
            image_property.SetColorWindow(window)
            image_property.SetColorLevel(level)
