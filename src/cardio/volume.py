import logging

import itk
import numpy as np
import pydantic as pc
import vtk

from .object import Object
from .utils import InterpolatorType, reset_direction
from .volume_property_presets import load_volume_property_preset


def create_reslice_matrix(normal, up, origin):
    """Create a 4x4 reslice matrix from normal vector, up vector, and origin"""
    normal = normal / np.linalg.norm(normal)
    up = up / np.linalg.norm(up)
    right = np.cross(normal, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, normal)
    matrix = vtk.vtkMatrix4x4()
    for i in range(3):
        matrix.SetElement(i, 0, right[i])
        matrix.SetElement(i, 1, up[i])
        matrix.SetElement(i, 2, normal[i])
        matrix.SetElement(i, 3, origin[i])
    matrix.SetElement(3, 3, 1.0)
    return matrix


class Volume(Object):
    """Volume object with transfer functions and clipping support."""

    pattern: str = pc.Field(
        default="${frame}.nii.gz",
        description="Filename pattern with $frame placeholder",
    )
    transfer_function_preset: str = pc.Field(
        default="bone", description="Transfer function preset key"
    )
    _actors: list[vtk.vtkVolume] = pc.PrivateAttr(default_factory=list)
    _mpr_actors: dict[str, list[vtk.vtkImageActor]] = pc.PrivateAttr(
        default_factory=dict
    )

    @pc.model_validator(mode="after")
    def initialize_volume(self):
        """Generate VTK volume actors for all frames."""
        for frame, path in enumerate(self.path_list):
            logging.info(f"{self.label}: Loading frame {frame}.")

            image = itk.imread(path)
            image = reset_direction(image, InterpolatorType.LINEAR)
            image = itk.vtk_image_from_image(image)

            mapper = vtk.vtkGPUVolumeRayCastMapper()
            mapper.SetInputData(image)

            actor = vtk.vtkVolume()
            actor.SetMapper(mapper)

            self._actors.append(actor)

        return self

    @property
    def actors(self) -> list[vtk.vtkVolume]:
        return self._actors

    @property
    def preset(self):
        """Load preset based on transfer_function_preset."""
        return load_volume_property_preset(self.transfer_function_preset)

    def configure_actors(self):
        """Configure volume properties without adding to renderer."""
        for volume in self._actors:
            volume.SetVisibility(False)
            volume.SetProperty(self.preset.vtk_property)

    def apply_preset_to_actors(self):
        """Apply the current preset to all actors."""
        for actor in self._actors:
            actor.SetProperty(self.preset.vtk_property)

    def create_mpr_actors(self, frame: int = 0):
        """Create MPR (reslice) actors for axial, sagittal, and coronal views."""
        if frame >= len(self._actors):
            frame = 0

        # Get the image data from the volume actor
        volume_actor = self._actors[frame]
        image_data = volume_actor.GetMapper().GetInput()

        # Create reslice actors for each orientation
        mpr_actors = {}

        for orientation in ["axial", "sagittal", "coronal"]:
            # Create reslice filter
            reslice = vtk.vtkImageReslice()
            reslice.SetInputData(image_data)
            reslice.SetOutputDimensionality(2)
            reslice.SetInterpolationModeToLinear()

            # Create image actor
            actor = vtk.vtkImageActor()
            actor.GetMapper().SetInputConnection(reslice.GetOutputPort())
            actor.SetVisibility(False)  # Start hidden

            mpr_actors[orientation] = {"reslice": reslice, "actor": actor}

        # Store actors for this frame
        if frame not in self._mpr_actors:
            self._mpr_actors[frame] = {}
        self._mpr_actors[frame] = mpr_actors

        # Set up initial reslice matrices for center slices
        self._setup_center_slices(image_data, frame)

        return mpr_actors

    def _setup_center_slices(self, image_data, frame: int):
        """Set up reslice matrices to show center slices using LAS coordinate system."""
        center = image_data.GetCenter()

        actors = self._mpr_actors[frame]

        # Base LAS vectors (Left-Anterior-Superior coordinate system)
        base_axial_normal = np.array([0.0, 0.0, 1.0])  # Z axis (Superior)
        base_axial_up = np.array([0.0, -1.0, 0.0])  # -Y axis (Anterior)

        base_sagittal_normal = np.array([1.0, 0.0, 0.0])  # X axis (Left)
        base_sagittal_up = np.array([0.0, 0.0, 1.0])  # Z axis (Superior)

        base_coronal_normal = np.array([0.0, 1.0, 0.0])  # Y axis (Posterior in data)
        base_coronal_up = np.array([0.0, 0.0, 1.0])  # Z axis (Superior)

        # Create reslice matrices with proper LAS vectors
        axial_origin = [center[0], center[1], center[2]]
        axial_matrix = create_reslice_matrix(
            base_axial_normal, base_axial_up, axial_origin
        )
        actors["axial"]["reslice"].SetResliceAxes(axial_matrix)

        sagittal_origin = [center[0], center[1], center[2]]
        sagittal_matrix = create_reslice_matrix(
            base_sagittal_normal, base_sagittal_up, sagittal_origin
        )
        actors["sagittal"]["reslice"].SetResliceAxes(sagittal_matrix)

        # Coronal view: LPS->LAS Y coordinate conversion
        coronal_origin = [center[0], center[1], center[2]]
        coronal_matrix = create_reslice_matrix(
            base_coronal_normal, base_coronal_up, coronal_origin
        )
        actors["coronal"]["reslice"].SetResliceAxes(coronal_matrix)

    @property
    def mpr_actors(self) -> dict[str, list[vtk.vtkImageActor]]:
        """Get MPR actors for all frames."""
        return self._mpr_actors

    def get_mpr_actors_for_frame(self, frame: int) -> dict:
        """Get MPR actors for a specific frame."""
        if frame not in self._mpr_actors:
            return self.create_mpr_actors(frame)
        return self._mpr_actors[frame]

    def update_slice_positions(
        self, frame: int, axial_frac: float, sagittal_frac: float, coronal_frac: float
    ):
        """Update slice positions for MPR views."""
        if frame not in self._mpr_actors:
            return

        volume_actor = self._actors[frame]
        image_data = volume_actor.GetMapper().GetInput()
        bounds = image_data.GetBounds()

        actors = self._mpr_actors[frame]

        # Calculate slice positions from fractions
        axial_pos = bounds[4] + axial_frac * (bounds[5] - bounds[4])  # Z bounds
        sagittal_pos = bounds[0] + sagittal_frac * (bounds[1] - bounds[0])  # X bounds
        # Coronal: LPS->LAS Y coordinate conversion (flip direction)
        coronal_pos = bounds[3] - coronal_frac * (
            bounds[3] - bounds[2]
        )  # Flipped Y bounds

        # Base LAS vectors
        base_axial_normal = np.array([0.0, 0.0, 1.0])
        base_axial_up = np.array([0.0, -1.0, 0.0])
        base_sagittal_normal = np.array([1.0, 0.0, 0.0])
        base_sagittal_up = np.array([0.0, 0.0, 1.0])
        base_coronal_normal = np.array([0.0, 1.0, 0.0])
        base_coronal_up = np.array([0.0, 0.0, 1.0])

        center = image_data.GetCenter()

        # Update axial slice
        axial_origin = [center[0], center[1], axial_pos]
        axial_matrix = create_reslice_matrix(
            base_axial_normal, base_axial_up, axial_origin
        )
        actors["axial"]["reslice"].SetResliceAxes(axial_matrix)

        # Update sagittal slice
        sagittal_origin = [sagittal_pos, center[1], center[2]]
        sagittal_matrix = create_reslice_matrix(
            base_sagittal_normal, base_sagittal_up, sagittal_origin
        )
        actors["sagittal"]["reslice"].SetResliceAxes(sagittal_matrix)

        # Update coronal slice
        coronal_origin = [center[0], coronal_pos, center[2]]
        coronal_matrix = create_reslice_matrix(
            base_coronal_normal, base_coronal_up, coronal_origin
        )
        actors["coronal"]["reslice"].SetResliceAxes(coronal_matrix)

    def update_mpr_window_level(self, frame: int, window: float, level: float):
        """Update window/level properties for MPR actors."""
        if frame not in self._mpr_actors:
            return

        actors = self._mpr_actors[frame]

        for orientation in ["axial", "sagittal", "coronal"]:
            if orientation in actors:
                actor = actors[orientation]["actor"]
                property_obj = actor.GetProperty()
                property_obj.SetColorWindow(window)
                property_obj.SetColorLevel(level)
