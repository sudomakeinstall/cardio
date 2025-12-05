import logging

import itk
import numpy as np
import pydantic as pc
import vtk

from .object import Object
from .orientation import (
    EulerAxis,
    axcode_transform_matrix,
    create_vtk_reslice_matrix,
    euler_angle_to_rotation_matrix,
    reset_direction,
)
from .volume_property_presets import load_volume_property_preset


class Volume(Object):
    """Volume object with transfer functions and clipping support."""

    pattern: str = pc.Field(
        default="{frame}.nii.gz",
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
            image = reset_direction(image)
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
            reslice.SetBackgroundLevel(-1000.0)

            # Create image actor
            actor = vtk.vtkImageActor()
            actor.GetMapper().SetInputConnection(reslice.GetOutputPort())
            actor.SetVisibility(False)

            mpr_actors[orientation] = {"reslice": reslice, "actor": actor}

        # Store actors for this frame
        if frame not in self._mpr_actors:
            self._mpr_actors[frame] = {}
        self._mpr_actors[frame] = mpr_actors

        # Set up initial reslice matrices for center slices
        self._setup_center_slices(image_data, frame)

        return mpr_actors

    def _setup_center_slices(self, image_data, frame: int):
        """Set up reslice matrices to show center slices using axcode-based coordinate systems."""
        center = image_data.GetCenter()
        actors = self._mpr_actors[frame]

        # Get coordinate system transformations for each MPR view
        transforms = self._get_mpr_coordinate_systems()

        # Create reslice matrices directly from transforms
        origin = [center[0], center[1], center[2]]

        for orientation in ["axial", "sagittal", "coronal"]:
            mat = create_vtk_reslice_matrix(transforms[orientation], origin)
            actors[orientation]["reslice"].SetResliceAxes(mat)

    @property
    def mpr_actors(self) -> dict[str, list[vtk.vtkImageActor]]:
        """Get MPR actors for all frames."""
        return self._mpr_actors

    def get_mpr_actors_for_frame(self, frame: int) -> dict:
        """Get MPR actors for a specific frame."""
        if frame not in self._mpr_actors:
            return self.create_mpr_actors(frame)
        return self._mpr_actors[frame]

    def _get_mpr_coordinate_systems(self):
        """Get coordinate system transformation matrices for MPR views."""
        view_axcodes = {
            "axial": "LAS",  # Left-Anterior-Superior
            "sagittal": "ASL",  # Anterior-Superior-Left
            "coronal": "LSA",  # Left-Superior-Anterior
        }

        transforms = {}
        for view, target_axcode in view_axcodes.items():
            transforms[view] = axcode_transform_matrix("LPS", target_axcode)

        return transforms

    def get_physical_bounds(
        self, frame: int = 0
    ) -> tuple[float, float, float, float, float, float]:
        """Get physical coordinate bounds for the volume.

        Returns:
            (x_min, x_max, y_min, y_max, z_min, z_max) in LAS coordinate system
        """
        if not self._actors:
            raise RuntimeError(f"No actors configured for volume '{self.label}'")
        if frame >= len(self._actors):
            raise IndexError(
                f"Frame {frame} out of range for volume '{self.label}' (max: {len(self._actors) - 1})"
            )

        volume_actor = self._actors[frame]
        image_data = volume_actor.GetMapper().GetInput()

        # Get VTK image metadata
        origin = np.array(image_data.GetOrigin())
        spacing = np.array(image_data.GetSpacing())
        dimensions = np.array(image_data.GetDimensions())
        direction_matrix = np.array(
            [
                [image_data.GetDirectionMatrix().GetElement(i, j) for j in range(3)]
                for i in range(3)
            ]
        )

        # Calculate antiorigin using direction matrix
        antiorigin = origin + direction_matrix @ (spacing * (dimensions - 1))

        # Transform both corners from LPS to LAS
        transform = axcode_transform_matrix("LPS", "LAS")
        origin_las = origin @ transform.T
        antiorigin_las = antiorigin @ transform.T

        # Interleave coordinates directly without min/max
        bounds = (
            origin_las[0],
            antiorigin_las[0],  # x bounds
            origin_las[1],
            antiorigin_las[1],  # y bounds
            origin_las[2],
            antiorigin_las[2],  # z bounds
        )

        return bounds

    def update_slice_positions(
        self,
        frame: int,
        axial_pos: float,
        sagittal_pos: float,
        coronal_pos: float,
        rotation_sequence: list = None,
        rotation_angles: dict = None,
    ):
        """Update slice positions for MPR views with optional rotation.

        Args:
            frame: Frame index
            axial_pos: Physical position along Z axis (LAS Superior)
            sagittal_pos: Physical position along X axis (LAS Left)
            coronal_pos: Physical position along Y axis (LAS Anterior)
        """
        if frame not in self._mpr_actors:
            return

        volume_actor = self._actors[frame]
        image_data = volume_actor.GetMapper().GetInput()
        las_bounds = self.get_physical_bounds(frame)

        actors = self._mpr_actors[frame]

        # Clamp positions to volume bounds (LAS coordinates from sliders)
        axial_pos = max(
            min(las_bounds[4], las_bounds[5]),
            min(max(las_bounds[4], las_bounds[5]), axial_pos),
        )
        sagittal_pos = max(
            min(las_bounds[0], las_bounds[1]),
            min(max(las_bounds[0], las_bounds[1]), sagittal_pos),
        )
        coronal_pos = max(
            min(las_bounds[2], las_bounds[3]),
            min(max(las_bounds[2], las_bounds[3]), coronal_pos),
        )

        # Convert coronal position from LAS to LPS (negate Y axis)
        coronal_pos_lps = -coronal_pos

        # Get coordinate system transformations for each MPR view
        transforms = self._get_mpr_coordinate_systems()

        center = image_data.GetCenter()

        # Step 1: Apply translation to determine slice origins in physical space (LPS)
        axial_origin = [center[0], center[1], axial_pos]
        sagittal_origin = [sagittal_pos, center[1], center[2]]
        coronal_origin = [center[0], coronal_pos_lps, center[2]]

        # Step 2: Apply cumulative rotation around the translated origins
        if rotation_sequence and rotation_angles:
            cumulative_rotation = np.eye(3)
            for i, rotation in enumerate(rotation_sequence):
                angle = rotation_angles.get(i, 0)
                rotation_matrix = euler_angle_to_rotation_matrix(
                    EulerAxis(rotation["axis"]), angle
                )
                cumulative_rotation = cumulative_rotation @ rotation_matrix

            # Apply rotation to base transforms
            axial_transform = cumulative_rotation @ transforms["axial"]
            sagittal_transform = cumulative_rotation @ transforms["sagittal"]
            coronal_transform = cumulative_rotation @ transforms["coronal"]
        else:
            # Use base transforms without rotation
            axial_transform = transforms["axial"]
            sagittal_transform = transforms["sagittal"]
            coronal_transform = transforms["coronal"]

        # Update slices with translated origins and rotated transforms
        axial_matrix = create_vtk_reslice_matrix(axial_transform, axial_origin)
        actors["axial"]["reslice"].SetResliceAxes(axial_matrix)

        sagittal_matrix = create_vtk_reslice_matrix(sagittal_transform, sagittal_origin)
        actors["sagittal"]["reslice"].SetResliceAxes(sagittal_matrix)

        coronal_matrix = create_vtk_reslice_matrix(coronal_transform, coronal_origin)
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
