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
    _crosshair_actors: dict[str, dict[str, vtk.vtkActor]] = pc.PrivateAttr(
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

    def create_crosshair_actors(self, colors: dict, line_width: float = 1.5) -> dict:
        """Create 2D crosshair overlay actors for each MPR view.

        Uses screen-space 2D actors that always appear centered in the view.

        Args:
            colors: Dict mapping view names to RGB tuples
            line_width: Width of the crosshair lines

        Returns:
            Dict mapping view names to dicts with line actors
        """
        crosshairs = {}

        for view_name in ["axial", "sagittal", "coronal"]:
            view_crosshairs = {}

            for line_name in ["line1", "line2"]:
                # Create 2D line using normalized viewport coordinates
                points = vtk.vtkPoints()
                lines = vtk.vtkCellArray()

                # Placeholder points - will set based on line orientation
                if line_name == "line1":
                    # Vertical line (one of the other planes)
                    points.InsertNextPoint(0.5, 0.0, 0.0)
                    points.InsertNextPoint(0.5, 1.0, 0.0)
                else:
                    # Horizontal line (other plane)
                    points.InsertNextPoint(0.0, 0.5, 0.0)
                    points.InsertNextPoint(1.0, 0.5, 0.0)

                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, 0)
                line.GetPointIds().SetId(1, 1)
                lines.InsertNextCell(line)

                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                polydata.SetLines(lines)

                # Use coordinate transform to map normalized coords to viewport
                coord = vtk.vtkCoordinate()
                coord.SetCoordinateSystemToNormalizedViewport()

                mapper = vtk.vtkPolyDataMapper2D()
                mapper.SetInputData(polydata)
                mapper.SetTransformCoordinate(coord)

                actor = vtk.vtkActor2D()
                actor.SetMapper(mapper)
                actor.GetProperty().SetLineWidth(line_width)
                actor.SetVisibility(False)

                view_crosshairs[line_name] = {
                    "polydata": polydata,
                    "actor": actor,
                }

            # Set colors based on which planes the lines represent
            if view_name == "axial":
                view_crosshairs["line1"]["actor"].GetProperty().SetColor(
                    *colors.get("sagittal", (1, 0, 0))
                )
                view_crosshairs["line2"]["actor"].GetProperty().SetColor(
                    *colors.get("coronal", (0, 1, 0))
                )
            elif view_name == "sagittal":
                view_crosshairs["line1"]["actor"].GetProperty().SetColor(
                    *colors.get("coronal", (0, 1, 0))
                )
                view_crosshairs["line2"]["actor"].GetProperty().SetColor(
                    *colors.get("axial", (0, 0, 1))
                )
            else:  # coronal
                view_crosshairs["line1"]["actor"].GetProperty().SetColor(
                    *colors.get("sagittal", (1, 0, 0))
                )
                view_crosshairs["line2"]["actor"].GetProperty().SetColor(
                    *colors.get("axial", (0, 0, 1))
                )

            crosshairs[view_name] = view_crosshairs

        self._crosshair_actors = crosshairs
        return crosshairs

    @property
    def crosshair_actors(self) -> dict:
        """Get crosshair actors."""
        return self._crosshair_actors

    def set_crosshairs_visible(self, visible: bool):
        """Set visibility of all crosshair actors."""
        for view_crosshairs in self._crosshair_actors.values():
            for line_data in view_crosshairs.values():
                line_data["actor"].SetVisibility(visible)

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

    def _build_cumulative_rotation(
        self, rotation_sequence: list, rotation_angles: dict, angle_units=None
    ) -> np.ndarray:
        """Build cumulative rotation matrix from sequence of rotations."""
        from .orientation import AngleUnits

        if angle_units is None:
            angle_units = AngleUnits.DEGREES

        cumulative_rotation = np.eye(3)
        if rotation_sequence and rotation_angles:
            for i, rotation in enumerate(rotation_sequence):
                angle = rotation_angles.get(i, 0)
                rotation_matrix = euler_angle_to_rotation_matrix(
                    EulerAxis(rotation["axis"]), angle, angle_units
                )
                cumulative_rotation = cumulative_rotation @ rotation_matrix
        return cumulative_rotation

    def get_scroll_vector(
        self,
        view_name: str,
        rotation_sequence: list = None,
        rotation_angles: dict = None,
        angle_units=None,
    ) -> np.ndarray:
        """Get the current normal vector for a view after rotation.

        Args:
            view_name: One of "axial", "sagittal", "coronal"
            rotation_sequence: List of rotation definitions
            rotation_angles: Dict mapping rotation index to angle
            angle_units: AngleUnits enum (degrees or radians)

        Returns:
            3D unit vector representing the scroll direction for this view
        """
        base_normals = {
            "axial": np.array([0.0, 0.0, 1.0]),
            "sagittal": np.array([1.0, 0.0, 0.0]),
            "coronal": np.array([0.0, 1.0, 0.0]),
        }

        if view_name not in base_normals:
            return np.array([0.0, 0.0, 1.0])

        cumulative_rotation = self._build_cumulative_rotation(
            rotation_sequence, rotation_angles, angle_units
        )
        return cumulative_rotation @ base_normals[view_name]

    def update_slice_positions(
        self,
        frame: int,
        origin: list,
        rotation_sequence: list = None,
        rotation_angles: dict = None,
        angle_units=None,
    ):
        """Update slice positions for MPR views with optional rotation.

        Args:
            frame: Frame index
            origin: [x, y, z] position in LPS coordinates (shared by all views)
            rotation_sequence: List of rotation definitions
            rotation_angles: Dict mapping rotation index to angle
            angle_units: AngleUnits enum (degrees or radians), defaults to DEGREES
        """
        from .orientation import AngleUnits

        if angle_units is None:
            angle_units = AngleUnits.DEGREES
        if frame not in self._mpr_actors:
            return

        actors = self._mpr_actors[frame]

        # Get coordinate system transformations for each MPR view
        transforms = self._get_mpr_coordinate_systems()

        # Build cumulative rotation matrix
        cumulative_rotation = self._build_cumulative_rotation(
            rotation_sequence, rotation_angles, angle_units
        )

        # Apply rotation to base transforms
        axial_transform = cumulative_rotation @ transforms["axial"]
        sagittal_transform = cumulative_rotation @ transforms["sagittal"]
        coronal_transform = cumulative_rotation @ transforms["coronal"]

        # All views share the same origin
        axial_matrix = create_vtk_reslice_matrix(axial_transform, origin)
        actors["axial"]["reslice"].SetResliceAxes(axial_matrix)

        sagittal_matrix = create_vtk_reslice_matrix(sagittal_transform, origin)
        actors["sagittal"]["reslice"].SetResliceAxes(sagittal_matrix)

        coronal_matrix = create_vtk_reslice_matrix(coronal_transform, origin)
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
