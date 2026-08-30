import logging
import typing as ty

import itk
import pydantic as pc
import vtk

from .object import Object
from .orientation import AngleUnits, read_frames
from .reslice import ResliceSet
from .volume_property_presets import load_volume_property_preset


class Volume(Object):
    """Volume object with transfer functions and clipping support."""

    kind: ty.ClassVar[str] = "volume"

    pattern: str = pc.Field(
        default="{frame}.nii.gz",
        description="Filename pattern with $frame placeholder",
    )
    transfer_function_preset: str = pc.Field(
        default="bone", description="Transfer function preset key"
    )
    _actors: list[vtk.vtkVolume] = pc.PrivateAttr(default_factory=list)
    _mpr_actors: dict[int, ResliceSet] = pc.PrivateAttr(default_factory=dict)
    _crosshair_actors: dict[str, dict[str, vtk.vtkActor]] = pc.PrivateAttr(
        default_factory=dict
    )

    @pc.model_validator(mode="after")
    def initialize_volume(self):
        """Generate VTK volume actors for all frames."""
        for path in self.path_list:
            for image in read_frames(path):
                logging.info(f"{self.label}: Loading frame {len(self._actors)}.")

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

    def add_to_renderer(self, renderer):
        """Volumes go through AddVolume rather than AddActor."""
        for actor in self._actors:
            renderer.AddVolume(actor)

    def configure_actors(self):
        """Configure volume properties without adding to renderer."""
        for volume in self._actors:
            volume.SetVisibility(False)
            volume.SetProperty(self.preset.vtk_property)

    def mpr_image_data(self, frame: int = 0):
        """The image a frame's reslice pipelines read, wrapping short series."""
        if frame >= len(self._actors):
            frame = 0
        return self._actors[frame].GetMapper().GetInput()

    def create_mpr_actors(self, frame: int = 0) -> ResliceSet:
        """Create the MPR reslice pipelines for a frame, centred on the image."""
        self._mpr_actors[frame] = ResliceSet(
            self.mpr_image_data(frame),
            interpolation="linear",
            background_level=-1000.0,
        )
        return self._mpr_actors[frame]

    def get_mpr_actors_for_frame(self, frame: int) -> ResliceSet:
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

    def update_slice_positions(
        self,
        frame: int,
        origin: list,
        rotation_sequence: list = None,
        rotation_angles: dict = None,
        angle_units: AngleUnits = None,
    ):
        """Aim a frame's MPR views at ``origin`` under the given rotation.

        ``origin`` and the rotation sequence are both in LPS (ITK) coordinates.
        """
        if frame not in self._mpr_actors:
            return

        self._mpr_actors[frame].set_pose_from_sequence(
            origin, rotation_sequence, rotation_angles, angle_units
        )

    def update_mpr_window_level(self, frame: int, window: float, level: float):
        """Update window/level properties for MPR actors."""
        if frame not in self._mpr_actors:
            return

        self._mpr_actors[frame].set_window_level(window, level)
