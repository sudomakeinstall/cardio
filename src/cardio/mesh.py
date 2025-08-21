# System
import enum
import logging

# Third Party
import numpy as np
import pydantic as pc
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkColorTransferFunction,
)
from vtkmodules.vtkFiltersModeling import vtkLoopSubdivisionFilter
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkFiltersVerdict import vtkMeshQuality

# Internal
from . import Object
from .property_config import Representation, PropertyConfig


class SurfaceType(enum.Enum):
    """Surface coloring types for mesh rendering."""

    SOLID = "solid"
    SQUEEZ = "squeez"


def apply_elementwise(
    array1: vtkFloatArray, array2: vtkFloatArray, func, name: str = "Result"
) -> vtkFloatArray:
    if array1.GetNumberOfTuples() != array2.GetNumberOfTuples():
        raise ValueError(
            f"Array lengths must match: {array1.GetNumberOfTuples()} != {array2.GetNumberOfTuples()}"
        )

    result_array = vtkFloatArray()
    result_array.SetName(name)
    result_array.SetNumberOfComponents(1)

    for i in range(array1.GetNumberOfTuples()):
        val1 = array1.GetValue(i)
        val2 = array2.GetValue(i)
        result = func(val1, val2)
        result_array.InsertNextValue(result)

    return result_array


def calculate_squeez(current_area: float, ref_area: float) -> float:
    if ref_area > 0:
        return (current_area / ref_area) ** 0.5
    return 1.0


class Mesh(Object):
    """Mesh object with subdivision support."""

    actors: list[vtkActor] = pc.Field(default_factory=list, exclude=True)
    property_config: PropertyConfig = pc.Field(default=None, exclude=True)
    loop_subdivision_iterations: int = pc.Field(ge=0, le=5, default=0)
    surface_type: SurfaceType = pc.Field(default=SurfaceType.SOLID)
    ctf_min: float = pc.Field(ge=0.0, default=0.7)
    ctf_max: float = pc.Field(ge=0.0, default=1.3)

    def __init__(self, cfg: str, renderer: vtkRenderer):
        # Validate loop subdivision iterations
        iterations = cfg.get("loop_subdivision_iterations", 0)

        # Parse surface_type from config
        surface_type_str = cfg.get("surface_type", "solid")
        try:
            surface_type = SurfaceType(surface_type_str)
        except ValueError:
            logging.warning(
                f"Invalid surface_type '{surface_type_str}' for mesh {cfg['label']}, using default 'solid'"
            )
            surface_type = SurfaceType.SOLID

        super().__init__(
            label=cfg["label"],
            directory=cfg["directory"],
            pattern=cfg.get("pattern", "${frame}.obj"),
            visible=cfg["visible"],
            renderer=renderer,
            clipping_enabled=cfg.get("clipping_enabled", True),
            loop_subdivision_iterations=iterations,
            surface_type=surface_type,
            property_config=PropertyConfig.model_validate(cfg["property"]),
        )

        # Pass 1: Load all frames and determine topology consistency
        frame_data = []
        for frame, path in enumerate(self.path_list):
            logging.info(f"{self.label}: Loading frame {frame}.")
            reader = vtkOBJReader()
            reader.SetFileName(path)
            reader.Update()

            if self.loop_subdivision_iterations > 0:
                subdivision_filter = vtkLoopSubdivisionFilter()
                subdivision_filter.SetInputConnection(reader.GetOutputPort())
                subdivision_filter.SetNumberOfSubdivisions(
                    self.loop_subdivision_iterations
                )
                subdivision_filter.Update()
                polydata = subdivision_filter.GetOutput()
            else:
                polydata = reader.GetOutput()

            if self.property_config.representation == Representation.Surface:
                polydata = self.calculate_mesh_areas(polydata)

            frame_data.append(polydata)

        consistent_topology = self._should_calculate_squeez(frame_data)

        # Pass 2: Create actors with appropriate coloring
        for frame, polydata in enumerate(frame_data):
            mapper = self._setup_coloring(
                polydata, frame_data, frame, consistent_topology
            )

            actor = vtkActor()
            actor.SetMapper(mapper)

            self.actors.append(actor)

    def _setup_coloring(self, polydata, frame_data, frame, consistent_topology):
        """Setup mesh coloring based on surface_type with smart fallback."""
        if self.surface_type == SurfaceType.SQUEEZ:
            if self._can_use_squeez_coloring(consistent_topology):
                return self._setup_squeez_coloring(polydata, frame_data, frame)
            else:
                self._log_squeez_fallback()
                return self._setup_solid_coloring(polydata)
        else:  # SurfaceType.SOLID
            return self._setup_solid_coloring(polydata)

    def _can_use_squeez_coloring(self, consistent_topology):
        """Check if SQUEEZ coloring is possible."""
        return (
            self.property_config.representation == Representation.Surface
            and consistent_topology
        )

    def _log_squeez_fallback(self):
        """Log warning when falling back from SQUEEZ to solid coloring."""
        if self.property_config.representation != Representation.Surface:
            logging.warning(
                f"SQUEEZ coloring requested for {self.label} but representation is not Surface, falling back to solid coloring"
            )
        else:
            logging.warning(
                f"SQUEEZ coloring requested for {self.label} but topology inconsistent across frames, falling back to solid coloring"
            )

    def _setup_solid_coloring(self, polydata):
        """Setup solid coloring using VTK property colors."""
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()
        return mapper

    def _setup_squeez_coloring(self, polydata, frame_data, frame):
        """Setup SQUEEZ coloring with scalar data."""
        ref_quality_array = frame_data[0].GetCellData().GetArray("Area")
        if frame == 0:
            ratio_array = apply_elementwise(
                ref_quality_array, ref_quality_array, lambda x, y: 1.0, "SQUEEZ"
            )
        else:
            current_quality_array = polydata.GetCellData().GetArray("Area")
            ratio_array = apply_elementwise(
                current_quality_array,
                ref_quality_array,
                calculate_squeez,
                "SQUEEZ",
            )
        polydata.GetCellData().AddArray(ratio_array)
        polydata.GetCellData().SetActiveScalars("SQUEEZ")

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper = self.setup_scalar_coloring(mapper)
        return mapper

    def color_transfer_function(self):
        ctf = vtkColorTransferFunction()
        ctf.AddRGBPoint(0.7, 0.0, 0.0, 1.0)
        ctf.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
        ctf.AddRGBPoint(1.3, 1.0, 1.0, 0.0)
        return ctf

    def setup_scalar_coloring(self, mapper):
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarModeToUseCellData()
        mapper.SetLookupTable(self.color_transfer_function())
        mapper.SetScalarRange(self.ctf_min, self.ctf_max)
        mapper.ScalarVisibilityOn()
        mapper.SetInterpolateScalarsBeforeMapping(False)
        return mapper

    def calculate_mesh_areas(self, polydata):
        """Calculate area of each triangle/cell in the mesh using VTK mesh quality."""
        # Ensure we have triangles
        triangle_filter = vtkTriangleFilter()
        triangle_filter.SetInputData(polydata)
        triangle_filter.Update()
        triangulated = triangle_filter.GetOutput()

        # Use VTK mesh quality to calculate areas
        mesh_quality = vtkMeshQuality()
        mesh_quality.SetInputData(triangulated)
        mesh_quality.SetTriangleQualityMeasureToArea()
        mesh_quality.SetQuadQualityMeasureToArea()
        mesh_quality.SaveCellQualityOn()
        mesh_quality.Update()

        quality_output = mesh_quality.GetOutput()
        quality_array = quality_output.GetCellData().GetArray("Quality")
        quality_array.SetName("Area")

        return quality_output

    def _should_calculate_squeez(self, frame_data: list) -> bool:
        if (
            self.property_config.representation != Representation.Surface
            or len(frame_data) <= 1
        ):
            return False

        reference_frame = frame_data[0]
        ref_area_array = reference_frame.GetCellData().GetArray("Area")

        if ref_area_array is None:
            return False

        ref_num_cells = reference_frame.GetNumberOfCells()
        ref_num_points = reference_frame.GetNumberOfPoints()
        ref_num_areas = ref_area_array.GetNumberOfTuples()

        # Check all frames have identical topology
        for polydata in frame_data[1:]:
            area_array = polydata.GetCellData().GetArray("Area")
            if (
                area_array is None
                or polydata.GetNumberOfCells() != ref_num_cells
                or polydata.GetNumberOfPoints() != ref_num_points
                or area_array.GetNumberOfTuples() != ref_num_areas
            ):
                return False

        return True

    def setup_pipeline(self, frame: int):
        for a in self.actors:
            self.renderer.AddActor(a)
            a.SetVisibility(False)
            a.SetProperty(self.property_config.vtk_property)

        # Apply flat shading for SQUEEZ coloring
        if self._is_using_squeez_coloring():
            for actor in self.actors:
                actor.GetProperty().SetInterpolationToFlat()

        if self.visible:
            self.actors[frame].SetVisibility(True)
        self.renderer.ResetCamera()

    def _is_using_squeez_coloring(self):
        """Check if SQUEEZ coloring is actually being used (not fell back to solid)."""
        if self.surface_type != SurfaceType.SQUEEZ:
            return False

        # Check if any actor has SQUEEZ scalar data
        for actor in self.actors:
            mapper = actor.GetMapper()
            if mapper.GetInput():
                squeez_array = mapper.GetInput().GetCellData().GetArray("SQUEEZ")
                if squeez_array is not None:
                    return True
        return False
