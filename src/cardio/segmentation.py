import logging
import typing as ty

import itk
import numpy as np
import pydantic as pc
import vtk
import vtk.util.numpy_support as vtk_np

from .object import Object
from .orientation import (
    minimal_rotation,
    read_source,
    slerp_rotation_matrices,
)
from .property_config import vtkPropertyConfig
from .reslice import ResliceSet
from .utils import label_color

logger = logging.getLogger(__name__)


_MARKER_ARRAY = "_snap_marker"


def masked_surface(
    mesh: vtk.vtkPolyData, mask: ty.Sequence[bool]
) -> vtk.vtkPolyData | None:
    """Sub-surface of the mesh cells selected by ``mask``."""
    if not any(mask):
        return None

    marker = vtk.vtkIntArray()
    marker.SetName(_MARKER_ARRAY)
    marker.SetNumberOfTuples(len(mask))
    for i, selected in enumerate(mask):
        marker.SetValue(i, 1 if selected else 0)

    masked = vtk.vtkPolyData()
    masked.ShallowCopy(mesh)
    masked.GetCellData().AddArray(marker)
    masked.GetCellData().SetActiveScalars(_MARKER_ARRAY)

    thresh = vtk.vtkThreshold()
    thresh.SetInputData(masked)
    thresh.SetLowerThreshold(1)
    thresh.SetUpperThreshold(1)
    thresh.SetThresholdFunction(thresh.THRESHOLD_BETWEEN)
    thresh.Update()
    if thresh.GetOutput().GetNumberOfCells() == 0:
        return None

    geom = vtk.vtkGeometryFilter()
    geom.SetInputConnection(thresh.GetOutputPort())
    geom.Update()
    return geom.GetOutput()


def masked_centroid(
    mesh: vtk.vtkPolyData, mask: ty.Sequence[bool]
) -> list[float] | None:
    """Center of mass of the mesh cells selected by ``mask``."""
    surface = masked_surface(mesh, mask)
    if surface is None:
        return None

    com = vtk.vtkCenterOfMass()
    com.SetInputData(surface)
    com.SetUseScalarsAsWeights(False)
    com.Update()
    return list(com.GetCenter())


def surface_points(surface: vtk.vtkPolyData) -> np.ndarray:
    """Point coordinates of a surface as an (N, 3) array.

    Widened to float64: VTK stores points as float32, which is not enough
    precision for the rotation and quaternion math downstream.
    """
    return vtk_np.vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float64)


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of a point cloud.

    Returns (centroid, axes, extents). The columns of ``axes`` are the two
    in-plane directions ordered by decreasing spread, then the plane normal.
    ``extents`` holds the corresponding singular values.
    """
    centroid = points.mean(axis=0)
    _, extents, basis = np.linalg.svd(points - centroid, full_matrices=True)
    return centroid, basis.T, extents


# LPS reference directions for the in-plane axes. Anterior reproduces the
# unrotated axial view; superior takes over when the normal is near anterior.
_IN_PLANE_REFERENCE = np.array([0.0, -1.0, 0.0])
_IN_PLANE_FALLBACK = np.array([0.0, 0.0, 1.0])
_REFERENCE_PARALLEL = 0.9


def plane_basis(normal: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
    """View basis for a plane with the given normal.

    The in-plane axes are never taken from PCA: the first two principal axes of
    a near-circular interface are interchangeable and their signs are arbitrary,
    so deriving them from the data makes the in-plane rotation jump between
    frames.

    Without an ``anchor`` the in-plane axes come from a fixed anatomical
    reference. Given an ``anchor`` basis, they are carried onto the new normal
    by the smallest rotation between the two normals, which is continuous in the
    normal and so keeps the view from spinning as the plane moves.

    Columns are (view x, view y, normal). The basis is left-handed to match the
    LPS to LAS axial transform, so the derived rotation stays proper.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    if anchor is not None:
        basis = minimal_rotation(anchor[:, 2], normal) @ anchor
        view_y = basis[:, 1] - (basis[:, 1] @ normal) * normal
        view_y = view_y / np.linalg.norm(view_y)
        return np.column_stack([np.cross(normal, view_y), view_y, normal])

    reference = _IN_PLANE_REFERENCE
    if abs(reference @ normal) > _REFERENCE_PARALLEL:
        reference = _IN_PLANE_FALLBACK

    view_y = reference - (reference @ normal) * normal
    view_y = view_y / np.linalg.norm(view_y)
    view_x = np.cross(normal, view_y)
    return np.column_stack([view_x, view_y, normal])


def set_label_opacity(pipelines, opacity: float):
    """Re-alpha the label colours of every pipeline ``label_color_filter`` built.

    Takes the pipeline dicts rather than a ``ResliceSet``, so the tile grid can
    hand it its own set.
    """
    for parts in pipelines:
        lut = parts["lut"]
        for i in range(1, lut.GetNumberOfTableValues()):
            rgba = list(lut.GetTableValue(i))
            rgba[3] = opacity
            lut.SetTableValue(i, *rgba)
        parts["image_to_colors"].Modified()


# ``plane_basis`` returns a left-handed basis, which has no quaternion and so
# cannot be interpolated directly. Right-multiplying by a fixed improper factor
# lands both endpoints in SO(3); the geodesic there is right-invariant, so
# undoing the factor afterwards recovers the answer the bases themselves imply,
# whichever improper factor is chosen.
_HANDEDNESS_FLIP = np.diag([1.0, -1.0, 1.0])


def interpolate_planes(
    start: tuple[np.ndarray, np.ndarray, float],
    end: tuple[np.ndarray, np.ndarray, float],
    fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Plane ``fraction`` of the way from ``start`` to ``end``.

    Both are ``(centroid, axes, flatness)`` as ``interface_plane`` returns them.
    ``fraction`` is clamped to [0, 1]. Flatness is the worse of the two: a plane
    between the endpoints is no better a description of its interface than they
    are of theirs.
    """
    fraction = min(1.0, max(0.0, float(fraction)))
    start_centroid, start_axes, start_flatness = start
    end_centroid, end_axes, end_flatness = end

    centroid = (1.0 - fraction) * np.asarray(
        start_centroid, dtype=np.float64
    ) + fraction * np.asarray(end_centroid, dtype=np.float64)

    axes = (
        slerp_rotation_matrices(
            start_axes @ _HANDEDNESS_FLIP, end_axes @ _HANDEDNESS_FLIP, fraction
        )
        @ _HANDEDNESS_FLIP
    )

    return centroid, axes, max(float(start_flatness), float(end_flatness))


class Segmentation(Object):
    """Segmentation object with multi-label mesh extraction using SurfaceNets."""

    kind: ty.ClassVar[str] = "segmentation"
    reads_dicom: ty.ClassVar[bool] = True

    pattern: str = pc.Field(
        default="{frame}.nii.gz",
        description="Filename pattern with $frame placeholder",
    )
    _actors: list[vtk.vtkActor] = pc.PrivateAttr(default_factory=list)
    _meshes: list[vtk.vtkPolyData] = pc.PrivateAttr(default_factory=list)
    _label_images: list[vtk.vtkImageData] = pc.PrivateAttr(default_factory=list)
    _mpr_actors: dict[int, ResliceSet] = pc.PrivateAttr(default_factory=dict)
    properties: vtkPropertyConfig = pc.Field(
        default_factory=vtkPropertyConfig, description="Property configuration"
    )
    include_labels: list[int] | None = pc.Field(default=None)
    mpr_overlay: bool = pc.Field(
        default=False,
        description="Whether this segmentation is initially drawn over the MPR views",
    )
    label_properties: dict[int, dict] = pc.Field(default_factory=dict)

    @pc.model_validator(mode="after")
    def initialize_segmentation(self):
        """Generate VTK actors for all frames using SurfaceNets3D."""
        for path in self.path_list:
            frames, source = read_source(path, self.series_uid)
            # A file-per-frame series is one acquisition, so the first path's
            # header describes all of them.
            self._source = self._source or source

            for image in frames:
                logger.info(
                    f"{self.label}: Loading segmentation frame {len(self._actors)}."
                )

                vtk_image = itk.vtk_image_from_image(image)
                self._label_images.append(vtk_image)

                mesh = self._extract_mesh(vtk_image)
                self._meshes.append(mesh)
                self._actors.append(self._create_segmentation_actor(mesh))

        return self

    def _extract_mesh(self, vtk_image) -> vtk.vtkPolyData:
        """Extract a multi-label surface mesh with per-cell label scalars."""
        surface_nets = vtk.vtkSurfaceNets3D()
        surface_nets.SetInputData(vtk_image)
        max_label = int(vtk_image.GetPointData().GetScalars().GetRange()[1])
        surface_nets.GenerateLabels(max_label, 1, max_label)

        if self.include_labels is not None:
            surface_nets.SetOutputStyle(surface_nets.OUTPUT_STYLE_SELECTED)
            surface_nets.InitializeSelectedLabelsList()
            for label in self.include_labels:
                surface_nets.AddSelectedLabel(label)

        surface_nets.Update()
        mesh = surface_nets.GetOutput()

        boundary_labels = mesh.GetCellData().GetArray("BoundaryLabels")

        if boundary_labels:
            scalar_array = vtk.vtkIntArray()
            scalar_array.SetName("Labels")
            scalar_array.SetNumberOfTuples(boundary_labels.GetNumberOfTuples())

            for i in range(boundary_labels.GetNumberOfTuples()):
                label1 = int(boundary_labels.GetComponent(i, 0))
                label2 = int(boundary_labels.GetComponent(i, 1))
                # Prefer the foreground label of the pair over background=0
                cell_label = (
                    max(label1, label2)
                    if max(label1, label2) > 0
                    else min(label1, label2)
                )
                scalar_array.SetValue(i, cell_label)

            mesh.GetCellData().AddArray(scalar_array)
            mesh.GetCellData().SetActiveScalars("Labels")

        return mesh

    @property
    def actors(self) -> list[vtk.vtkActor]:
        return self._actors

    def _create_segmentation_actor(self, mesh):
        """Create a VTK actor with scalar-based coloring for the segmentation mesh."""
        # Create a mapper with scalar coloring
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.SetScalarModeToUseCellData()
        mapper.ScalarVisibilityOn()

        # Create color transfer function for label-based coloring
        color_func = vtk.vtkColorTransferFunction()

        # Get the label range from the scalar array
        scalar_array = mesh.GetCellData().GetArray("Labels")
        if scalar_array:
            scalar_range = scalar_array.GetRange()
            min_label = int(scalar_range[0])
            max_label = int(scalar_range[1])

            # Set colors for each label
            for label in range(min_label, max_label + 1):
                if label == 0:  # Skip background
                    continue

                if label in self.label_properties:
                    props = self.label_properties[label]
                    color_func.AddRGBPoint(
                        label,
                        props.get("r", 1.0),
                        props.get("g", 0.0),
                        props.get("b", 0.0),
                    )
                else:
                    color_func.AddRGBPoint(label, *label_color(label))

            mapper.SetLookupTable(color_func)
            mapper.SetScalarRange(min_label, max_label)

        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def configure_actors(self):
        """Configure actor properties without adding to renderer."""
        for actor in self._actors:
            actor.SetVisibility(False)
            # Apply base property configuration if available
            base_prop = self.properties.vtk_property
            if base_prop:
                # Note: For scalar-colored actors, we preserve the color transfer function
                # by not overriding the mapper's lookup table
                pass

    def mpr_image_data(self, frame: int = 0):
        """The label image a frame's reslice pipelines read, wrapping short series."""
        if frame >= len(self._label_images):
            frame = 0
        return self._label_images[frame]

    def create_mpr_actors(self, frame: int = 0) -> ResliceSet:
        """Create the MPR reslice pipelines for a frame, centred on the image."""
        image_data = self.mpr_image_data(frame)

        self._mpr_actors[frame] = ResliceSet(
            image_data,
            interpolation="nearest",
            background_level=0,
            output_filter=self.label_color_filter(image_data),
        )
        return self._mpr_actors[frame]

    def label_color_filter(self, image_data):
        """The reslice output filter that maps label values to colours.

        Anything reslicing this segmentation needs it, not just the MPR views,
        so it is a method rather than a closure inside ``create_mpr_actors``.
        """

        def color_labels(reslice):
            lut = self._create_label_lookup_table(image_data, opacity=1.0)
            image_to_colors = vtk.vtkImageMapToColors()
            image_to_colors.SetInputConnection(reslice.GetOutputPort())
            image_to_colors.SetLookupTable(lut)
            image_to_colors.SetOutputFormatToRGBA()
            return image_to_colors, {"image_to_colors": image_to_colors, "lut": lut}

        return color_labels

    def _create_label_lookup_table(self, image_data, opacity: float = 1.0):
        """Create lookup table for label-to-color mapping."""
        scalar_range = image_data.GetPointData().GetScalars().GetRange()
        min_label = int(scalar_range[0])
        max_label = int(scalar_range[1])

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(max_label + 1)
        lut.SetRange(min_label, max_label)

        lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)

        for label in range(min_label + 1, max_label + 1):
            if label in self.label_properties:
                r = self.label_properties[label].get("r", 1.0)
                g = self.label_properties[label].get("g", 1.0)
                b = self.label_properties[label].get("b", 1.0)
            else:
                r, g, b = label_color(label)
            lut.SetTableValue(label, r, g, b, opacity)

        lut.Build()
        return lut

    def get_mpr_actors_for_frame(self, frame: int) -> ResliceSet:
        """Get MPR actors for a specific frame."""
        if frame not in self._mpr_actors:
            return self.create_mpr_actors(frame)
        return self._mpr_actors[frame]

    def get_labels(self, frame: int = 0) -> list[int]:
        if frame >= len(self._label_images):
            frame = 0
        image_data = self._label_images[frame]
        max_label = int(image_data.GetPointData().GetScalars().GetRange()[1])
        if max_label < 1:
            return []
        acc = vtk.vtkImageAccumulate()
        acc.SetInputData(image_data)
        acc.SetComponentExtent(0, max_label, 0, 0, 0, 0)
        acc.SetComponentOrigin(0, 0, 0)
        acc.SetComponentSpacing(1, 0, 0)
        acc.Update()
        hist = acc.GetOutput().GetPointData().GetScalars()
        return [i for i in range(1, max_label + 1) if hist.GetTuple1(i) > 0]

    def _frame_mesh(self, frame: int) -> vtk.vtkPolyData | None:
        """Mesh shown at ``frame``, wrapping as the renderer does for short series."""
        if not self._meshes:
            return None
        return self._meshes[frame % len(self._meshes)]

    def _label_mask(self, mesh, labels: list[int]) -> list[bool] | None:
        """Cells whose label is in ``labels``."""
        scalars = mesh.GetCellData().GetArray("Labels")
        if not scalars:
            return None
        label_set = set(labels)
        return [
            int(scalars.GetTuple1(i)) in label_set
            for i in range(scalars.GetNumberOfTuples())
        ]

    def _interface_mask(
        self, mesh, labels_a: list[int], labels_b: list[int]
    ) -> list[bool] | None:
        """Cells separating a label in ``labels_a`` from one in ``labels_b``."""
        boundary = mesh.GetCellData().GetArray("BoundaryLabels")
        if not boundary:
            return None
        set_a, set_b = set(labels_a), set(labels_b)
        mask = []
        for i in range(boundary.GetNumberOfTuples()):
            l0 = int(boundary.GetComponent(i, 0))
            l1 = int(boundary.GetComponent(i, 1))
            mask.append((l0 in set_a and l1 in set_b) or (l0 in set_b and l1 in set_a))
        return mask

    def label_centroid(self, labels: list[int], frame: int = 0) -> list[float] | None:
        mesh = self._frame_mesh(frame)
        if mesh is None or not labels:
            return None
        mask = self._label_mask(mesh, labels)
        if mask is None:
            return None
        return masked_centroid(mesh, mask)

    def interface_centroid(
        self, labels_a: list[int], labels_b: list[int], frame: int = 0
    ) -> list[float] | None:
        mesh = self._frame_mesh(frame)
        if mesh is None or not labels_a or not labels_b:
            return None
        mask = self._interface_mask(mesh, labels_a, labels_b)
        if mask is None:
            return None
        return masked_centroid(mesh, mask)

    def interface_plane(
        self,
        labels_a: list[int],
        labels_b: list[int],
        frame: int = 0,
        anchor: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Dominant plane of the A/B interface.

        Returns (centroid, axes, flatness) where the columns of ``axes`` are the
        two in-plane directions and the plane normal, oriented from group A
        toward group B. ``flatness`` is the out-of-plane spread relative to the
        smaller in-plane spread: 0 is perfectly planar.

        ``anchor`` is a previously computed basis; passing the same one across
        a series of frames keeps the in-plane rotation continuous.
        """
        mesh = self._frame_mesh(frame)
        if mesh is None or not labels_a or not labels_b:
            return None
        mask = self._interface_mask(mesh, labels_a, labels_b)
        if mask is None:
            return None
        surface = masked_surface(mesh, mask)
        if surface is None or surface.GetNumberOfPoints() < 3:
            return None

        centroid, axes, extents = principal_axes(surface_points(surface))
        if extents[1] <= 0:
            return None

        normal = self._normal_sign(axes[:, 2], labels_a, labels_b, frame, anchor)
        return centroid, plane_basis(normal, anchor), float(extents[2] / extents[1])

    def _normal_sign(
        self,
        normal: np.ndarray,
        labels_a: list[int],
        labels_b: list[int],
        frame: int,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        """Resolve the arbitrary sign PCA returns for the plane normal.

        With an ``anchor`` the sign is taken from it, which keeps the normal
        consistent across a series and avoids re-deciding from geometry that may
        be marginal on any given frame. It also skips the two centroid passes
        ``_orient_normal`` needs.

        Without one -- the frame that establishes the anchor -- the sign is
        decided geometrically, from group A toward group B.
        """
        if anchor is not None:
            return -normal if normal @ anchor[:, 2] < 0 else normal
        return self._orient_normal(normal, labels_a, labels_b, frame)

    def _orient_normal(
        self, normal: np.ndarray, labels_a: list[int], labels_b: list[int], frame: int
    ) -> np.ndarray:
        """Point the plane normal from group A toward group B.

        This compares whole-group centroids, so it assumes group B lies on the
        far side of the interface. That does not hold when a group is
        disconnected or wraps around the other, in which case the two centroids
        can fall on the same side and the sign becomes marginal -- which is why
        it is used only to establish the anchor, not on every frame.
        """
        centroid_a = self.label_centroid(labels_a, frame)
        centroid_b = self.label_centroid(labels_b, frame)
        if centroid_a is not None and centroid_b is not None:
            direction = np.asarray(centroid_b) - np.asarray(centroid_a)
            if normal @ direction < 0:
                return -normal
        return normal

    def update_mpr_opacity(self, frame: int, opacity: float):
        """Update opacity for all MPR overlay labels."""
        if frame not in self._mpr_actors:
            return

        set_label_opacity(self._mpr_actors[frame].values(), opacity)
