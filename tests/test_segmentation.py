"""Test Segmentation centroid extraction."""

# Third Party
import numpy as np
import pytest
import vtk

# Internal
from cardio.segmentation import Segmentation, masked_centroid, masked_surface
from tests.phantoms import (
    facet_interface_array,
    facet_interface_centroid,
    l_block_array,
    l_block_centroid,
    write_segmentation,
)

# Label 1 and label 2 are adjacent blocks sharing the plane x = 4.5.
# Label 3 is separated from both by a band of background.
BLOCK_CENTER = 4.5
INTERFACE_X = 4.5


def label_array() -> np.ndarray:
    """Build a (k, j, i) label volume with two adjacent blocks and one isolated."""
    array = np.zeros((16, 16, 16), dtype=np.uint8)
    array[2:8, 2:8, 2:5] = 1
    array[2:8, 2:8, 5:8] = 2
    array[2:8, 2:8, 11:14] = 3
    return array


@pytest.fixture
def segmentation(tmp_path) -> Segmentation:
    """A Segmentation built from a synthetic on-disk label image."""
    return write_segmentation(tmp_path, [label_array()], label="test")


@pytest.fixture
def mesh(segmentation) -> vtk.vtkPolyData:
    return segmentation._meshes[0]


def cell_labels(mesh) -> vtk.vtkDataArray:
    return mesh.GetCellData().GetArray("Labels")


def test_fixture_has_expected_arrays(mesh):
    """The synthetic mesh carries both arrays the centroid functions rely on."""
    assert mesh.GetNumberOfCells() > 0
    assert mesh.GetCellData().GetArray("Labels") is not None
    boundary = mesh.GetCellData().GetArray("BoundaryLabels")
    assert boundary is not None
    assert boundary.GetNumberOfComponents() == 2


def test_masked_centroid_empty_mask(mesh):
    assert masked_centroid(mesh, []) is None


def test_masked_centroid_all_false(mesh):
    mask = [False] * mesh.GetNumberOfCells()
    assert masked_centroid(mesh, mask) is None


def test_masked_centroid_symmetric_subset(mesh):
    """Selecting both adjacent blocks yields their shared symmetric center."""
    scalars = cell_labels(mesh)
    mask = [
        int(scalars.GetTuple1(i)) in {1, 2} for i in range(scalars.GetNumberOfTuples())
    ]
    center = masked_centroid(mesh, mask)
    assert center == pytest.approx([BLOCK_CENTER] * 3, abs=1e-6)


def test_masked_centroid_all_true_covers_the_whole_mesh(mesh):
    """An all-true mask centres on the whole mesh, which here is symmetric.

    The three blocks are placed symmetrically in j and k, so whatever the
    weighting, those two coordinates have only one answer.
    """
    mask = [True] * mesh.GetNumberOfCells()
    center = masked_centroid(mesh, mask)
    assert center[1] == pytest.approx(BLOCK_CENTER, abs=1e-6)
    assert center[2] == pytest.approx(BLOCK_CENTER, abs=1e-6)


# --- weighting ----------------------------------------------------------------

# Sampled ever more finely, an area-weighted centroid closes on the interface's
# own geometry. Averaging the vertices instead does not: SurfaceNets puts about
# as many of them on the flat facet as on the 45-degree one, though the second
# has half again the area, and refining the grid adds vertices to both in the
# same proportion. That answer sits about 1.1 mm out at every resolution below.
FACET_SPACINGS = ((1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0.25, 0.5, 0.25))
FACET_TOLERANCE = 0.25


def facet_errors(tmp_path) -> list[float]:
    """How far the two-facet interface centroid falls from its analytic value."""
    expected = facet_interface_centroid()
    errors = []
    for index, spacing in enumerate(FACET_SPACINGS):
        segmentation = write_segmentation(
            tmp_path,
            [facet_interface_array(spacing)],
            stem=f"facet{index}",
            spacing=spacing,
        )
        center = segmentation.interface_centroid([1], [2])
        errors.append(float(np.linalg.norm(np.array(center) - expected)))
    return errors


def test_interface_centroid_converges_on_the_interface(tmp_path):
    """Refining the grid moves the centroid onto the surface it describes."""
    errors = facet_errors(tmp_path)

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < FACET_TOLERANCE


def test_interface_centroid_beats_an_unweighted_mean(tmp_path):
    """Weighting by area lands nearer the interface than counting vertices.

    The contrast is the whole point of the change, so it is worth stating
    against the answer that used to be given rather than only against the
    tolerance: the vertices are spread by how the surface lies against the
    grid, so their mean is not a centre of anything the interface has.
    """
    expected = facet_interface_centroid()
    spacing = (0.5, 0.5, 0.5)
    segmentation = write_segmentation(
        tmp_path, [facet_interface_array(spacing)], stem="facet", spacing=spacing
    )

    mesh = segmentation._meshes[0]
    surface = masked_surface(mesh, segmentation._interface_mask(mesh, [1], [2]))
    unweighted = vtk.vtkCenterOfMass()
    unweighted.SetInputData(surface)
    unweighted.SetUseScalarsAsWeights(False)
    unweighted.Update()

    center = np.array(segmentation.interface_centroid([1], [2]))

    assert (
        np.linalg.norm(center - expected)
        < np.linalg.norm(np.array(unweighted.GetCenter()) - expected) / 2.0
    )


def test_label_centroid_single_label(segmentation):
    """A single block's centroid lies within that block's extent."""
    center = segmentation.label_centroid([1])
    assert 2.0 <= center[0] <= 5.0
    assert center[1] == pytest.approx(BLOCK_CENTER, abs=1e-6)
    assert center[2] == pytest.approx(BLOCK_CENTER, abs=1e-6)


def test_label_centroid_multiple_labels(segmentation):
    """The two adjacent blocks together are symmetric about their shared center."""
    center = segmentation.label_centroid([1, 2])
    assert center == pytest.approx([BLOCK_CENTER] * 3, abs=1e-6)


def test_label_centroid_isolated_label(segmentation):
    center = segmentation.label_centroid([3])
    assert 11.0 <= center[0] <= 14.0


def test_label_centroid_empty_labels(segmentation):
    assert segmentation.label_centroid([]) is None


def test_label_centroid_absent_label(segmentation):
    assert segmentation.label_centroid([7]) is None


def test_label_centroid_wraps_frame(segmentation):
    """A short series repeats, matching how update_frame indexes actors."""
    assert segmentation.label_centroid([1], 5) == pytest.approx(
        segmentation.label_centroid([1], 0)
    )


def test_interface_centroid_adjacent_labels(segmentation):
    """The interface centroid lands on the plane shared by the two blocks."""
    center = segmentation.interface_centroid([1], [2])
    assert center[0] == pytest.approx(INTERFACE_X, abs=1e-6)
    assert center[1] == pytest.approx(BLOCK_CENTER, abs=1e-6)
    assert center[2] == pytest.approx(BLOCK_CENTER, abs=1e-6)


def test_interface_centroid_group_order_does_not_matter(segmentation):
    forward = segmentation.interface_centroid([1], [2])
    reverse = segmentation.interface_centroid([2], [1])
    assert forward == pytest.approx(reverse, abs=1e-9)


def test_interface_centroid_non_adjacent_labels(segmentation):
    """Labels separated by background have no interface."""
    assert segmentation.interface_centroid([1], [3]) is None
    assert segmentation.interface_centroid([2], [3]) is None


def test_interface_centroid_empty_groups(segmentation):
    assert segmentation.interface_centroid([], [2]) is None
    assert segmentation.interface_centroid([1], []) is None


def test_interface_centroid_wraps_frame(segmentation):
    assert segmentation.interface_centroid([1], [2], 5) == pytest.approx(
        segmentation.interface_centroid([1], [2], 0)
    )


# --- a label is centred on the voxels it is made of ---------------------------


def test_label_centroid_is_the_solid_centroid(tmp_path):
    """An L centres where its mass is, not where its surface averages out."""
    segmentation = write_segmentation(tmp_path, [l_block_array()], stem="ell")

    center = segmentation.label_centroid([1])

    assert center == pytest.approx(l_block_centroid(), abs=1e-9)


def test_label_centroid_reads_the_image_geometry(tmp_path):
    """Spacing, origin and an oblique direction all reach the answer."""
    spacing = (0.8, 1.3, 2.5)
    origin = (-11.0, 4.0, 7.0)
    angle = np.radians(20.0)
    direction = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    segmentation = write_segmentation(
        tmp_path,
        [l_block_array()],
        stem="oblique",
        spacing=spacing,
        origin=origin,
        direction=direction,
    )

    center = segmentation.label_centroid([1])

    assert center == pytest.approx(
        l_block_centroid(spacing, origin, direction), abs=1e-6
    )
