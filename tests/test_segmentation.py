"""Test Segmentation centroid extraction."""

import itk
import numpy as np
import pytest
import vtk

from cardio.segmentation import Segmentation, masked_centroid

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
    image = itk.image_from_array(label_array())
    itk.imwrite(image, str(tmp_path / "seg.nii.gz"))
    return Segmentation(label="test", directory=tmp_path, file_paths=["seg.nii.gz"])


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


def test_masked_centroid_all_true_matches_full_mesh(mesh):
    """An all-true mask reproduces the center of mass of the whole mesh."""
    com = vtk.vtkCenterOfMass()
    com.SetInputData(mesh)
    com.SetUseScalarsAsWeights(False)
    com.Update()
    expected = list(com.GetCenter())

    mask = [True] * mesh.GetNumberOfCells()
    assert masked_centroid(mesh, mask) == pytest.approx(expected, abs=1e-6)


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
