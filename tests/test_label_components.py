"""Thinning a label down to its largest connected components as it is read."""

# Third Party
import numpy as np
import pydantic as pc
import pytest

# Internal
from cardio.segmentation import (
    Segmentation,
    component_ranks,
    keep_largest_components,
    label_mask,
)
from tests.phantoms import write_segmentation

N = 24

# The island sits well away from the block it shares a label with, which is
# what makes it worth erasing: near enough to matter to a plane fit, far enough
# that nobody looking at the labels would call it part of the structure.
ISLAND = (slice(20, 22), slice(20, 22), slice(20, 22))


def island_array() -> np.ndarray:
    """Label 1 as a block and a detached cube; label 2 as one block beside it."""
    array = np.zeros((N,) * 3, dtype=np.uint8)
    array[4:12, 4:12, 4:12] = 1
    array[ISLAND] = 1
    array[4:12, 4:12, 12:16] = 2
    return array


def test_component_ranks_number_the_largest_first():
    mask = island_array() == 1
    ranks = component_ranks(mask)

    assert ranks.max() == 2
    assert ranks[ISLAND].ravel().tolist() == [2] * 8
    assert (ranks == 1).sum() == 8**3


def test_component_ranks_counts_faces_rather_than_corners():
    """Two cubes meeting at a corner alone are two components, not one."""
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    mask[3:5, 3:5, 3:5] = True

    assert component_ranks(mask).max() == 2


def test_keeping_one_component_erases_the_island():
    array = island_array()

    dropped = keep_largest_components(array, {1: 1})

    assert dropped == {1: 8}
    assert array[ISLAND].sum() == 0
    assert (array == 1).sum() == 8**3


def test_labels_not_named_keep_every_component():
    array = island_array()

    assert keep_largest_components(array, {2: 1}) == {}
    assert array[ISLAND].ravel().tolist() == [1] * 8


def test_keeping_more_components_than_there_are_drops_nothing():
    array = island_array()

    assert keep_largest_components(array, {1: 5}) == {}
    assert (array == 1).sum() == 8**3 + 8


def test_a_label_the_frame_does_not_have_is_passed_over():
    array = island_array()

    assert keep_largest_components(array, {7: 1}) == {}


def test_the_erased_voxels_become_background_rather_than_a_neighbour():
    array = island_array()
    keep_largest_components(array, {1: 1})

    assert set(np.unique(array).tolist()) == {0, 1, 2}


def thinned(directory, arrays, counts) -> Segmentation:
    """The same frames read twice: once to write them, once thinned."""
    written = write_segmentation(directory, arrays, stem="island")
    return Segmentation(
        label=written.label,
        directory=written.directory,
        file_paths=written.file_paths,
        label_components=counts,
    )


def test_a_segmentation_reads_its_frames_thinned(tmp_path):
    """The label image the rest of the app measures has no island in it."""
    segmentation = thinned(tmp_path, [island_array()], {1: 1})

    assert label_mask(segmentation._label_images[0], [1]).sum() == 8**3


def test_every_frame_is_thinned(tmp_path):
    segmentation = thinned(tmp_path, [island_array()] * 3, {1: 1})

    assert len(segmentation._label_images) == 3
    for image in segmentation._label_images:
        assert label_mask(image, [1]).sum() == 8**3


def test_an_unconfigured_segmentation_keeps_what_it_read(tmp_path):
    segmentation = thinned(tmp_path, [island_array()], {})

    assert label_mask(segmentation._label_images[0], [1]).sum() == 8**3 + 8


def test_keeping_nothing_is_refused(tmp_path):
    """A zero is a mistyped count rather than a way to spell "keep all"."""
    with pytest.raises(pc.ValidationError):
        Segmentation(
            label="s",
            directory=tmp_path,
            pattern="{frame}.nii.gz",
            label_components={1: 0},
        )


def test_rank_by_size_orders_more_components_than_a_narrow_label_can_hold():
    """The speckle this pass thins is what a 16-bit relabelling would refuse."""
    mask = np.zeros((80,) * 3, dtype=np.uint8)
    mask[::2, ::2, ::2] = 1
    mask[0, 0, :2] = 1

    ranks = component_ranks(mask == 1)

    assert ranks.max() > np.iinfo(np.int16).max
    assert ranks[0, 0, 0] == 1, "the one two-voxel component ranks ahead of the rest"
