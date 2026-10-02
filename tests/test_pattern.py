"""A frame pattern names a different file for every frame."""

# Third Party
import itk
import numpy as np
import pydantic as pc
import pytest

# Internal
from cardio.segmentation import Segmentation


@pytest.mark.parametrize(
    "pattern",
    [
        "$frame.nii.gz",
        "frame.nii.gz",
        "${frame}.nii.gz",
        "{frame}_{phase}.nii.gz",
        "{frame.nii.gz",
    ],
)
def test_a_pattern_without_a_lone_frame_placeholder_is_refused(tmp_path, pattern):
    """Each of these would name one file for every frame, or fail to format."""
    with pytest.raises(pc.ValidationError):
        Segmentation(label="s", directory=tmp_path, pattern=pattern)


def test_a_formatted_frame_placeholder_reads_each_frame(tmp_path):
    names = ["0000.nii.gz", "0001.nii.gz"]
    for name in names:
        image = itk.image_from_array(np.zeros((4, 4, 4), dtype=np.uint8))
        itk.imwrite(image, str(tmp_path / name))

    segmentation = Segmentation(
        label="s", directory=tmp_path, pattern="{frame:04d}.nii.gz"
    )

    assert [path.name for path in segmentation.path_list] == names
