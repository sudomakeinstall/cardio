"""A scripted session on an install without the ui extra.

The widgets and the browser's image push come with ``cardio[ui]``, and mp4
with ``cardio[video]``; a deployment that only scripts captures installs
neither. Blocking them here stands in for their not being installed.
"""

# System
import subprocess
import sys

UI_ONLY = ("trame_vuetify", "trame_vtk", "trame.widgets", "trame.ui", "imageio_ffmpeg")

SCRIPT = """
import pathlib as pl
import sys

for name in {blocked!r}:
    sys.modules[name] = None

import itk
import numpy as np

import cardio

directory = pl.Path({directory!r})
volume = np.zeros((16, 16, 16), np.int16)
volume[4:12, 4:12, 4:12] = 400
itk.imwrite(itk.image_from_array(volume), directory / "vol.nii.gz")
itk.imwrite(
    itk.image_from_array((volume > 0).astype(np.uint8)), directory / "seg.nii.gz"
)

do = cardio.script(
    None,
    volumes=[{{"label": "vol", "directory": directory, "file_paths": ["vol.nii.gz"]}}],
    segmentations=[
        {{"label": "seg", "directory": directory, "file_paths": ["seg.nii.gz"]}}
    ],
    active_volume_label="vol",
    serialization_directory=directory / "out",
    capture_format="png",
)
do.add_rotation(axis="Z")
do.screenshot()
assert list((directory / "out").rglob("*.png")), "nothing captured"
"""


def test_a_session_scripts_a_capture_without_the_ui_extra(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(blocked=UI_ONLY, directory=str(tmp_path))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
