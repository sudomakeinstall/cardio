"""What the metadata sheet says about a loaded object.

The gap these guard: the header a file arrived with exists for one moment,
between the reader and the conversion to VTK. If it is not captured there it is
gone, and the sheet quietly shows a shorter table instead of failing.
"""

# System
import pathlib as pl

# Third Party
import itk
import numpy as np
import pytest
import vtk

# Internal
from cardio import metadata
from cardio.mesh import Mesh
from cardio.orientation import axcode_from_direction, geometry_of, read_source
from cardio.scene import Scene
from cardio.segmentation import Segmentation
from cardio.volume import Volume
from tests.phantoms import write_cine_series


def write_volume(path, direction=None, spacing=(1.0, 2.0, 3.0)):
    """A small volume whose geometry a test can state in one sentence."""
    array = np.linspace(-500, 500, 4 * 5 * 6, dtype=np.float32).reshape(6, 5, 4)
    image = itk.image_from_array(array)
    image.SetSpacing(spacing)
    image.SetOrigin((-1.0, 2.0, -3.0))
    if direction is not None:
        image.SetDirection(itk.matrix_from_array(np.asarray(direction)))
    itk.imwrite(image, str(path))
    return image


def rows(section) -> dict[str, str]:
    return {row.name: row.value for row in section.rows}


def section(entry, title):
    for candidate in entry.sections:
        if candidate.title == title:
            return candidate
    raise AssertionError(f"No {title!r} section in {[s.title for s in entry.sections]}")


# --- orientation codes --------------------------------------------------------


def test_the_identity_direction_is_lps():
    assert axcode_from_direction(np.eye(3)) == "LPS"


def test_a_negated_direction_is_rai():
    assert axcode_from_direction(-np.eye(3)) == "RAI"


def test_a_permuted_direction_names_the_axes_in_index_order():
    # Index axis 0 points superior, 1 posterior, 2 left.
    direction = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert axcode_from_direction(direction) == "SPL"


def test_an_oblique_direction_takes_the_nearest_axes():
    theta = np.radians(20.0)
    tilt = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert axcode_from_direction(tilt) == "LPS"


# --- geometry -----------------------------------------------------------------


def test_geometry_reports_the_image_it_was_read_from(tmp_path):
    write_volume(tmp_path / "0.nii.gz")

    frames, _ = read_source(tmp_path / "0.nii.gz")
    geometry = geometry_of(frames[0])

    assert geometry.size == (4, 5, 6)
    assert geometry.spacing == pytest.approx((1.0, 2.0, 3.0))
    assert geometry.origin == pytest.approx((-1.0, 2.0, -3.0))
    assert geometry.extent == pytest.approx((4.0, 10.0, 18.0))
    assert geometry.axcode == "LPS"
    assert geometry.voxel_type == "float32"
    assert geometry.intensity_range == pytest.approx((-500.0, 500.0))


def test_the_geometry_section_reports_every_field(tmp_path):
    write_volume(tmp_path / "0.nii.gz")
    volume = Volume(label="vol", directory=tmp_path)

    fields = rows(section(metadata.describe(volume), "Geometry"))

    assert fields["Size (voxels)"] == "4  5  6"
    assert fields["Orientation"] == "LPS"
    assert fields["Voxel type"] == "float32"
    assert fields["Direction"].count("\n") == 2


# --- what came with the image -------------------------------------------------


def test_a_nifti_volume_keeps_its_header(tmp_path):
    write_volume(tmp_path / "0.nii.gz")
    volume = Volume(label="vol", directory=tmp_path)

    header = section(metadata.describe(volume), "NIfTI header")

    assert header.rows
    assert rows(header)["qform_code"]


def test_a_dicom_volume_keeps_its_patient_and_series_tags(tmp_path):
    write_cine_series(tmp_path, slices=3, phases=2)
    volume = Volume(label="vol", directory=tmp_path)

    fields = rows(section(metadata.describe(volume), "DICOM series header"))

    assert fields["PatientName"] == "Phantom^Cine"
    assert fields["PatientID"] == "PHANTOM-1"
    assert fields["Modality"] == "MR"
    assert fields["SeriesDescription"] == "cine"
    assert fields["TransferSyntaxUID"]


def test_the_source_section_names_the_format_and_the_frames(tmp_path):
    write_cine_series(tmp_path, slices=3, phases=2)
    volume = Volume(label="vol", directory=tmp_path)

    fields = rows(section(metadata.describe(volume), "Source"))

    assert fields["Format"] == "DICOM series"
    assert fields["Kind"] == "volume"
    assert fields["Frames"] == "2"


def test_a_left_handed_image_says_it_was_corrected(tmp_path):
    """The flip is silent at load; the sheet is the only place it is visible."""
    left_handed = np.diag([1.0, 1.0, -1.0])
    write_volume(tmp_path / "0.nii.gz", direction=left_handed)
    volume = Volume(label="vol", directory=tmp_path)

    fields = rows(section(metadata.describe(volume), "Handedness"))

    assert fields["Right-handed correction"].startswith("applied")


def test_a_right_handed_image_says_no_correction_was_needed(tmp_path):
    write_volume(tmp_path / "0.nii.gz")
    volume = Volume(label="vol", directory=tmp_path)

    fields = rows(section(metadata.describe(volume), "Handedness"))

    assert fields["Right-handed correction"] == "not needed"


# --- the objects that are not volumes -----------------------------------------


def write_mesh(path):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(3.0)
    sphere.Update()
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(sphere.GetOutput())
    writer.Write()


def test_a_mesh_is_described_by_its_surface(tmp_path):
    write_mesh(tmp_path / "0.obj")
    mesh = Mesh(label="mesh", directory=tmp_path)

    entry = metadata.describe(mesh)
    fields = rows(section(entry, "Surface"))

    assert [s.title for s in entry.sections] == ["Source", "Surface"]
    assert int(fields["Points"]) > 0
    assert int(fields["Cells"]) > 0
    assert rows(section(entry, "Source"))["Format"] == "OBJ"


def test_a_volume_has_no_surface_section(tmp_path):
    """A volume mapper has an input too, but it is image data, not a surface."""
    write_volume(tmp_path / "0.nii.gz")

    titles = [
        s.title
        for s in metadata.describe(Volume(label="v", directory=tmp_path)).sections
    ]

    assert "Surface" not in titles


def test_a_segmentation_reports_its_label_range(tmp_path):
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[2:6, 2:6, 1:3] = 1
    array[2:6, 2:6, 3:5] = 2
    itk.imwrite(itk.image_from_array(array), str(tmp_path / "0.nii.gz"))

    fields = rows(
        section(
            metadata.describe(Segmentation(label="seg", directory=tmp_path)), "Labels"
        )
    )

    assert fields["Label range"] == "0 to 2"


# --- the scene ----------------------------------------------------------------


def test_an_empty_scene_describes_nothing():
    assert metadata.describe_scene(Scene()) == []


def test_a_mesh_and_a_volume_sharing_a_label_get_distinct_keys(tmp_path):
    """The keys are what the sheet's dropdown switches on, so they must differ."""
    write_volume(tmp_path / "0.nii.gz")
    write_mesh(tmp_path / "0.obj")

    scene = Scene(
        volumes=[Volume(label="both", directory=tmp_path)],
        meshes=[Mesh(label="both", directory=tmp_path)],
    )

    keys = [entry.key for entry in metadata.describe_scene(scene)]

    assert sorted(keys) == ["mesh:both", "volume:both"]


def test_describe_scene_covers_every_renderable(tmp_path):
    write_volume(tmp_path / "0.nii.gz")
    write_mesh(tmp_path / "0.obj")
    itk.imwrite(
        itk.image_from_array(np.ones((8, 8, 8), dtype=np.uint8)),
        str(tmp_path / "seg0.nii.gz"),
    )

    scene = Scene(
        volumes=[Volume(label="vol", directory=tmp_path)],
        meshes=[Mesh(label="mesh", directory=tmp_path)],
        segmentations=[
            Segmentation(label="seg", directory=tmp_path, file_paths=["seg0.nii.gz"])
        ],
    )

    assert [entry.kind for entry in metadata.describe_scene(scene)] == [
        "mesh",
        "volume",
        "segmentation",
    ]


def test_a_file_per_frame_series_keeps_the_first_frames_header(tmp_path):
    """Every frame is one acquisition, so one header describes the series."""
    for frame in range(3):
        write_volume(tmp_path / f"{frame}.nii.gz")

    volume = Volume(label="vol", directory=pl.Path(tmp_path))

    assert volume.source.frame_count == 1
    assert rows(section(metadata.describe(volume), "Source"))["Frames"] == "3"
