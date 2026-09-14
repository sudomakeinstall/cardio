"""Writing a segmentation out as something another system can read.

The label volume is the app's own until it leaves; a DICOM SEG is how it
travels, and what these check is that it leaves saying what it is, where it
sits and what it was drawn on.
"""

# System
import pathlib as pl

# Third Party
import itk
import numpy as np
import pydicom as pd
import pytest

# Internal
from cardio.capture import seg, uid
from cardio.capture.equipment import Equipment
from cardio.segmentation import Segmentation
from cardio.volume import Volume
from cardio.volumetry import StructureGroup
from tests.phantoms import (
    CINE_COLUMNS,
    CINE_PIXEL_SPACING,
    CINE_ROWS,
    CINE_SLICE_SPACING,
    write_cine_series,
)

SLICES, PHASES = 3, 2


def groups() -> list[StructureGroup]:
    return [
        StructureGroup(name="LV", labels=[1], code="87878005", color=(0.8, 0.1, 0.1)),
        StructureGroup(name="Myocardium", labels=[2, 3]),
    ]


def drawn_on(tmp_path) -> tuple[Volume, Segmentation]:
    """A volume read from DICOM, and a segmentation on the same grid."""
    source = tmp_path / "source"
    write_cine_series(source, slices=SLICES, phases=PHASES)
    volume = Volume(label="vol", directory=source)

    labels = tmp_path / "labels"
    labels.mkdir()
    for frame in range(PHASES):
        array = np.zeros((SLICES, CINE_ROWS, CINE_COLUMNS), np.uint8)
        array[:, 2:6, 2:5] = 1
        array[:, 6:9, 2:5] = 2
        array[0, 9:11, 2:5] = 3
        image = itk.image_from_array(array)
        image.SetSpacing(
            [CINE_PIXEL_SPACING[1], CINE_PIXEL_SPACING[0], CINE_SLICE_SPACING]
        )
        itk.imwrite(image, str(labels / f"{frame}.nii.gz"))

    return volume, Segmentation(label="seg", directory=labels)


def write(tmp_path, **kwargs) -> list[pd.dataset.Dataset]:
    volume, segmentation = drawn_on(tmp_path)
    fields = {"groups": groups()}
    fields.update(kwargs)
    paths = seg.write_segmentation(
        segmentation, volume.source, tmp_path / "out", **fields
    )
    return [pd.dcmread(path) for path in paths]


# --- what comes out ------------------------------------------------------------


def test_a_moving_segmentation_is_one_instance_per_phase(tmp_path):
    """highdicom's fourth pixel axis is segments, not time."""
    datasets = write(tmp_path)

    assert len(datasets) == PHASES
    assert [dataset.InstanceNumber for dataset in datasets] == [1, 2]
    assert len({dataset.SeriesInstanceUID for dataset in datasets}) == 1
    assert len({dataset.SOPInstanceUID for dataset in datasets}) == PHASES


def test_it_is_written_as_the_segmentation_storage_a_viewer_reads(tmp_path):
    """A label map is the tidier object and the one nothing can open yet."""
    dataset = write(tmp_path)[0]

    assert dataset.SOPClassUID == pd.uid.SegmentationStorage
    assert dataset.Modality == "SEG"


def test_a_segment_is_a_configured_structure_rather_than_a_label_value(tmp_path):
    """DICOM numbers segments from one; the labels an editor wrote are its own."""
    dataset = write(tmp_path)[0]

    numbered = {
        item.SegmentNumber: item.SegmentLabel for item in dataset.SegmentSequence
    }

    assert numbered == {1: "LV", 2: "Myocardium"}


def test_a_coded_structure_says_what_it_is(tmp_path):
    dataset = write(tmp_path)[0]

    typed = dataset.SegmentSequence[0].SegmentedPropertyTypeCodeSequence[0]

    assert typed.CodeValue == "87878005"
    assert typed.CodingSchemeDesignator == "SCT"


def test_an_uncoded_structure_is_typed_as_tissue_and_said_so(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        dataset = write(tmp_path)[0]

    typed = dataset.SegmentSequence[1].SegmentedPropertyTypeCodeSequence[0]

    assert typed.CodeValue == seg.UNTYPED_SEGMENT.value
    assert "Myocardium" in caplog.text
    assert "set its code to say what it is" in caplog.text


def test_the_curve_colour_is_what_a_viewer_is_asked_to_draw(tmp_path):
    dataset = write(tmp_path)[0]

    assert "RecommendedDisplayCIELabValue" in dataset.SegmentSequence[0]
    assert "RecommendedDisplayCIELabValue" not in dataset.SegmentSequence[1]


def test_the_labels_a_group_names_are_one_segment(tmp_path):
    """Myocardium is labels 2 and 3, which is one structure and not two."""
    dataset = write(tmp_path)[0]
    _volume, segmentation = drawn_on(tmp_path / "again")

    mapped = seg.label_map(segmentation.mpr_image_data(0), groups())

    assert set(np.unique(mapped)) == {0, 1, 2}
    assert len(dataset.SegmentSequence) == 2


# --- where it sits -------------------------------------------------------------


def test_the_planes_come_from_the_grid_the_app_measured_on(tmp_path):
    _volume, segmentation = drawn_on(tmp_path)
    image_data = segmentation.mpr_image_data(0)

    positions, orientation, measures = seg.geometry_of(image_data)

    assert len(positions) == SLICES
    assert np.allclose(positions[0][0].ImagePositionPatient, image_data.GetOrigin())
    assert np.allclose(
        measures[0].PixelSpacing,
        [CINE_PIXEL_SPACING[0], CINE_PIXEL_SPACING[1]],
    )
    assert float(measures[0].SliceThickness) == CINE_SLICE_SPACING
    assert len(orientation[0].ImageOrientationPatient) == 6


def test_it_keeps_the_study_and_frame_of_reference_it_was_drawn_in(tmp_path):
    datasets = write(tmp_path)
    source = pd.dcmread(next((tmp_path / "source").glob("*.dcm")))

    for dataset in datasets:
        assert dataset.StudyInstanceUID == source.StudyInstanceUID
        assert str(dataset.PatientName) == "Phantom^Cine"
        assert dataset.FrameOfReferenceUID == source.FrameOfReferenceUID


def test_it_cites_the_images_it_was_drawn_on(tmp_path):
    dataset = write(tmp_path)[0]
    written = {
        str(pd.dcmread(path).SOPInstanceUID)
        for path in (tmp_path / "source").glob("*.dcm")
    }

    cited = {
        str(item.ReferencedSOPInstanceUID)
        for item in dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence
    }

    assert cited == written


def test_every_uid_is_minted_under_the_configured_root(tmp_path):
    dataset = write(tmp_path, uid_root="1.2.840.99999.1")[0]

    assert dataset.SeriesInstanceUID.startswith("1.2.840.99999.1.")
    assert dataset.SOPInstanceUID.startswith("1.2.840.99999.1.")
    assert dataset.file_meta.ImplementationClassUID == "1.2.840.99999.1.1"


def test_the_configured_equipment_is_what_it_names(tmp_path):
    dataset = write(tmp_path, equipment=Equipment(manufacturer="Acme"))[0]

    assert dataset.Manufacturer == "Acme"


def test_it_says_where_it_was_produced(tmp_path):
    """The two names highdicom's constructor does not take, and the one object
    of these a viewer draws: a capture that says its site and a segmentation
    that does not would be the same session filed two ways."""
    dataset = write(
        tmp_path,
        equipment=Equipment(institution_name="St Elsewhere", station_name="READING-3"),
    )[0]

    assert dataset.InstitutionName == "St Elsewhere"
    assert dataset.StationName == "READING-3"


def test_a_site_that_was_never_named_is_left_out_rather_than_written_empty(tmp_path):
    """Type 3: absent says not recorded, empty claims the value itself is."""
    dataset = write(tmp_path, equipment=Equipment())[0]

    assert "InstitutionName" not in dataset
    assert "StationName" not in dataset


# --- what it will not write ----------------------------------------------------


def test_a_segmentation_of_a_file_has_nothing_to_refer_to(tmp_path):
    """A SEG names the images it segments by UID, and a NIfTI volume has none."""
    _volume, segmentation = drawn_on(tmp_path)

    with pytest.raises(ValueError, match="read from DICOM"):
        seg.write_segmentation(
            segmentation, segmentation.source, tmp_path / "out", groups=groups()
        )


def test_nothing_is_written_until_a_structure_is_named(tmp_path):
    volume, segmentation = drawn_on(tmp_path)

    with pytest.raises(ValueError, match="at least one"):
        seg.write_segmentation(segmentation, volume.source, tmp_path / "out", groups=[])


def test_a_sparse_research_source_is_filled_rather_than_refused(tmp_path):
    """An absent Type 2 element should be a message, not an AttributeError."""
    volume, _segmentation = drawn_on(tmp_path)
    sparse = volume.source.datasets
    for dataset in sparse:
        del dataset.PatientBirthDate
        del dataset.StudyID

    filled = seg.conforming(sparse)

    assert all("PatientBirthDate" in dataset for dataset in filled)
    assert all(dataset.StudyID is None for dataset in filled)
    # The originals are untouched, so nothing downstream sees an invented field.
    assert all("PatientBirthDate" not in dataset for dataset in sparse)


def test_the_default_uid_root_is_the_one_a_research_session_gets(tmp_path):
    dataset = write(tmp_path)[0]

    assert dataset.SeriesInstanceUID.startswith(uid.DEFAULT_ROOT)


def test_written_paths_are_numbered_in_frame_order(tmp_path):
    volume, segmentation = drawn_on(tmp_path)

    paths = seg.write_segmentation(
        segmentation, volume.source, tmp_path / "out", groups=groups()
    )

    assert [path.name for path in paths] == ["0000.dcm", "0001.dcm"]
    assert all(isinstance(path, pl.Path) for path in paths)
