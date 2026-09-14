"""Writing the volume curves out as something a reporting system can read.

Two CSV files are what a person opens.  What these check is that the numbers
also leave in a form that says, for each of them, what it is, what it is in and
what it is of -- which a column heading does not.
"""

# Third Party
import pydicom as pd
import pytest

# Internal
from cardio.capture import seg, sr
from cardio.capture.equipment import Equipment
from cardio.volumetry import StructureGroup, measure
from tests.test_seg import PHASES, drawn_on, groups

VOXEL_ML = 0.012
BSA = 1.8


def measured(segmentation, structures=None, bsa=BSA):
    structures = structures or groups()
    counts = [segmentation.label_counts(frame) for frame in range(PHASES)]
    return measure(counts, voxel_ml=VOXEL_ML, groups=structures, bsa=bsa)


def report(tmp_path, structures=None, bsa=BSA, with_segmentation=True, **kwargs):
    volume, segmentation = drawn_on(tmp_path)
    structures = structures or groups()
    result = measured(segmentation, structures, bsa)

    segmentations = []
    if with_segmentation:
        paths = seg.write_segmentation(
            segmentation, volume.source, tmp_path / "seg", groups=structures
        )
        segmentations = [pd.dcmread(path) for path in paths]

    path = sr.write_measurements(
        result,
        volume.source,
        tmp_path / "sr.dcm",
        segmentations=segmentations,
        **kwargs,
    )
    return pd.dcmread(path), result


def content(dataset, kind: str) -> list:
    """Every content item of ``kind`` anywhere in the document."""
    found = []

    def walk(sequence):
        for item in sequence:
            if item.ValueType == kind:
                found.append(item)
            if "ContentSequence" in item:
                walk(item.ContentSequence)

    walk(dataset.ContentSequence)
    return found


def numbers(dataset) -> list[tuple[str, float, str]]:
    """Every measurement as (what it is, the number, the unit)."""
    return [
        (
            item.ConceptNameCodeSequence[0].CodeMeaning,
            float(item.MeasuredValueSequence[0].NumericValue),
            item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue,
        )
        for item in content(dataset, "NUM")
    ]


# --- what comes out ------------------------------------------------------------


def test_it_is_written_as_a_structured_report(tmp_path):
    dataset, _result = report(tmp_path)

    assert dataset.SOPClassUID == pd.uid.Comprehensive3DSRStorage
    assert dataset.Modality == "SR"


def test_it_lands_in_the_study_the_curves_were_measured_in(tmp_path):
    dataset, _result = report(tmp_path)
    source = pd.dcmread(next((tmp_path / "source").glob("*.dcm")))

    assert dataset.StudyInstanceUID == source.StudyInstanceUID
    assert str(dataset.PatientName) == "Phantom^Cine"


def test_a_measurement_group_is_one_named_structure(tmp_path):
    dataset, result = report(tmp_path)

    tracked = [item.TextValue for item in content(dataset, "TEXT")]

    assert set(result.names) <= set(tracked)


def test_the_extremes_are_the_curve_s_own_rather_than_a_cardiac_phase(tmp_path):
    """Nothing here was told which phase a frame was acquired at."""
    dataset, _result = report(tmp_path)

    derivations = {
        item.ConceptCodeSequence[0].CodeMeaning for item in content(dataset, "CODE")
    }

    assert {"Maximum", "Minimum"} <= derivations
    assert not any("diastol" in value.lower() for value in derivations)


def test_every_number_carries_the_unit_it_is_in(tmp_path):
    dataset, _result = report(tmp_path)

    assert numbers(dataset)
    assert all(unit for _name, _value, unit in numbers(dataset))
    assert {"ml", "%", "ml/m2"} <= {unit for _n, _v, unit in numbers(dataset)}


def test_the_measured_values_are_the_ones_the_curves_came_to(tmp_path):
    dataset, result = report(tmp_path)

    reported = {round(value, 6) for _n, value, _u in numbers(dataset)}
    expected = result.measurements[0].metrics.maximum

    assert round(expected, 6) in reported


def test_indexed_values_appear_only_when_a_body_surface_area_is_known(tmp_path):
    dataset, _result = report(tmp_path, bsa=None)

    assert "ml/m2" not in {unit for _n, _v, unit in numbers(dataset)}


def test_a_structure_given_a_density_is_reported_as_a_mass(tmp_path):
    structures = [
        StructureGroup(name="LV", labels=[1], code="87878005"),
        StructureGroup(name="Myocardium", labels=[2, 3], density=1.05),
    ]

    dataset, _result = report(tmp_path, structures=structures)

    quoted = {(name, unit) for name, _value, unit in numbers(dataset)}

    assert ("Mass", "g") in quoted
    assert ("Mass", "g/m2") in quoted
    assert ("Volume", "ml") in quoted


def test_a_coded_structure_says_where_in_the_body_it_is(tmp_path):
    dataset, _result = report(tmp_path)

    sites = [
        item
        for item in content(dataset, "CODE")
        if item.ConceptNameCodeSequence[0].CodeMeaning == "Finding Site"
    ]

    assert [item.ConceptCodeSequence[0].CodeValue for item in sites] == ["87878005"]


def test_it_points_at_the_segmentation_it_was_measured_from(tmp_path):
    dataset, _result = report(tmp_path)

    referenced = content(dataset, "IMAGE")

    assert len(referenced) == 2
    assert all(
        item.ReferencedSOPSequence[0].ReferencedSOPClassUID
        == pd.uid.SegmentationStorage
        for item in referenced
    )


def test_without_a_segmentation_the_groups_are_still_reported(tmp_path):
    dataset, result = report(tmp_path, with_segmentation=False)

    assert content(dataset, "IMAGE") == []
    assert len(numbers(dataset)) >= 2 * len(result.measurements)


def test_the_observer_is_the_application_that_measured(tmp_path):
    dataset, _result = report(
        tmp_path,
        equipment=Equipment(manufacturer="Acme", model_name="Cardio Station"),
        uid_root="1.2.840.99999.1",
    )

    texts = {item.TextValue for item in content(dataset, "TEXT")}

    assert "Acme" in texts
    assert "Cardio Station" in texts
    assert dataset.file_meta.ImplementationClassUID == "1.2.840.99999.1.1"
    assert dataset.SeriesInstanceUID.startswith("1.2.840.99999.1.")


def test_the_images_it_is_evidence_about_are_named(tmp_path):
    dataset, _result = report(tmp_path)

    assert dataset.CurrentRequestedProcedureEvidenceSequence


# --- what it will not write ----------------------------------------------------


def test_a_report_about_a_file_has_no_images_to_name(tmp_path):
    _volume, segmentation = drawn_on(tmp_path)
    result = measured(segmentation)

    with pytest.raises(ValueError, match="read from DICOM"):
        sr.write_measurements(result, segmentation.source, tmp_path / "sr.dcm")


def test_nothing_is_reported_until_a_structure_is_named(tmp_path):
    volume, _segmentation = drawn_on(tmp_path)
    result = measure([], voxel_ml=VOXEL_ML, groups=[])

    with pytest.raises(ValueError, match="nothing to report"):
        sr.write_measurements(result, volume.source, tmp_path / "sr.dcm")


# --- what the save action writes -----------------------------------------------


def test_a_research_save_writes_the_tables_and_says_why_that_is_all(tmp_path):
    """A volume read from a file gives a report nothing to name."""
    from tests.test_volumetry_logic import built

    _server, scene, logic = built(tmp_path)
    logic.volumetry.save_volumetry()

    directory = max((scene.volumetry_directory).iterdir())

    assert (directory / "timeseries.csv").exists()
    assert (directory / "metrics.csv").exists()
    assert not (directory / "measurements.dcm").exists()
    assert logic.volumetry.server.state.volumetry_ok
    assert "DICOM" not in logic.volumetry.server.state.volumetry_summary


def dicom_backed_app(tmp_path, research: bool = True, **volumetry_overrides):
    """An app whose volume came from DICOM and whose labels are on its grid.

    A research session unless a test says otherwise: the phantom is written
    under pydicom's root, which a deployment that has not said it is research
    refuses to write a report under.
    """
    from cardio.scene import Scene
    from cardio.volumetry import Volumetry
    from tests.test_app_smoke import build_app, connect

    # Written for their side effect on disk; the scene reads them back.
    drawn_on(tmp_path)
    scene = Scene(
        volumes=[{"label": "vol", "directory": tmp_path / "source"}],
        segmentations=[{"label": "seg", "directory": tmp_path / "labels"}],
        serialization_directory=tmp_path / "out",
        active_volume_label="vol",
        research=research,
        volumetry=Volumetry(
            segmentation_label="seg", groups=groups(), **volumetry_overrides
        ),
        view={"layout": "volumetry"},
    )
    server, _scene, logic, _ui = build_app(scene)
    connect(server)
    return server, scene, logic


def test_a_dicom_backed_save_writes_the_segmentation_and_the_report(tmp_path):
    _server, scene, logic = dicom_backed_app(tmp_path)

    logic.volumetry.save_volumetry()

    directory = max(scene.volumetry_directory.iterdir())
    instances = sorted((directory / "segmentation").glob("*.dcm"))

    assert (directory / "timeseries.csv").exists()
    assert len(instances) == PHASES
    assert (directory / "measurements.dcm").exists()
    assert "DICOM instance(s)" in logic.volumetry.server.state.volumetry_summary


def test_a_save_under_a_borrowed_root_writes_the_tables_and_not_the_report(tmp_path):
    """The tables are nobody's to mistake for an archive object; the report is."""
    _server, scene, logic = dicom_backed_app(tmp_path, research=False)

    logic.volumetry.save_volumetry()

    directory = max(scene.volumetry_directory.iterdir())

    assert (directory / "timeseries.csv").exists()
    assert not (directory / "measurements.dcm").exists()
    assert not (directory / "segmentation").exists()
    assert "DICOM" not in logic.volumetry.server.state.volumetry_summary


def test_the_written_report_points_at_the_written_segmentation(tmp_path):
    _server, scene, logic = dicom_backed_app(tmp_path)

    logic.volumetry.save_volumetry()

    directory = max(scene.volumetry_directory.iterdir())
    written = pd.dcmread(min((directory / "segmentation").glob("*.dcm")))
    document = pd.dcmread(directory / "measurements.dcm")

    cited = {
        item.ReferencedSOPSequence[0].ReferencedSOPInstanceUID
        for item in content(document, "IMAGE")
    }

    assert cited == {written.SOPInstanceUID}
    assert document.StudyInstanceUID == written.StudyInstanceUID
