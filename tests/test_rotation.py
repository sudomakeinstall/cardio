import numpy as np
import pydantic as pc
import pytest

from cardio.orientation import AngleUnits, IndexOrder
from cardio.rotation import RotationMetadata, RotationSequence, RotationStep


def test_rotation_step_creation():
    step = RotationStep(axis="X", angle=1.57)
    assert step.axis == "X"
    assert step.angle == 1.57
    assert step.visible is True
    assert step.name == ""
    assert step.name_editable is True
    assert step.deletable is True


# NOTE: Conversion methods removed - data is always stored in current convention/units
# Conversions happen at UI layer when user changes settings via sync_index_order() and sync_angle_units()


def test_rotation_metadata_creation():
    meta = RotationMetadata(volume_label="CCTA")
    assert meta.coordinate_system == "LPS"
    assert meta.index_order == IndexOrder.ITK
    assert meta.angle_units == AngleUnits.RADIANS
    assert meta.volume_label == "CCTA"


def test_rotation_metadata_invalid_coordinate_system():
    with pytest.raises(pc.ValidationError):
        RotationMetadata(coordinate_system="RAS")


def test_rotation_sequence_creation():
    seq = RotationSequence()
    assert seq.metadata.coordinate_system == "LPS"
    assert len(seq.angles_list) == 0


def test_rotation_sequence_with_steps():
    steps = [
        RotationStep(axis="X", angle=0.5),
        RotationStep(axis="Y", angle=1.0),
    ]
    seq = RotationSequence(angles_list=steps)
    assert len(seq.angles_list) == 2
    assert seq.angles_list[0].axis == "X"
    assert seq.angles_list[1].axis == "Y"


def test_rotation_sequence_model_dump():
    """Test that model_dump includes metadata, angles_list, and mpr_origin."""
    steps = [
        RotationStep(axis="X", angle=0.5, name="First"),
        RotationStep(axis="Y", angle=1.0, name="Second", visible=False),
    ]
    seq = RotationSequence(angles_list=steps)
    seq.metadata.volume_label = "Test"

    ui_dict = seq.model_dump(mode="json")

    # Check angles_list
    assert "angles_list" in ui_dict
    assert len(ui_dict["angles_list"]) == 2
    assert ui_dict["angles_list"][0]["axis"] == "X"
    assert ui_dict["angles_list"][0]["angle"] == 0.5
    assert ui_dict["angles_list"][0]["name"] == "First"
    assert ui_dict["angles_list"][1]["visible"] is False

    # Check metadata is included
    assert "metadata" in ui_dict
    assert ui_dict["metadata"]["coordinate_system"] == "LPS"
    assert ui_dict["metadata"]["index_order"] == "itk"
    assert ui_dict["metadata"]["angle_units"] == "radians"
    assert ui_dict["metadata"]["volume_label"] == "Test"

    # Check mpr_origin
    assert "mpr_origin" in ui_dict
    assert ui_dict["mpr_origin"] == [0.0, 0.0, 0.0]


def test_rotation_sequence_from_dict():
    """Test creating RotationSequence from full dict structure."""
    data = {
        "metadata": {
            "coordinate_system": "LPS",
            "index_order": "itk",
            "angle_units": "radians",
            "timestamp": "2026-01-16T12:00:00",
            "volume_label": "CCTA",
            "deletable": True,
        },
        "angles_list": [
            {
                "axis": "X",
                "angle": 0.5,
                "name": "Test",
                "visible": True,
                "name_editable": True,
                "deletable": True,
            },
            {
                "axis": "Y",
                "angle": 1.0,
                "visible": True,
                "name": "",
                "name_editable": True,
                "deletable": True,
            },
        ],
        "mpr_origin": [10.0, 20.0, 30.0],
    }
    seq = RotationSequence(**data)

    assert len(seq.angles_list) == 2
    assert seq.angles_list[0].axis == "X"
    assert seq.angles_list[0].angle == 0.5
    assert seq.metadata.volume_label == "CCTA"
    assert seq.mpr_origin == [10.0, 20.0, 30.0]


def test_rotation_sequence_to_toml_itk():
    steps = [RotationStep(axis="X", angle=1.57, name="Rotate X")]
    seq = RotationSequence(angles_list=steps)
    seq.metadata.volume_label = "CCTA"
    seq.metadata.index_order = IndexOrder.ITK
    seq.metadata.angle_units = AngleUnits.RADIANS

    toml_str = seq.to_toml()

    assert 'axis = "X"' in toml_str
    assert "angle = 1.57" in toml_str
    assert 'index_order = "itk"' in toml_str
    assert 'angle_units = "radians"' in toml_str
    assert 'volume_label = "CCTA"' in toml_str


def test_rotation_sequence_to_toml_roma():
    steps = [RotationStep(axis="Z", angle=-1.57, name="Rotate Z")]
    seq = RotationSequence(angles_list=steps)
    seq.metadata.index_order = IndexOrder.ROMA
    seq.metadata.angle_units = AngleUnits.RADIANS

    toml_str = seq.to_toml()

    assert 'axis = "Z"' in toml_str
    assert "angle = -1.57" in toml_str
    assert 'index_order = "roma"' in toml_str


def test_rotation_sequence_to_toml_degrees():
    steps = [RotationStep(axis="Y", angle=90.0)]
    seq = RotationSequence(angles_list=steps)
    seq.metadata.index_order = IndexOrder.ITK
    seq.metadata.angle_units = AngleUnits.DEGREES

    toml_str = seq.to_toml()

    assert 'angle_units = "degrees"' in toml_str
    assert "90.0" in toml_str


def test_rotation_sequence_version_comment_in_toml():
    """Test that cardio version comment is included when serializing."""
    from cardio import __version__

    seq = RotationSequence()
    seq.metadata.volume_label = "Test"

    toml_str = seq.to_toml()

    assert f"# Generated by cardio version {__version__}" in toml_str
    assert toml_str.startswith("# Generated by cardio version")


def test_rotation_sequence_from_toml_itk():
    toml_content = """
[metadata]
coordinate_system = "LPS"
index_order = "itk"
angle_units = "radians"
timestamp = "2026-01-16T12:00:00"
volume_label = "CCTA"

[[angles_list]]
axis = "X"
angle = 1.57
visible = true
name = "First"
name_editable = true
deletable = true

[[angles_list]]
axis = "Y"
angle = 0.5
visible = false
name = "Second"
name_editable = true
deletable = true
"""
    seq = RotationSequence.from_toml(toml_content)

    assert len(seq.angles_list) == 2
    assert seq.angles_list[0].axis == "X"
    assert seq.angles_list[0].angle == 1.57
    assert seq.angles_list[0].name == "First"
    assert seq.angles_list[1].axis == "Y"
    assert seq.angles_list[1].angle == 0.5
    assert seq.angles_list[1].visible is False
    assert seq.metadata.volume_label == "CCTA"


def test_rotation_sequence_from_toml_roma():
    toml_content = """
[metadata]
coordinate_system = "LPS"
index_order = "roma"
angle_units = "radians"
timestamp = "2026-01-16T12:00:00"
volume_label = "Test"

[[angles_list]]
axis = "Z"
angle = -1.57
visible = true
name = ""
name_editable = true
deletable = true
"""
    seq = RotationSequence.from_toml(toml_content)

    assert len(seq.angles_list) == 1
    assert seq.angles_list[0].axis == "Z"
    assert np.isclose(seq.angles_list[0].angle, -1.57)


def test_rotation_sequence_from_toml_degrees():
    toml_content = """
[metadata]
coordinate_system = "LPS"
index_order = "itk"
angle_units = "degrees"
timestamp = "2026-01-16T12:00:00"
volume_label = "Test"

[[angles_list]]
axis = "Y"
angle = 90.0
visible = true
name = ""
name_editable = true
deletable = true
"""
    seq = RotationSequence.from_toml(toml_content)

    assert len(seq.angles_list) == 1
    assert seq.angles_list[0].axis == "Y"
    assert np.isclose(seq.angles_list[0].angle, 90.0)


def test_rotation_sequence_round_trip_itk():
    original_steps = [
        RotationStep(axis="X", angle=1.57, name="First", visible=True),
        RotationStep(axis="Y", angle=0.5, name="Second", visible=False),
        RotationStep(axis="Z", angle=-0.3, name="Third"),
    ]
    original = RotationSequence(angles_list=original_steps)
    original.metadata.volume_label = "CCTA"
    original.metadata.index_order = IndexOrder.ITK
    original.metadata.angle_units = AngleUnits.RADIANS

    toml_str = original.to_toml()
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.angles_list) == len(original.angles_list)
    for orig, rest in zip(original.angles_list, restored.angles_list):
        assert rest.axis == orig.axis
        assert np.isclose(rest.angle, orig.angle)
        assert rest.name == orig.name
        assert rest.visible == orig.visible


def test_rotation_sequence_round_trip_roma():
    original_steps = [
        RotationStep(axis="X", angle=1.57, name="First"),
        RotationStep(axis="Z", angle=-0.5, name="Second"),
    ]
    original = RotationSequence(angles_list=original_steps)
    original.metadata.volume_label = "Test"
    original.metadata.index_order = IndexOrder.ROMA
    original.metadata.angle_units = AngleUnits.RADIANS

    toml_str = original.to_toml()
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.angles_list) == len(original.angles_list)
    for orig, rest in zip(original.angles_list, restored.angles_list):
        assert rest.axis == orig.axis
        assert np.isclose(rest.angle, orig.angle)


def test_rotation_sequence_round_trip_degrees():
    original_steps = [
        RotationStep(axis="Y", angle=90.0),
        RotationStep(axis="X", angle=180.0),
    ]
    original = RotationSequence(angles_list=original_steps)
    original.metadata.index_order = IndexOrder.ITK
    original.metadata.angle_units = AngleUnits.DEGREES

    toml_str = original.to_toml()
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.angles_list) == len(original.angles_list)
    for orig, rest in zip(original.angles_list, restored.angles_list):
        assert rest.axis == orig.axis
        assert np.isclose(rest.angle, orig.angle, atol=1e-10)


def test_rotation_sequence_round_trip_roma_degrees():
    original_steps = [
        RotationStep(axis="X", angle=45.0),
        RotationStep(axis="Z", angle=-30.0),
    ]
    original = RotationSequence(angles_list=original_steps)
    original.metadata.index_order = IndexOrder.ROMA
    original.metadata.angle_units = AngleUnits.DEGREES

    toml_str = original.to_toml()
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.angles_list) == len(original.angles_list)
    for orig, rest in zip(original.angles_list, restored.angles_list):
        assert rest.axis == orig.axis
        assert np.isclose(rest.angle, orig.angle, atol=1e-10)


def test_rotation_sequence_mpr_origin_default():
    seq = RotationSequence()
    assert seq.mpr_origin == [0.0, 0.0, 0.0]


def test_rotation_sequence_mpr_origin_custom():
    seq = RotationSequence(mpr_origin=[10.0, 20.0, 30.0])
    assert seq.mpr_origin == [10.0, 20.0, 30.0]


def test_rotation_sequence_mpr_origin_validation():
    with pytest.raises(pc.ValidationError):
        RotationSequence(mpr_origin=[10.0, 20.0])  # Too few elements

    with pytest.raises(pc.ValidationError):
        RotationSequence(mpr_origin=[10.0, 20.0, 30.0, 40.0])  # Too many elements


def test_rotation_sequence_mpr_origin_in_toml():
    seq = RotationSequence(mpr_origin=[33.4, -188.9, -129.9])
    seq.metadata.volume_label = "Test"
    seq.metadata.index_order = IndexOrder.ITK

    toml_str = seq.to_toml()

    assert "mpr_origin" in toml_str
    assert "[33.4, -188.9, -129.9]" in toml_str


def test_rotation_sequence_mpr_origin_round_trip_itk():
    original = RotationSequence(mpr_origin=[10.0, 20.0, 30.0])
    original.metadata.volume_label = "Test"
    original.metadata.index_order = IndexOrder.ITK

    toml_str = original.to_toml()
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.mpr_origin) == 3
    assert np.isclose(restored.mpr_origin[0], 10.0)
    assert np.isclose(restored.mpr_origin[1], 20.0)
    assert np.isclose(restored.mpr_origin[2], 30.0)


def test_rotation_sequence_mpr_origin_round_trip_roma():
    original = RotationSequence(mpr_origin=[30.0, 20.0, 10.0])
    original.metadata.volume_label = "Test"
    original.metadata.index_order = IndexOrder.ROMA

    toml_str = original.to_toml()
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.mpr_origin) == 3
    assert np.isclose(restored.mpr_origin[0], 30.0)
    assert np.isclose(restored.mpr_origin[1], 20.0)
    assert np.isclose(restored.mpr_origin[2], 10.0)
