import numpy as np
import pydantic as pc
import pytest

from cardio.orientation import AngleUnits, AxisConvention
from cardio.rotation import RotationMetadata, RotationSequence, RotationStep


def test_rotation_step_creation():
    step = RotationStep(axes="X", angle=1.57)
    assert step.axes == "X"
    assert step.angle == 1.57
    assert step.visible is True
    assert step.name == ""
    assert step.name_editable is True
    assert step.deletable is True


def test_rotation_step_legacy_format():
    data = {"axes": "Y", "angles": [0.5], "visible": False}
    step = RotationStep(**data)
    assert step.axes == "Y"
    assert step.angle == 0.5
    assert step.visible is False


def test_rotation_step_legacy_axis_field():
    data = {"axis": "Z", "angle": 1.0}
    step = RotationStep(**data)
    assert step.axes == "Z"
    assert step.angle == 1.0


def test_rotation_step_to_convention_itk():
    step = RotationStep(axes="X", angle=1.57)
    axis, angle = step.to_convention(AxisConvention.ITK, AngleUnits.RADIANS)
    assert axis == "X"
    assert angle == 1.57


def test_rotation_step_to_convention_roma():
    step = RotationStep(axes="X", angle=1.57)
    axis, angle = step.to_convention(AxisConvention.ROMA, AngleUnits.RADIANS)
    assert axis == "Z"
    assert angle == -1.57


def test_rotation_step_to_convention_degrees():
    step = RotationStep(axes="Y", angle=np.pi)
    axis, angle = step.to_convention(AxisConvention.ITK, AngleUnits.DEGREES)
    assert axis == "Y"
    assert np.isclose(angle, 180.0)


def test_rotation_step_to_convention_roma_degrees():
    step = RotationStep(axes="X", angle=np.pi / 2)
    axis, angle = step.to_convention(AxisConvention.ROMA, AngleUnits.DEGREES)
    assert axis == "Z"
    assert np.isclose(angle, -90.0)


def test_rotation_step_from_convention_itk():
    step = RotationStep.from_convention(
        axes="X", angle=1.57, convention=AxisConvention.ITK, units=AngleUnits.RADIANS
    )
    assert step.axes == "X"
    assert step.angle == 1.57


def test_rotation_step_from_convention_roma():
    step = RotationStep.from_convention(
        axes="Z", angle=-1.57, convention=AxisConvention.ROMA, units=AngleUnits.RADIANS
    )
    assert step.axes == "X"
    assert step.angle == 1.57


def test_rotation_step_from_convention_degrees():
    step = RotationStep.from_convention(
        axes="Y", angle=90.0, convention=AxisConvention.ITK, units=AngleUnits.DEGREES
    )
    assert step.axes == "Y"
    assert np.isclose(step.angle, np.pi / 2)


def test_rotation_step_round_trip_itk_to_roma():
    original = RotationStep(axes="X", angle=1.57, name="test")
    axis, angle = original.to_convention(AxisConvention.ROMA, AngleUnits.RADIANS)
    restored = RotationStep.from_convention(
        axes=axis,
        angle=angle,
        convention=AxisConvention.ROMA,
        units=AngleUnits.RADIANS,
        name="test",
    )
    assert restored.axes == original.axes
    assert np.isclose(restored.angle, original.angle)
    assert restored.name == original.name


def test_rotation_step_round_trip_all_axes():
    for axis in ["X", "Y", "Z"]:
        original = RotationStep(axes=axis, angle=1.0)
        conv_axis, conv_angle = original.to_convention(
            AxisConvention.ROMA, AngleUnits.RADIANS
        )
        restored = RotationStep.from_convention(
            axes=conv_axis,
            angle=conv_angle,
            convention=AxisConvention.ROMA,
            units=AngleUnits.RADIANS,
        )
        assert restored.axes == original.axes
        assert np.isclose(restored.angle, original.angle)


def test_rotation_metadata_creation():
    meta = RotationMetadata(volume_label="CCTA")
    assert meta.coordinate_system == "LPS"
    assert meta.axis_convention == AxisConvention.ITK
    assert meta.units == AngleUnits.RADIANS
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
        RotationStep(axes="X", angle=0.5),
        RotationStep(axes="Y", angle=1.0),
    ]
    seq = RotationSequence(angles_list=steps)
    assert len(seq.angles_list) == 2
    assert seq.angles_list[0].axes == "X"
    assert seq.angles_list[1].axes == "Y"


def test_rotation_sequence_to_dict_for_ui():
    steps = [
        RotationStep(axes="X", angle=0.5, name="First"),
        RotationStep(axes="Y", angle=1.0, name="Second", visible=False),
    ]
    seq = RotationSequence(angles_list=steps)
    ui_dict = seq.to_dict_for_ui()

    assert "angles_list" in ui_dict
    assert len(ui_dict["angles_list"]) == 2
    assert ui_dict["angles_list"][0]["axes"] == "X"
    assert ui_dict["angles_list"][0]["angles"] == [0.5]
    assert ui_dict["angles_list"][0]["name"] == "First"
    assert ui_dict["angles_list"][1]["visible"] is False


def test_rotation_sequence_from_ui_dict():
    ui_data = {
        "angles_list": [
            {"axes": "X", "angles": [0.5], "name": "Test"},
            {"axes": "Y", "angles": [1.0]},
        ]
    }
    seq = RotationSequence.from_ui_dict(ui_data, volume_label="CCTA")

    assert len(seq.angles_list) == 2
    assert seq.angles_list[0].axes == "X"
    assert seq.angles_list[0].angle == 0.5
    assert seq.metadata.volume_label == "CCTA"


def test_rotation_sequence_to_toml_itk():
    steps = [RotationStep(axes="X", angle=1.57, name="Rotate X")]
    seq = RotationSequence(angles_list=steps)
    seq.metadata.volume_label = "CCTA"

    toml_str = seq.to_toml(AxisConvention.ITK, AngleUnits.RADIANS)

    assert 'axes = "X"' in toml_str
    assert "angle = 1.57" in toml_str
    assert 'axis_convention = "itk"' in toml_str
    assert 'units = "radians"' in toml_str
    assert 'volume_label = "CCTA"' in toml_str


def test_rotation_sequence_to_toml_roma():
    steps = [RotationStep(axes="X", angle=1.57, name="Rotate X")]
    seq = RotationSequence(angles_list=steps)

    toml_str = seq.to_toml(AxisConvention.ROMA, AngleUnits.RADIANS)

    assert 'axes = "Z"' in toml_str
    assert "angle = -1.57" in toml_str
    assert 'axis_convention = "roma"' in toml_str


def test_rotation_sequence_to_toml_degrees():
    steps = [RotationStep(axes="Y", angle=np.pi / 2)]
    seq = RotationSequence(angles_list=steps)

    toml_str = seq.to_toml(AxisConvention.ITK, AngleUnits.DEGREES)

    assert 'units = "degrees"' in toml_str
    assert "90.0" in toml_str


def test_rotation_sequence_from_toml_itk():
    toml_content = """
[metadata]
coordinate_system = "LPS"
axis_convention = "itk"
units = "radians"
timestamp = "2026-01-16T12:00:00"
volume_label = "CCTA"

[[angles_list]]
axes = "X"
angle = 1.57
visible = true
name = "First"
name_editable = true
deletable = true

[[angles_list]]
axes = "Y"
angle = 0.5
visible = false
name = "Second"
name_editable = true
deletable = true
"""
    seq = RotationSequence.from_toml(toml_content)

    assert len(seq.angles_list) == 2
    assert seq.angles_list[0].axes == "X"
    assert seq.angles_list[0].angle == 1.57
    assert seq.angles_list[0].name == "First"
    assert seq.angles_list[1].axes == "Y"
    assert seq.angles_list[1].angle == 0.5
    assert seq.angles_list[1].visible is False
    assert seq.metadata.volume_label == "CCTA"


def test_rotation_sequence_from_toml_roma():
    toml_content = """
[metadata]
coordinate_system = "LPS"
axis_convention = "roma"
units = "radians"
timestamp = "2026-01-16T12:00:00"
volume_label = "Test"

[[angles_list]]
axes = "Z"
angle = -1.57
visible = true
name = ""
name_editable = true
deletable = true
"""
    seq = RotationSequence.from_toml(toml_content)

    assert len(seq.angles_list) == 1
    assert seq.angles_list[0].axes == "X"
    assert np.isclose(seq.angles_list[0].angle, 1.57)


def test_rotation_sequence_from_toml_degrees():
    toml_content = """
[metadata]
coordinate_system = "LPS"
axis_convention = "itk"
units = "degrees"
timestamp = "2026-01-16T12:00:00"
volume_label = "Test"

[[angles_list]]
axes = "Y"
angle = 90.0
visible = true
name = ""
name_editable = true
deletable = true
"""
    seq = RotationSequence.from_toml(toml_content)

    assert len(seq.angles_list) == 1
    assert seq.angles_list[0].axes == "Y"
    assert np.isclose(seq.angles_list[0].angle, np.pi / 2)


def test_rotation_sequence_round_trip_itk():
    original_steps = [
        RotationStep(axes="X", angle=1.57, name="First", visible=True),
        RotationStep(axes="Y", angle=0.5, name="Second", visible=False),
        RotationStep(axes="Z", angle=-0.3, name="Third"),
    ]
    original = RotationSequence(angles_list=original_steps)
    original.metadata.volume_label = "CCTA"

    toml_str = original.to_toml(AxisConvention.ITK, AngleUnits.RADIANS)
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.angles_list) == len(original.angles_list)
    for orig, rest in zip(original.angles_list, restored.angles_list):
        assert rest.axes == orig.axes
        assert np.isclose(rest.angle, orig.angle)
        assert rest.name == orig.name
        assert rest.visible == orig.visible


def test_rotation_sequence_round_trip_roma():
    original_steps = [
        RotationStep(axes="X", angle=1.57, name="First"),
        RotationStep(axes="Z", angle=-0.5, name="Second"),
    ]
    original = RotationSequence(angles_list=original_steps)
    original.metadata.volume_label = "Test"

    toml_str = original.to_toml(AxisConvention.ROMA, AngleUnits.RADIANS)
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.angles_list) == len(original.angles_list)
    for orig, rest in zip(original.angles_list, restored.angles_list):
        assert rest.axes == orig.axes
        assert np.isclose(rest.angle, orig.angle)


def test_rotation_sequence_round_trip_degrees():
    original_steps = [
        RotationStep(axes="Y", angle=np.pi / 2),
        RotationStep(axes="X", angle=np.pi),
    ]
    original = RotationSequence(angles_list=original_steps)

    toml_str = original.to_toml(AxisConvention.ITK, AngleUnits.DEGREES)
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.angles_list) == len(original.angles_list)
    for orig, rest in zip(original.angles_list, restored.angles_list):
        assert rest.axes == orig.axes
        assert np.isclose(rest.angle, orig.angle, atol=1e-10)


def test_rotation_sequence_round_trip_roma_degrees():
    original_steps = [
        RotationStep(axes="X", angle=np.pi / 4),
        RotationStep(axes="Z", angle=-np.pi / 6),
    ]
    original = RotationSequence(angles_list=original_steps)

    toml_str = original.to_toml(AxisConvention.ROMA, AngleUnits.DEGREES)
    restored = RotationSequence.from_toml(toml_str)

    assert len(restored.angles_list) == len(original.angles_list)
    for orig, rest in zip(original.angles_list, restored.angles_list):
        assert rest.axes == orig.axes
        assert np.isclose(rest.angle, orig.angle, atol=1e-10)


def test_rotation_sequence_legacy_list_format():
    legacy_data = [
        {"axes": "X", "angles": [1.57], "visible": True},
        {"axis": "Y", "angle": 0.5, "visible": False},
    ]
    seq = RotationSequence(angles_list=legacy_data)

    assert len(seq.angles_list) == 2
    assert seq.angles_list[0].axes == "X"
    assert seq.angles_list[0].angle == 1.57
    assert seq.angles_list[1].axes == "Y"
    assert seq.angles_list[1].angle == 0.5
