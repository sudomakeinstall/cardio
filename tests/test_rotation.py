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


def test_rotation_step_quaternion_creation():
    q = [0.0, -0.2588, 0.0, 0.9659]
    step = RotationStep(quaternion=q, name="vla", name_editable=False, deletable=False)
    assert step.quaternion == q
    assert step.axis is None
    assert step.angle is None
    assert step.name == "vla"


def test_rotation_step_quaternion_requires_4_elements():
    with pytest.raises(pc.ValidationError):
        RotationStep(quaternion=[1.0, 0.0, 0.0])


def test_rotation_step_rejects_both_forms():
    with pytest.raises(pc.ValidationError):
        RotationStep(axis="X", angle=0.5, quaternion=[0.0, 0.0, 0.0, 1.0])


def test_rotation_step_rejects_neither_form():
    with pytest.raises(pc.ValidationError):
        RotationStep()


def test_rotation_step_euler_to_rotation_matrix():
    from cardio.orientation import AngleUnits

    step = RotationStep(axis="Z", angle=90.0)
    mat = step.to_rotation_matrix(AngleUnits.DEGREES)
    assert mat.shape == (3, 3)
    assert np.isclose(mat[0, 1], -1.0, atol=1e-10)
    assert np.isclose(mat[1, 0], 1.0, atol=1e-10)


def test_rotation_step_quaternion_to_rotation_matrix():
    from cardio.orientation import AngleUnits

    # Identity quaternion [x, y, z, w] = [0, 0, 0, 1]
    step = RotationStep(quaternion=[0.0, 0.0, 0.0, 1.0])
    mat = step.to_rotation_matrix(AngleUnits.RADIANS)
    assert mat.shape == (3, 3)
    assert np.allclose(mat, np.eye(3))


def test_rotation_sequence_with_quaternion_step():
    steps = [
        RotationStep(axis="X", angle=0.5),
        RotationStep(quaternion=[0.0, 0.0, 0.0, 1.0], name="vla"),
    ]
    seq = RotationSequence(angles_list=steps)
    assert len(seq.angles_list) == 2
    assert seq.angles_list[1].quaternion == [0.0, 0.0, 0.0, 1.0]


def test_rotation_metadata_creation():
    meta = RotationMetadata(volume_label="CCTA")
    assert meta.coordinate_system == "LPS"
    assert meta.index_order == IndexOrder.ITK
    assert meta.angle_units == AngleUnits.RADIANS
    assert meta.volume_label == "CCTA"


def test_rotation_metadata_invalid_coordinate_system():
    with pytest.raises(pc.ValidationError):
        RotationMetadata(coordinate_system="RAS")


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


# --- TOML serialization ------------------------------------------------------
#
# to_toml and from_toml store and load the sequence as it stands, converting
# nothing, so neither the index order nor the angle units change the path taken.
# These cover that one path rather than repeating it per convention and unit.


@pytest.mark.parametrize(
    "index_order,angle_units",
    [(IndexOrder.ITK, AngleUnits.RADIANS), (IndexOrder.ROMA, AngleUnits.DEGREES)],
)
def test_to_toml_writes_the_steps_and_the_metadata_it_holds(index_order, angle_units):
    seq = RotationSequence(
        metadata=RotationMetadata(
            index_order=index_order, angle_units=angle_units, volume_label="CCTA"
        ),
        angles_list=[RotationStep(axis="X", angle=1.57, name="Rotate X")],
    )

    toml_str = seq.to_toml()

    assert 'axis = "X"' in toml_str
    assert "angle = 1.57" in toml_str
    assert 'name = "Rotate X"' in toml_str
    assert f'index_order = "{index_order.value}"' in toml_str
    assert f'angle_units = "{angle_units.value}"' in toml_str
    assert 'volume_label = "CCTA"' in toml_str


def test_from_toml_reads_a_hand_written_file():
    """The format is a saved artefact, so it is pinned as literal text."""
    seq = RotationSequence.from_toml("""
mpr_origin = [10.0, 20.0, 30.0]

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
""")

    assert [step.axis for step in seq.angles_list] == ["X", "Y"]
    assert [step.angle for step in seq.angles_list] == [1.57, 0.5]
    assert [step.name for step in seq.angles_list] == ["First", "Second"]
    assert [step.visible for step in seq.angles_list] == [True, False]
    assert seq.metadata.volume_label == "CCTA"
    assert seq.mpr_origin == [10.0, 20.0, 30.0]


def test_a_sequence_round_trips_through_toml():
    """Both step forms, the flags that ride along, and the origin."""
    quaternion = [0.0, -0.2588, 0.0, 0.9659]
    original = RotationSequence(
        metadata=RotationMetadata(volume_label="CCTA"),
        angles_list=[
            RotationStep(axis="X", angle=1.57, name="euler", visible=False),
            RotationStep(
                quaternion=quaternion, name="vla", name_editable=False, deletable=False
            ),
        ],
        mpr_origin=[10.0, 20.0, 30.0],
    )

    restored = RotationSequence.from_toml(original.to_toml())

    euler, vla = restored.angles_list
    assert euler.axis == "X"
    assert euler.angle == pytest.approx(1.57)
    assert euler.visible is False
    assert vla.quaternion == pytest.approx(quaternion)
    assert vla.name == "vla"
    assert vla.name_editable is False
    assert vla.deletable is False
    assert restored.mpr_origin == pytest.approx([10.0, 20.0, 30.0])
    assert restored.metadata.volume_label == "CCTA"


def test_rotation_sequence_version_comment_in_toml():
    """Test that cardio version comment is included when serializing."""
    from cardio import __version__

    seq = RotationSequence()
    seq.metadata.volume_label = "Test"

    toml_str = seq.to_toml()

    assert f"# Generated by cardio version {__version__}" in toml_str
    assert toml_str.startswith("# Generated by cardio version")


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


# --- convention and unit conversion on the model -----------------------------


def test_with_units_converts_euler_angles_and_leaves_quaternions():
    seq = RotationSequence(
        metadata=RotationMetadata(angle_units=AngleUnits.DEGREES),
        angles_list=[
            RotationStep(axis="X", angle=180.0),
            RotationStep(quaternion=[0.0, 0.0, 0.0, 1.0]),
        ],
    )

    converted = seq.with_units(AngleUnits.RADIANS)

    assert converted.metadata.angle_units == AngleUnits.RADIANS
    assert converted.angles_list[0].angle == pytest.approx(np.pi)
    assert converted.angles_list[1].quaternion == [0.0, 0.0, 0.0, 1.0]
    # the original is untouched
    assert seq.angles_list[0].angle == 180.0


def test_with_units_is_a_no_op_for_the_same_units():
    seq = RotationSequence(metadata=RotationMetadata(angle_units=AngleUnits.RADIANS))
    assert seq.with_units(AngleUnits.RADIANS) is seq


def test_with_units_round_trips():
    seq = RotationSequence(
        metadata=RotationMetadata(angle_units=AngleUnits.DEGREES),
        angles_list=[RotationStep(axis="Y", angle=37.5)],
    )

    restored = seq.with_units(AngleUnits.RADIANS).with_units(AngleUnits.DEGREES)

    assert restored.angles_list[0].angle == pytest.approx(37.5)
    assert restored.metadata.angle_units == AngleUnits.DEGREES


def test_with_index_order_exchanges_steps_and_origin():
    seq = RotationSequence(
        metadata=RotationMetadata(index_order=IndexOrder.ITK),
        angles_list=[
            RotationStep(axis="X", angle=30.0),
            RotationStep(quaternion=[0.1, 0.2, 0.3, 0.9]),
        ],
        mpr_origin=[1.0, 2.0, 3.0],
    )

    converted = seq.with_index_order(IndexOrder.ROMA)

    assert converted.metadata.index_order == IndexOrder.ROMA
    assert converted.angles_list[0].axis == "Z"
    assert converted.angles_list[0].angle == -30.0
    assert converted.angles_list[1].quaternion == pytest.approx([-0.3, -0.2, -0.1, 0.9])
    assert converted.mpr_origin == [3.0, 2.0, 1.0]


def test_with_index_order_round_trips():
    seq = RotationSequence(
        angles_list=[
            RotationStep(axis="X", angle=30.0),
            RotationStep(quaternion=[0.1, 0.2, 0.3, 0.9]),
        ],
        mpr_origin=[1.0, 2.0, 3.0],
    )

    restored = seq.with_index_order(IndexOrder.ROMA).with_index_order(IndexOrder.ITK)

    assert restored.angles_list[0].axis == "X"
    assert restored.angles_list[0].angle == pytest.approx(30.0)
    assert restored.angles_list[1].quaternion == pytest.approx([0.1, 0.2, 0.3, 0.9])
    assert restored.mpr_origin == [1.0, 2.0, 3.0]
    assert restored.metadata.index_order == IndexOrder.ITK
