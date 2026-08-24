"""Test the ITK/ROMA boundary.

The invariant these guard: state is stored in the user's index order, VTK is
only ever handed ITK, and ``Convention`` is the only thing that crosses between.
"""

# Third Party
import numpy as np
import pytest

# Internal
from cardio.convention import (
    Convention,
    exchange_angle,
    exchange_axis,
    exchange_point,
    exchange_quaternion,
    exchange_step,
)
from cardio.orientation import (
    AngleUnits,
    IndexOrder,
    cumulative_rotation_matrix,
)

ITK = Convention(index_order=IndexOrder.ITK, angle_units=AngleUnits.DEGREES)
ROMA = Convention(index_order=IndexOrder.ROMA, angle_units=AngleUnits.DEGREES)

# The index exchange as a matrix: swapping the first and last axis.
EXCHANGE = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_exchanges_are_involutions():
    point = [1.0, 2.0, 3.0]
    assert exchange_point(exchange_point(point)) == point

    for axis in ("X", "Y", "Z"):
        assert exchange_axis(exchange_axis(axis)) == axis

    assert exchange_angle(exchange_angle(0.7)) == pytest.approx(0.7)

    quaternion = [0.1, 0.2, 0.3, 0.9]
    assert exchange_quaternion(exchange_quaternion(quaternion)) == pytest.approx(
        quaternion
    )


def test_exchange_step_is_an_involution():
    euler = {"axis": "X", "angle": 30.0, "visible": True}
    assert exchange_step(exchange_step(euler)) == euler

    quat = {"quaternion": [0.1, 0.2, 0.3, 0.9], "visible": True}
    round_tripped = exchange_step(exchange_step(quat))
    assert round_tripped["quaternion"] == pytest.approx(quat["quaternion"])


def test_itk_conversions_are_no_ops():
    assert ITK.point_to_itk([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert ITK.axis_to_itk("X") == "X"
    assert ITK.angle_to_itk(45.0) == 45.0
    assert ITK.quaternion_to_itk([0.1, 0.2, 0.3, 0.9]) == [0.1, 0.2, 0.3, 0.9]


def test_roma_conversions_apply_the_exchange():
    assert ROMA.point_to_itk([1.0, 2.0, 3.0]) == [3.0, 2.0, 1.0]
    assert ROMA.axis_to_itk("X") == "Z"
    assert ROMA.axis_to_itk("Y") == "Y"
    assert ROMA.angle_to_itk(45.0) == -45.0
    assert ROMA.quaternion_to_itk([0.1, 0.2, 0.3, 0.9]) == [-0.3, -0.2, -0.1, 0.9]


@pytest.mark.parametrize("convention", [ITK, ROMA])
def test_point_round_trips_through_itk(convention):
    point = [3.0, -4.0, 5.0]
    assert convention.point_from_itk(convention.point_to_itk(point)) == point


def legacy_sequence_to_itk(steps, index_order):
    """The conversion exactly as logic.py wrote it before convention.py existed."""
    sequence, angles = [], {}
    for index, step in enumerate(steps):
        if step.get("quaternion") is not None:
            sequence.append({"quaternion": step["quaternion"]})
            angles[index] = 0
        else:
            sequence.append({"axis": step["axis"]})
            angles[index] = step.get("angle", 0)

    if index_order != IndexOrder.ROMA:
        return sequence, angles

    converted_sequence, converted_angles = [], {}
    for index, step in enumerate(sequence):
        if step.get("quaternion") is not None:
            q = step["quaternion"]
            converted_sequence.append({"quaternion": [-q[2], -q[1], -q[0], q[3]]})
            converted_angles[index] = 0
        else:
            converted_sequence.append(
                {"axis": {"X": "Z", "Y": "Y", "Z": "X"}[step["axis"]]}
            )
            converted_angles[index] = -angles[index]
    return converted_sequence, converted_angles


@pytest.mark.parametrize("convention", [ITK, ROMA])
def test_sequence_to_itk_matches_the_pre_refactor_implementation(convention):
    steps = [
        {"axis": "X", "angle": 30.0},
        {"quaternion": [0.1, 0.2, 0.3, 0.9]},
        {"axis": "Z", "angle": -12.5},
        {"axis": "Y", "angle": 0},
    ]

    expected = legacy_sequence_to_itk(steps, convention.index_order)
    assert convention.sequence_to_itk(steps) == expected


def test_visible_sequence_skips_hidden_steps_and_reindexes():
    steps = [
        {"axis": "X", "angle": 10.0, "visible": True},
        {"axis": "Y", "angle": 20.0, "visible": False},
        {"axis": "Z", "angle": 30.0, "visible": True},
    ]

    sequence, angles = ITK.visible_sequence_to_itk(steps)

    assert sequence == [{"axis": "X"}, {"axis": "Z"}]
    assert angles == {0: 10.0, 1: 30.0}


def test_a_step_with_no_visible_flag_counts_as_visible():
    sequence, _ = ITK.visible_sequence_to_itk([{"axis": "X", "angle": 5.0}])
    assert sequence == [{"axis": "X"}]


@pytest.mark.parametrize(
    "steps",
    [
        [{"axis": "X", "angle": 30.0}],
        [{"axis": "Y", "angle": -45.0}, {"axis": "Z", "angle": 15.0}],
        [{"axis": "X", "angle": 12.0}, {"quaternion": [0.1, 0.2, 0.3, 0.9]}],
    ],
)
def test_roma_rotation_is_the_itk_rotation_conjugated_by_the_exchange(steps):
    """The property the whole boundary rests on.

    Re-expressing a rotation between the two orders is a change of basis by the
    exchange matrix. Because that matrix is a reflection, the rotation's sense
    flips -- which is why axes are permuted *and* angles negated together.
    """
    itk_sequence, itk_angles = ITK.sequence_to_itk(steps)
    roma_sequence, roma_angles = ROMA.sequence_to_itk(steps)

    as_itk = cumulative_rotation_matrix(itk_sequence, itk_angles, AngleUnits.DEGREES)
    as_roma = cumulative_rotation_matrix(roma_sequence, roma_angles, AngleUnits.DEGREES)

    np.testing.assert_allclose(as_roma, EXCHANGE @ as_itk @ EXCHANGE, atol=1e-12)


@pytest.mark.parametrize("view_normal", [[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
def test_scroll_vector_in_itk_then_back_equals_the_old_direct_computation(view_normal):
    """The UI used to reverse the base normal and compose unconverted steps.

    Going through ITK and converting the result back must give the same vector,
    or right-drag scrolling would move along the wrong axis under ROMA.
    """
    steps = [{"axis": "X", "angle": 25.0}, {"axis": "Z", "angle": -40.0}]
    normal = np.array(view_normal)

    sequence, angles = ROMA.sequence_to_itk(steps)
    via_itk = ROMA.point_from_itk(
        cumulative_rotation_matrix(sequence, angles, AngleUnits.DEGREES) @ normal
    )

    # The old path: compose the steps as stored, against a reversed base normal.
    stored_sequence = [{"axis": s["axis"]} for s in steps]
    stored_angles = {i: s["angle"] for i, s in enumerate(steps)}
    directly = (
        cumulative_rotation_matrix(stored_sequence, stored_angles, AngleUnits.DEGREES)
        @ normal[::-1]
    )

    np.testing.assert_allclose(via_itk, directly, atol=1e-12)


def test_from_metadata_reads_both_fields():
    class Metadata:
        index_order = IndexOrder.ROMA
        angle_units = AngleUnits.RADIANS

    convention = Convention.from_metadata(Metadata())

    assert convention.index_order == IndexOrder.ROMA
    assert convention.angle_units == AngleUnits.RADIANS
    assert not convention.is_itk
