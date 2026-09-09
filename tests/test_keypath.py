"""Naming a place inside a document key, and saying what moved there.

Plain dicts throughout: none of this knows about trame, a scene or a rotation,
and the point of the module is that it does not have to.
"""

# Third Party
import pytest

# Internal
import cardio.keypath as keypath

SEQUENCE = {
    "metadata": {"angle_units": "degrees", "index_order": "itk"},
    "angles_list": [
        {"axis": "Z", "angle": 0.0, "visible": True},
        {"axis": "X", "angle": 30.0, "visible": True},
    ],
}


# ---------------------------------------------------------------- the path ----


@pytest.mark.parametrize(
    "key,expected",
    [
        ("tile_cols", ("tile_cols", [])),
        ("mpr_rotation_data.metadata", ("mpr_rotation_data", ["metadata"])),
        (
            "mpr_rotation_data.angles_list.1.visible",
            ("mpr_rotation_data", ["angles_list", "1", "visible"]),
        ),
    ],
)
def test_a_path_is_a_key_and_the_way_in_from_it(key, expected):
    assert keypath.split(key) == expected


def test_reading_reaches_the_leaf():
    assert keypath.read(SEQUENCE, ["angles_list", "1", "angle"]) == 30.0


def test_reading_nothing_is_the_value_itself():
    assert keypath.read(SEQUENCE, []) is SEQUENCE


def test_a_negative_position_counts_from_the_end():
    assert keypath.read(SEQUENCE, ["angles_list", "-1", "axis"]) == "X"


# --------------------------------------------------------------- refusals ----


@pytest.mark.parametrize(
    "segments,expected",
    [
        (["nonsense"], "holds angles_list, metadata"),
        (["angles_list", "9", "axis"], "has 2 entries, so there is no 9"),
        (["angles_list", "middle"], "not a position in one"),
        (["metadata", "angle_units", "deeper"], "nothing is inside of"),
    ],
)
def test_a_path_that_leads_nowhere_says_what_was_there(segments, expected):
    with pytest.raises(ValueError, match=expected):
        keypath.read(SEQUENCE, segments, "mpr_rotation_data")


def test_a_refusal_names_how_far_it_got():
    with pytest.raises(ValueError, match=r"mpr_rotation_data\.angles_list"):
        keypath.read(SEQUENCE, ["angles_list", "9"], "mpr_rotation_data")


# --------------------------------------------------------------- the write ----


def test_writing_puts_the_value_where_it_was_asked_for():
    written = keypath.write(SEQUENCE, ["angles_list", "1", "visible"], False)

    assert written["angles_list"][1]["visible"] is False


def test_writing_leaves_the_original_alone():
    """The console holds the last value it saw; a write must not reach it."""
    keypath.write(SEQUENCE, ["angles_list", "1", "visible"], False)

    assert SEQUENCE["angles_list"][1]["visible"] is True


def test_writing_hands_back_a_different_object():
    """Which is how trame is told the key moved at all."""
    written = keypath.write(SEQUENCE, ["angles_list", "0", "angle"], 45.0)

    assert written is not SEQUENCE
    assert written["angles_list"] is not SEQUENCE["angles_list"]
    assert written["metadata"] is SEQUENCE["metadata"], "and copies only the way in"


def test_writing_nothing_is_the_value_itself():
    assert keypath.write(SEQUENCE, [], "replaced") == "replaced"


def test_what_was_written_is_what_reads_back():
    written = keypath.write(SEQUENCE, ["metadata", "angle_units"], "radians")

    assert keypath.read(written, ["metadata", "angle_units"]) == "radians"


# ------------------------------------------------------------- the changes ----


def test_nothing_moved_is_nothing_to_say():
    assert keypath.changes(SEQUENCE, SEQUENCE, "mpr_rotation_data") == []


def test_one_field_moved_is_one_path():
    after = keypath.write(SEQUENCE, ["angles_list", "1", "visible"], False)

    assert keypath.changes(SEQUENCE, after, "mpr_rotation_data") == [
        ("mpr_rotation_data.angles_list.1.visible", False)
    ]


def test_two_fields_moved_are_two_paths():
    after = keypath.write(SEQUENCE, ["angles_list", "0", "angle"], 45.0)
    after = keypath.write(after, ["metadata", "angle_units"], "radians")

    assert sorted(keypath.changes(SEQUENCE, after)) == [
        ("angles_list.0.angle", 45.0),
        ("metadata.angle_units", "radians"),
    ]


def test_a_flat_list_of_scalars_is_one_value_and_not_two():
    """A range slider holds a pair; a drag moved the pair, not either end."""
    assert keypath.changes([12.0, 345.0], [12.0, 400.0], "clip_depth") == [
        ("clip_depth", [12.0, 400.0])
    ]


def test_a_scalar_is_itself():
    assert keypath.changes(3, 4, "tile_cols") == [("tile_cols", 4)]


def test_a_list_that_changed_length_is_reported_whole():
    """There is no saying which entry a shorter list is missing."""
    after = {**SEQUENCE, "angles_list": SEQUENCE["angles_list"][:1]}

    assert keypath.changes(SEQUENCE, after, "mpr_rotation_data") == [
        ("mpr_rotation_data.angles_list", after["angles_list"])
    ]


def test_a_dict_that_gained_a_key_is_reported_whole():
    after = {**SEQUENCE["metadata"], "timestamp": "now"}

    assert keypath.changes(SEQUENCE["metadata"], after, "metadata") == [
        ("metadata", after)
    ]


def test_a_value_that_changed_shape_is_reported_whole():
    assert keypath.changes({"a": 1}, [1, 2], "key") == [("key", [1, 2])]


def test_every_change_is_a_path_that_reads_back():
    """Which is the whole of what makes one of these a line worth writing."""
    after = keypath.write(SEQUENCE, ["angles_list", "1", "angle"], 90.0)

    for path, value in keypath.changes(SEQUENCE, after, "mpr_rotation_data"):
        name, segments = keypath.split(path)
        assert name == "mpr_rotation_data"
        assert keypath.read(after, segments) == value
