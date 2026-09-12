"""What a segmentation measures, and in what units.

The gap this guards: the measurement used to live in a script whose label
groups, densities and body-size formula were all literals, so nothing could
say whether a volume was millilitres or cubic millimetres, whether an
anisotropic voxel was measured by its own spacing, or whether a body surface
area had been given a height in the units its formula wanted.  Each of those
is one wrong constant away from a number that looks entirely plausible.
"""

# Third Party
import numpy as np
import pydantic as pc
import pytest

# Internal
from cardio.utils import label_color
from cardio.volumetry import (
    OPTIONAL_FIELDS,
    STRUCTURES,
    BSAFormula,
    StructureGroup,
    Volumetry,
    body_size,
    body_surface_area,
    bsa_mosteller,
    group_amount,
    measure,
    metrics,
    page_table,
    structure_rows,
    structure_state,
    usable_groups,
    voxel_volume_ml,
)
from tests.phantoms import write_segmentation

# A block of label 1, a block of label 2 and a block of label 3, each a whole
# number of voxels, so what every measurement below should come to can be
# written down rather than recorded.
BLOCK = (4, 5, 6)
COUNT = BLOCK[0] * BLOCK[1] * BLOCK[2]


def block_array(labels=(1, 2, 3)) -> np.ndarray:
    """Three separated blocks of ``COUNT`` voxels each, as a (k, j, i) array."""
    array = np.zeros((24, 16, 16), dtype=np.uint8)
    for index, label in enumerate(labels):
        k = 2 + index * 7
        array[k : k + BLOCK[2], 2 : 2 + BLOCK[1], 2 : 2 + BLOCK[0]] = label
    return array


def counts(**by_label) -> dict[int, int]:
    return {int(label): count for label, count in by_label.items()}


# --- what a voxel stands for --------------------------------------------------


def test_a_voxel_is_measured_in_millilitres_rather_than_cubic_millimetres(tmp_path):
    """Image spacing is millimetres; a millilitre is a thousand cubic of them.

    Dropping the factor of a thousand gives a left ventricle of 120 000, which
    is wrong in a way no plausibility check would catch.
    """
    segmentation = write_segmentation(
        tmp_path, [block_array()], label="s", spacing=(1.0, 1.0, 1.0)
    )
    assert voxel_volume_ml(segmentation._label_images[0]) == pytest.approx(0.001)


def test_an_anisotropic_voxel_is_measured_by_its_own_spacing(tmp_path):
    """Spacing is a product of three, not a cube of one.

    Thick-sliced cine is the normal case, so a measurement that assumed
    isotropy would be wrong on almost every real series.
    """
    segmentation = write_segmentation(
        tmp_path, [block_array()], label="s", spacing=(1.0, 2.0, 3.0)
    )
    assert voxel_volume_ml(segmentation._label_images[0]) == pytest.approx(0.006)


# --- what a group comes to ----------------------------------------------------


def test_a_group_measures_the_voxels_carrying_any_of_its_labels():
    """The sum is the point: one chamber may be segmented as several labels."""
    group = StructureGroup(name="LV", labels=[1, 3])
    amount = group_amount(counts(**{"1": 10, "2": 99, "3": 5}), group, 0.001)
    assert amount == pytest.approx(0.015)


def test_a_group_naming_a_label_the_frame_lacks_measures_the_rest_of_it():
    """A sub-label can empty between phases without the chamber disappearing."""
    group = StructureGroup(name="LV", labels=[1, 7])
    assert group_amount(counts(**{"1": 10}), group, 0.001) == pytest.approx(0.010)


def test_a_group_with_a_density_is_reported_as_a_mass():
    """Myocardium is quoted in grams, which is millilitres times its density."""
    group = StructureGroup(name="Myo", labels=[1], density=1.05)
    assert group_amount(counts(**{"1": 1000}), group, 0.001) == pytest.approx(1.05)
    assert group.unit == "g"


def test_a_group_without_a_density_is_reported_as_a_volume():
    assert StructureGroup(name="LV", labels=[1]).unit == "mL"


def test_a_group_that_names_no_colour_is_drawn_in_its_first_labels_own_colour():
    """So an unconfigured curve matches the overlay the reader already knows."""
    group = StructureGroup(name="LV", labels=[6, 3, 8])
    assert group.rgb == label_color(6)


def test_a_group_that_names_a_colour_keeps_it():
    group = StructureGroup(name="LV", labels=[6], color=(0.0, 0.5, 1.0))
    assert group.rgb == (0.0, 0.5, 1.0)


# --- what a curve comes to ----------------------------------------------------


def test_the_extremes_are_the_largest_and_smallest_the_curve_reaches():
    result = metrics(np.array([100.0, 140.0, 60.0, 90.0]))
    assert result.maximum == pytest.approx(140.0)
    assert result.minimum == pytest.approx(60.0)


def test_the_extremes_say_which_frame_they_fell_on():
    """The chart marks them, so the frame is part of the measurement."""
    result = metrics(np.array([100.0, 140.0, 60.0, 90.0]))
    assert (result.maximum_frame, result.minimum_frame) == (1, 2)


def test_the_ejection_fraction_is_the_stroke_volume_over_the_largest():
    result = metrics(np.array([120.0, 48.0]))
    assert result.stroke == pytest.approx(72.0)
    assert result.ejection == pytest.approx(60.0)


def test_a_structure_that_does_not_pump_reports_no_stroke_or_ejection():
    """Myocardium changes shape rather than filling and emptying.

    The difference between its extremes is not a stroke volume, and quoting
    one invites a reader to treat it as flow.
    """
    result = metrics(np.array([140.0, 150.0]), chamber=False)
    assert result.stroke is None
    assert result.ejection is None


def test_a_curve_that_never_leaves_zero_has_no_ejection_fraction():
    """Nothing divided by nothing is not a hundred per cent."""
    assert metrics(np.zeros(4)).ejection is None


# --- body surface area --------------------------------------------------------


def test_mostellers_area_is_computed_from_a_height_given_in_metres():
    """DICOM PatientSize is metres and Mosteller is written in centimetres.

    The prototype passed the tag through unconverted, which is an area out by
    a factor of ten and so an indexed volume out by the same.
    """
    assert bsa_mosteller(1.78, 74.0) == pytest.approx(1.9128, abs=1e-4)


def test_every_formula_agrees_to_within_a_tenth_of_a_square_metre():
    """They are different fits to the same quantity, not different quantities."""
    areas = [
        body_surface_area(formula, 1.78, 74.0)
        for formula in BSAFormula
        if formula is not BSAFormula.NONE
    ]
    assert max(areas) - min(areas) < 0.1


def test_no_area_is_computed_when_a_measurement_is_missing():
    """None rather than a default: an indexed volume against somebody else's
    body is worse than no indexed volume at all."""
    assert body_surface_area(BSAFormula.MOSTELLER, None, 74.0) is None
    assert body_surface_area(BSAFormula.MOSTELLER, 1.78, None) is None
    assert body_surface_area(BSAFormula.NONE, 1.78, 74.0) is None


def test_a_body_size_is_read_off_the_header_the_images_arrived_with():
    height, weight = body_size({"PatientSize": "1.78", "PatientWeight": "74.0"})
    assert (height, weight) == (pytest.approx(1.78), pytest.approx(74.0))


@pytest.mark.parametrize(
    "header",
    [{}, {"PatientSize": ""}, {"PatientSize": "unknown"}, {"PatientSize": "0"}],
)
def test_a_header_that_says_nothing_usable_about_a_body_yields_nothing(header):
    """A header is whatever the writer put in it, which includes rubbish."""
    assert body_size(header)[0] is None


# --- the whole measurement ----------------------------------------------------


def test_a_series_is_measured_frame_by_frame(tmp_path):
    """The curve is the point; a single number would not need a chart."""
    segmentation = write_segmentation(
        tmp_path,
        [block_array(), block_array(labels=(1, 1, 1))],
        label="s",
        spacing=(1.0, 1.0, 1.0),
    )
    result = measure(
        [segmentation.label_counts(frame) for frame in range(2)],
        voxel_volume_ml(segmentation._label_images[0]),
        [StructureGroup(name="one", labels=[1])],
    )

    assert result.frames == 2
    assert result.of("one").values == pytest.approx([COUNT * 0.001, 3 * COUNT * 0.001])


def test_a_group_naming_a_label_no_frame_carries_measures_zero_and_says_so(
    tmp_path, caplog
):
    """A mistyped label id measures zero, which in silence looks like a finding."""
    segmentation = write_segmentation(
        tmp_path, [block_array()], label="s", spacing=(1.0, 1.0, 1.0)
    )
    with caplog.at_level("WARNING"):
        result = measure(
            [segmentation.label_counts(0)],
            voxel_volume_ml(segmentation._label_images[0]),
            [StructureGroup(name="typo", labels=[9])],
        )

    assert result.of("typo").values == pytest.approx([0.0])
    assert "does not carry" in caplog.text


def test_the_measurement_carries_the_area_it_was_given_to_index_by():
    result = measure([{1: 10}], 0.001, [StructureGroup(name="a", labels=[1])], bsa=1.9)
    assert result.bsa == pytest.approx(1.9)


# --- the counts the measurement reads -----------------------------------------


def test_the_labels_of_a_frame_are_counted_once_however_often_they_are_asked_for(
    tmp_path,
):
    """A curve is re-read whenever the structures change; a 4D label series is
    not something to histogram again at that rate."""
    segmentation = write_segmentation(tmp_path, [block_array()], label="s")
    assert segmentation.label_counts(0) is segmentation.label_counts(0)


def test_the_labels_present_are_still_the_ones_the_counts_name(tmp_path):
    """``get_labels`` now reads the histogram rather than building its own.

    The snap, zoom and tile pickers are all populated from it, so a refactor
    that changed what it returned would empty three panels at once.
    """
    segmentation = write_segmentation(tmp_path, [block_array()], label="s")
    assert segmentation.get_labels(0) == [1, 2, 3]
    assert segmentation.label_counts(0) == {1: COUNT, 2: COUNT, 3: COUNT}


def test_a_frame_past_the_end_of_a_short_series_is_counted_as_the_first(tmp_path):
    """The same wrapping every other per-frame reader does."""
    segmentation = write_segmentation(tmp_path, [block_array()], label="s")
    assert segmentation.label_counts(7) == segmentation.label_counts(0)


# --- the configuration --------------------------------------------------------


def test_two_groups_by_one_name_are_refused():
    """Two curves nothing could tell apart, in a chart whose cells are named."""
    with pytest.raises(pc.ValidationError):
        Volumetry(
            groups=[
                StructureGroup(name="LV", labels=[1]),
                StructureGroup(name="LV", labels=[2]),
            ]
        )


def test_a_group_must_name_at_least_one_label():
    """Otherwise it is a curve of zeroes with no colour to draw it in."""
    with pytest.raises(pc.ValidationError):
        StructureGroup(name="LV", labels=[])


@pytest.mark.parametrize("density", [0, -1])
def test_a_config_that_names_an_unusable_density_still_refuses_to_open(density):
    """Only the form seam is lenient.  A hand-written config saying density=0
    is a file to correct, and correcting it is easier than wondering why the
    myocardium came out in millilitres."""
    with pytest.raises(pc.ValidationError):
        StructureGroup(name="Myo", labels=[1], density=density)


def test_a_misspelled_volumetry_key_says_so():
    with pytest.raises(pc.ValidationError):
        Volumetry(colums=2)


# --- what a page calls the extremes -------------------------------------------
#
# The cardiac vocabulary is a reader's claim, not the measurement's: nothing
# here knows what phase anything was acquired at, or that the series covered a
# cycle at all. Declaring a structure a pumping chamber is what licenses it.


def paged(chamber: bool):
    """One structure's page rows, off a curve that rises and falls."""
    group = StructureGroup(name="X", labels=[1], chamber=chamber)
    result = measure([{1: 100}, {1: 60}], 1.0, [group])
    return [row[0] for row in page_table(result.measurements[0])[1]]


def test_a_declared_chamber_is_read_in_the_cardiac_vocabulary():
    assert paged(chamber=True) == ["EDV", "ESV", "SV", "EF"]


def test_anything_else_is_read_as_a_maximum_and_a_minimum():
    """An aorta pulses and a lung moves; neither has an end-diastolic volume,
    and the great vessel is the case that shows why the flag is not "cardiac"
    -- the aorta is cardiac anatomy and still has no ejection fraction."""
    assert paged(chamber=False) == ["Maximum", "Minimum"]


# --- the rows a form leaves behind --------------------------------------------
#
# The drawer holds a structure while it is being filled in, which the models
# refuse and are right to: a config naming no labels is a config to correct.
# What stands between the two is usable_groups, and every rule it applies is a
# rule the drawer would otherwise break -- or, worse, a Save Session that
# raises on the way out.


def test_a_structure_with_no_labels_yet_is_not_one():
    assert usable_groups([{"name": "LV", "labels": []}]) == []


def test_a_structure_with_no_name_yet_is_not_one():
    assert usable_groups([{"name": "  ", "labels": [1]}]) == []


def test_the_second_structure_by_one_name_is_dropped_rather_than_refused():
    """The models raise on a duplicate, which a half-typed name would trip."""
    rows = [
        {"name": "LV", "labels": [1]},
        {"name": "LV", "labels": [2]},
        {"name": "RV", "labels": [3]},
    ]

    assert [group.name for group in usable_groups(rows)] == ["LV", "RV"]


def test_a_name_is_taken_without_the_space_around_it():
    """It is drawn as a chart title and matched against the page picker."""
    assert usable_groups([{"name": " LV ", "labels": [1]}])[0].name == "LV"


def test_the_structures_keep_the_order_they_were_given_in():
    rows = [{"name": name, "labels": [1]} for name in ("Myo", "LV", "RV")]

    assert [group.name for group in usable_groups(rows)] == ["Myo", "LV", "RV"]


@pytest.mark.parametrize("density", ["0", 0, "-1", "abc", "1.0.0"])
def test_a_density_that_cannot_be_used_is_ignored_rather_than_fatal(density):
    """It used to delete the structure, and typing 1.05 passes through 1 and
    0 on the way -- so a curve vanished and came back between keystrokes with
    nothing said about why.  A row with a name and labels is a structure
    whatever else has been typed into it."""
    group = usable_groups([{"name": "LV", "labels": [1], "density": density}])[0]

    assert group.name == "LV"
    assert group.density is None
    assert group.unit == "mL"


def test_a_colour_that_cannot_be_used_falls_back_to_the_labels_own():
    """The other optional field, and the same rule -- so the derivation of
    which fields these are is doing the work rather than a list of one."""
    group = usable_groups([{"name": "LV", "labels": [1], "color": "not a colour"}])[0]

    assert group.color is None
    assert group.rgb == label_color(1)


def test_the_optional_fields_are_the_ones_a_form_may_leave_unset():
    assert OPTIONAL_FIELDS == {"color", "density"}


@pytest.mark.parametrize(
    "row",
    [
        {"name": "LV", "labels": [], "density": "0"},
        {"name": "", "labels": [1], "density": "0"},
        "not a row at all",
    ],
)
def test_a_row_that_is_not_a_structure_is_still_dropped(row):
    """The leniency is about the optional fields only: a row with no labels is
    not a structure with a bad field, it is not a structure."""
    assert usable_groups([row]) == []


@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_a_cleared_density_field_reads_as_no_density(blank):
    """An emptied number field hands back "", not nothing at all -- and what
    is left behind after a clear may be a space rather than an empty string."""
    group = usable_groups([{"name": "LV", "labels": [1], "density": blank}])[0]

    assert group.density is None
    assert group.unit == "mL"


def test_structures_already_built_are_taken_as_they_are():
    """The config's own list goes through the same door as the drawer's."""
    groups = [StructureGroup(name="LV", labels=[1])]

    assert usable_groups(groups) == groups


def test_the_structures_are_held_as_an_object_and_not_as_a_bare_list():
    """trame's deep-reactive widget mirrors a state variable into a plain
    ``reactive({})`` and assigns over it, so a bare list arrives as
    ``{"0": ..., "1": ...}``: no length to test, so nothing renders, and that
    shape pushed back as the new value.  The rotation sequence is an object
    with its list inside it and works; this is the same wrapper, for the same
    reason, and a bug that shows up as a button doing nothing at all."""
    held = structure_state([StructureGroup(name="LV", labels=[1])])

    assert isinstance(held, dict)
    assert isinstance(held[STRUCTURES], list)
    assert structure_rows(held)[0]["name"] == "LV"


@pytest.mark.parametrize("held", [None, {}, {STRUCTURES: None}, {STRUCTURES: []}])
def test_a_drawer_holding_nothing_yet_reads_as_no_structures(held):
    assert structure_rows(held) == []
