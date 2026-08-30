"""Test that ResliceSet poses the three MPR views as the per-object code used to."""

# Third Party
import numpy as np
import pytest
import vtk

# Internal
from cardio.orientation import (
    AngleUnits,
    EulerAxis,
    axcode_transform_matrix,
    create_vtk_reslice_matrix,
    euler_angle_to_rotation_matrix,
)
from cardio.reslice import VIEWS, ResliceSet
from tests.geometry import matrix_array
from tests.phantoms import make_image


def reslice_axes(reslice) -> np.ndarray:
    return matrix_array(reslice.GetResliceAxes())


def expected_axes(view: str, rotation: np.ndarray, origin: list[float]) -> np.ndarray:
    """Recompute the pre-refactor matrix independently of ResliceSet."""
    axcodes = {"axial": "LAS", "sagittal": "ASL", "coronal": "LSA"}
    transform = rotation @ axcode_transform_matrix("LPS", axcodes[view])
    return matrix_array(create_vtk_reslice_matrix(transform, origin))


@pytest.fixture
def reslice_set() -> ResliceSet:
    return ResliceSet(make_image(), interpolation="linear", background_level=-1000.0)


def test_views_are_the_three_mpr_orientations(reslice_set):
    assert set(reslice_set.views) == {"axial", "sagittal", "coronal"}
    assert VIEWS == ("axial", "sagittal", "coronal")


def test_new_set_is_centred_on_the_image(reslice_set):
    center = list(make_image().GetCenter())
    for view in VIEWS:
        axes = reslice_axes(reslice_set[view]["reslice"])
        np.testing.assert_allclose(axes, expected_axes(view, np.eye(3), center))


@pytest.mark.parametrize("axis", [EulerAxis.X, EulerAxis.Y, EulerAxis.Z])
@pytest.mark.parametrize("angle", [0.0, 30.0, -75.0])
def test_set_pose_matches_the_pre_refactor_matrices(reslice_set, axis, angle):
    origin = [3.0, -4.0, 5.0]
    rotation = euler_angle_to_rotation_matrix(axis, angle, AngleUnits.DEGREES)

    reslice_set.set_pose(origin, rotation)

    for view in VIEWS:
        axes = reslice_axes(reslice_set[view]["reslice"])
        np.testing.assert_allclose(
            axes, expected_axes(view, rotation, origin), atol=1e-12
        )


def test_set_pose_from_sequence_composes_in_order(reslice_set):
    origin = [1.0, 2.0, 3.0]
    sequence = [{"axis": "X"}, {"axis": "Z"}]
    angles = {0: 20.0, 1: 45.0}

    reslice_set.set_pose_from_sequence(origin, sequence, angles, AngleUnits.DEGREES)

    rotation = euler_angle_to_rotation_matrix(
        EulerAxis.X, 20.0, AngleUnits.DEGREES
    ) @ euler_angle_to_rotation_matrix(EulerAxis.Z, 45.0, AngleUnits.DEGREES)

    for view in VIEWS:
        axes = reslice_axes(reslice_set[view]["reslice"])
        np.testing.assert_allclose(
            axes, expected_axes(view, rotation, origin), atol=1e-12
        )


def test_all_views_share_one_origin(reslice_set):
    origin = [7.0, 8.0, 9.0]
    reslice_set.set_pose(origin, np.eye(3))

    for view in VIEWS:
        axes = reslice_axes(reslice_set[view]["reslice"])
        np.testing.assert_allclose(axes[:3, 3], origin)


def test_set_window_level_reaches_every_view(reslice_set):
    reslice_set.set_window_level(350.0, 60.0)

    for view in VIEWS:
        image_property = reslice_set[view]["actor"].GetProperty()
        assert image_property.GetColorWindow() == pytest.approx(350.0)
        assert image_property.GetColorLevel() == pytest.approx(60.0)


def test_output_filter_contributes_extra_pipeline_entries():
    def add_filter(reslice):
        shifter = vtk.vtkImageShiftScale()
        shifter.SetInputConnection(reslice.GetOutputPort())
        return shifter, {"shifter": shifter}

    views = ResliceSet(
        make_image(),
        interpolation="nearest",
        background_level=0,
        output_filter=add_filter,
    )

    for view in VIEWS:
        assert "shifter" in views[view]
        assert views[view]["actor"].GetVisibility() == 0


def test_unknown_interpolation_is_rejected():
    with pytest.raises(ValueError, match="Unknown interpolation mode"):
        ResliceSet(make_image(), interpolation="cubic", background_level=0)
