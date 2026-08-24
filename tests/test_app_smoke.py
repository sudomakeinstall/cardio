"""Construct Logic and UI against a real Scene.

The unit tests cover pure functions; nothing else builds the trame layout, so a
method deleted or renamed out from under `setup()` shows up only at runtime.
This is the cheap guard against that.
"""

# System
import itertools

# Third Party
import itk
import numpy as np
import pytest
import trame.app
import vtk

# Internal
from cardio.logic import Logic
from cardio.orientation import AngleUnits
from cardio.rotation import RotationMetadata
from cardio.scene import Scene
from cardio.state import DEFAULT_THEME_MODE, THEME_LIGHT, ObjectState
from cardio.ui import UI

_server_names = itertools.count()


def write_volume(path):
    array = np.linspace(-500, 500, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    itk.imwrite(itk.image_from_array(array), str(path))


def write_segmentation(path):
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[2:6, 2:6, 1:4] = 1
    array[2:6, 2:6, 4:7] = 2
    itk.imwrite(itk.image_from_array(array), str(path))


def write_mesh(path):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(3.0)
    sphere.Update()
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(sphere.GetOutput())
    writer.Write()


@pytest.fixture
def scene(tmp_path) -> Scene:
    """One object of every renderable type, so every UI branch is built."""
    write_volume(tmp_path / "vol0.nii.gz")
    write_segmentation(tmp_path / "seg0.nii.gz")
    write_mesh(tmp_path / "mesh0.obj")

    return Scene(
        volumes=[
            {"label": "vol", "directory": tmp_path, "file_paths": ["vol0.nii.gz"]}
        ],
        segmentations=[
            {"label": "seg", "directory": tmp_path, "file_paths": ["seg0.nii.gz"]}
        ],
        meshes=[{"label": "mesh", "directory": tmp_path, "file_paths": ["mesh0.obj"]}],
    )


@pytest.fixture
def app(scene):
    """Logic then UI, in the order app.CardioApp builds them."""
    server = trame.app.get_server(f"test-{next(_server_names)}", client_type="vue3")
    logic = Logic(server, scene)
    ui = UI(server, scene, logic)
    return server, scene, logic, ui


def test_logic_and_ui_construct(app):
    server, _, _, _ = app
    assert server.state.trame__title.startswith("cardio v")


def test_every_object_gets_its_state_keys_registered(app):
    server, scene, _, _ = app

    for obj in scene.renderables:
        keys = ObjectState.of(obj)
        assert hasattr(server.state, keys.visibility), keys.visibility
        assert hasattr(server.state, keys.clipping), keys.clipping
        assert hasattr(server.state, keys.clip_panel), keys.clip_panel
        for key in keys.clip_bounds:
            assert hasattr(server.state, key), key


def test_volume_and_segmentation_specific_keys_are_registered(app):
    server, scene, _, _ = app

    for volume in scene.volumes:
        assert hasattr(server.state, ObjectState.of(volume).preset)
        assert hasattr(server.state, ObjectState.of(volume).preset_panel)

    for seg in scene.segmentations:
        assert hasattr(server.state, ObjectState.of(seg).mpr_overlay)


def test_controller_entry_points_the_ui_binds_all_exist(app):
    server, _, _, _ = app

    for name in (
        "increment_frame",
        "decrement_frame",
        "screenshot",
        "save_rotation_angles",
        "reset_all",
        "close_application",
        "finalize_mpr_initialization",
        "add_x_rotation",
        "add_y_rotation",
        "add_z_rotation",
        "remove_rotation_event",
        "reset_rotation_angle",
        "reset_rotations",
        "reset_mpr_origin",
        "snap_to_centroid",
        "align_to_interface",
        "swap_snap_groups",
        "view_update",
        "view_reset_camera",
    ):
        assert getattr(server.controller, name) is not None, name


def test_mpr_views_are_built_and_shared_with_the_scene(app):
    _, scene, _, _ = app

    assert scene.mpr_views is not None
    for view in ("axial", "coronal", "sagittal"):
        assert scene.mpr_views[view] is not None


def test_renderables_covers_every_type(scene):
    assert [obj.kind for obj in scene.renderables] == [
        "mesh",
        "volume",
        "segmentation",
    ]


def test_frame_actor_wraps_for_short_series(scene):
    for obj in scene.renderables:
        assert obj.frame_actor(0) is obj.frame_actor(len(obj.actors))


def test_sync_visibility_follows_the_state_toggles(app):
    server, scene, logic, _ = app
    server.state.frame = 0

    for obj in scene.renderables:
        server.state[ObjectState.of(obj).visibility] = False
    logic.visibility.sync_visibility()
    assert all(obj.frame_actor(0).GetVisibility() == 0 for obj in scene.renderables)

    for obj in scene.renderables:
        server.state[ObjectState.of(obj).visibility] = True
    logic.visibility.sync_visibility()
    assert all(obj.frame_actor(0).GetVisibility() == 1 for obj in scene.renderables)


def test_sync_clipping_applies_bounds_to_every_type(app):
    server, scene, logic, _ = app

    for obj in scene.renderables:
        keys = ObjectState.of(obj)
        server.state[keys.clipping] = True
        bounds = obj.combined_bounds
        for key, low in zip(keys.clip_bounds, (0, 2, 4)):
            server.state[key] = [bounds[low], bounds[low + 1]]

    logic.clipping.sync_clipping()

    for obj in scene.renderables:
        assert obj.actors[0].GetMapper().GetClippingPlanes() is not None


# --- rotation state has a single writer --------------------------------------


def test_rotation_state_starts_in_step_across_all_three_representations(app):
    server, scene, logic, _ = app

    sequence = logic.rotations.rotation_sequence()
    assert sequence.metadata.angle_units.value == server.state.angle_units
    assert sequence.metadata.index_order.value == server.state.index_order
    assert scene.mpr_rotation_sequence.metadata.index_order == (
        sequence.metadata.index_order
    )


def test_adding_and_removing_rotations_round_trips(app):
    server, _, logic, _ = app

    logic.rotations.add_mpr_rotation("X")
    logic.rotations.add_mpr_rotation("Y")
    steps = server.state.mpr_rotation_data["angles_list"]
    assert [s["axis"] for s in steps] == ["X", "Y"]

    logic.rotations.remove_mpr_rotation(0)
    steps = server.state.mpr_rotation_data["angles_list"]
    assert [s["axis"] for s in steps] == ["Y"]


def test_reset_rotation_angle_zeroes_only_that_step(app):
    server, _, logic, _ = app

    logic.rotations.add_mpr_rotation("X")
    logic.rotations.add_mpr_rotation("Y")
    data = server.state.mpr_rotation_data
    data["angles_list"][0]["angle"] = 42.0
    data["angles_list"][1]["angle"] = 17.0
    server.state.mpr_rotation_data = data

    logic.rotations.reset_rotation_angle(0)

    steps = server.state.mpr_rotation_data["angles_list"]
    assert steps[0]["angle"] == 0.0
    assert steps[1]["angle"] == 17.0


def test_out_of_range_edits_are_ignored(app):
    server, _, logic, _ = app
    logic.rotations.add_mpr_rotation("X")

    logic.rotations.remove_mpr_rotation(9)
    logic.rotations.reset_rotation_angle(-3)

    assert len(server.state.mpr_rotation_data["angles_list"]) == 1


def test_reset_mpr_rotations_restores_the_model_defaults(app):
    server, _, logic, _ = app
    logic.rotations.add_mpr_rotation("X")

    logic.rotations.reset_mpr_rotations()

    assert server.state.mpr_rotation_data["angles_list"] == []
    # the mirrors follow the model rather than a hard-coded literal
    assert server.state.angle_units == RotationMetadata().angle_units.value
    assert server.state.index_order == RotationMetadata().index_order.value


def test_switching_units_keeps_the_mirror_variable_in_step(app):
    server, _, logic, _ = app
    logic.rotations.add_mpr_rotation("X")

    logic.rotations.sync_angle_units("degrees")

    assert server.state.angle_units == "degrees"
    assert (
        logic.rotations.rotation_sequence().metadata.angle_units == AngleUnits.DEGREES
    )


def test_switching_index_order_also_exchanges_the_origin(app):
    server, _, logic, _ = app
    server.state.mpr_origin = [1.0, 2.0, 3.0]

    logic.rotations.sync_index_order("roma")

    assert server.state.index_order == "roma"
    assert server.state.mpr_origin == [3.0, 2.0, 1.0]


def test_an_unrecognised_index_order_is_rejected(app):
    _, _, logic, _ = app

    with pytest.raises(ValueError, match="Unrecognized index order"):
        logic.rotations.sync_index_order("nonsense")


def test_the_renderer_starts_in_the_default_theme(app):
    server, scene, logic, _ = app

    assert server.state.theme_mode == DEFAULT_THEME_MODE
    assert scene.renderer.GetBackground() == pytest.approx(
        logic.visibility.background_color(DEFAULT_THEME_MODE)
    )


def test_toggling_the_theme_repaints_the_background(app):
    _, scene, logic, _ = app

    logic.visibility.sync_background_color(THEME_LIGHT)

    assert scene.renderer.GetBackground() == pytest.approx(scene.background.light)
