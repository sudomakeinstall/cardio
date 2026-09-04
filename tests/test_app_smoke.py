"""Construct Logic and UI against a real Scene.

The unit tests cover pure functions; nothing else builds the trame layout, so a
method deleted or renamed out from under `setup()` shows up only at runtime.
This is the cheap guard against that.
"""

# System
import itertools
import re

# Third Party
import itk
import numpy as np
import pytest
import trame.app
import vtk

# Internal
from cardio.logic import ALIGN_STEP_NAME, Logic
from cardio.orientation import AngleUnits
from cardio.rotation import RotationMetadata
from cardio.scene import Scene
from cardio.state import ObjectState
from cardio.ui import UI
from cardio.view import Theme

_server_names = itertools.count()


def write_volume(path):
    array = np.linspace(-500, 500, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    itk.imwrite(itk.image_from_array(array), str(path))


def write_segmentation(path):
    """Three stacked labels, so the traverse branch has two interfaces."""
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[2:6, 2:6, 1:3] = 1
    array[2:6, 2:6, 3:5] = 2
    array[2:6, 2:6, 5:7] = 3
    itk.imwrite(itk.image_from_array(array), str(path))


def write_mesh(path):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(3.0)
    sphere.Update()
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(sphere.GetOutput())
    writer.Write()


def build_scene(directory, segmentation_overrides=None, **overrides) -> Scene:
    """One object of every renderable type, so every UI branch is built."""
    write_volume(directory / "vol0.nii.gz")
    write_segmentation(directory / "seg0.nii.gz")
    write_mesh(directory / "mesh0.obj")

    return Scene(
        volumes=[
            {"label": "vol", "directory": directory, "file_paths": ["vol0.nii.gz"]}
        ],
        segmentations=[
            {
                "label": "seg",
                "directory": directory,
                "file_paths": ["seg0.nii.gz"],
                **(segmentation_overrides or {}),
            }
        ],
        meshes=[{"label": "mesh", "directory": directory, "file_paths": ["mesh0.obj"]}],
        **overrides,
    )


def build_app(scene):
    """Logic then UI, in the order app.CardioApp builds them."""
    server = trame.app.get_server(f"test-{next(_server_names)}", client_type="vue3")
    logic = Logic(server, scene)
    ui = UI(server, scene, logic)
    return server, scene, logic, ui


@pytest.fixture
def scene(tmp_path) -> Scene:
    return build_scene(tmp_path)


@pytest.fixture
def app(scene):
    """A build of its own, for the tests that write to it."""
    return build_app(scene)


def snapshot(state) -> str:
    """Every state value, as text.

    ``state.initial`` is the whole state; ``to_dict`` is only what has been
    pushed to a client, which for a server nothing has connected to is nothing
    at all. Rendered as text because the values include arrays, which do not
    answer ``==`` with a bool.
    """
    return repr(sorted(dict(state.initial).items(), key=str))


@pytest.fixture(scope="module")
def read_only_app(tmp_path_factory):
    """One build shared by the tests that only read it.

    Building Logic and UI is nearly the whole cost of this file, so the tests
    that write nothing share a single build. The teardown is what keeps that
    claim honest: a test that does write is named here, rather than surfacing
    as a failure in whichever test happened to run after it.
    """
    built = build_app(build_scene(tmp_path_factory.mktemp("smoke")))
    server = built[0]
    before = snapshot(server.state)

    yield built

    assert snapshot(server.state) == before, (
        "read_only_app was mutated: give that test the function-scoped `app`"
    )


def test_logic_and_ui_construct(read_only_app):
    server, _, _, _ = read_only_app
    assert server.state.trame__title.startswith("cardio v")


def test_every_object_gets_its_state_keys_registered(read_only_app):
    server, scene, _, _ = read_only_app

    for obj in scene.renderables:
        keys = ObjectState.of(obj)
        assert hasattr(server.state, keys.visibility), keys.visibility
        assert hasattr(server.state, keys.clipping), keys.clipping
        assert hasattr(server.state, keys.clip_panel), keys.clip_panel
        for key in keys.clip_bounds:
            assert hasattr(server.state, key), key


def test_volume_and_segmentation_specific_keys_are_registered(read_only_app):
    server, scene, _, _ = read_only_app

    for volume in scene.volumes:
        assert hasattr(server.state, ObjectState.of(volume).preset)
        assert hasattr(server.state, ObjectState.of(volume).preset_panel)

    for seg in scene.segmentations:
        assert hasattr(server.state, ObjectState.of(seg).mpr_overlay)


def test_controller_entry_points_the_ui_binds_all_exist(read_only_app):
    server, _, _, _ = read_only_app

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
        "reset_snap",
        "reset_tile_cameras",
        "view_update",
        "view_reset_camera",
    ):
        assert getattr(server.controller, name) is not None, name


def test_mpr_views_are_built_and_shared_with_the_scene(read_only_app):
    _, scene, _, _ = read_only_app

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


def test_rotation_state_starts_in_step_across_all_three_representations(read_only_app):
    server, scene, logic, _ = read_only_app

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


def test_an_unrecognised_index_order_is_rejected(read_only_app):
    _, _, logic, _ = read_only_app

    with pytest.raises(ValueError, match="Unrecognized index order"):
        logic.rotations.sync_index_order("nonsense")


def test_the_renderer_starts_in_the_configured_theme(read_only_app):
    server, scene, logic, _ = read_only_app

    assert server.state.theme_mode == scene.view.theme.value
    assert scene.renderer.GetBackground() == pytest.approx(
        logic.visibility.background_color(scene.view.theme.value)
    )


def test_toggling_the_theme_repaints_the_background(app):
    _, scene, logic, _ = app

    logic.visibility.sync_background_color(Theme.LIGHT.value)

    assert scene.renderer.GetBackground() == pytest.approx(scene.background.light)


# --- traverse runs through the real server state ------------------------------


def connect(server):
    """Do what the running app does when a client connects.

    ``state.ready()`` is what arms the change listeners, and without it a write
    reaches no listener. ``finalize_mpr_initialization`` is hooked to
    ``on_server_ready``, which nothing here fires, and it is what sets the
    active volume the views resample.
    """
    server.state.ready()
    server.controller.finalize_mpr_initialization()


def traverse_selection(server):
    """Select the three stacked labels the smoke segmentation carries."""
    connect(server)
    with server.state:
        server.state.snap_mode = "traverse"
        server.state.snap_labels_a = [1]
        server.state.snap_labels_b = [2]
        server.state.snap_labels_c = [3]


def test_traverse_walks_the_origin_between_the_interfaces(app):
    server, _, _, _ = app
    traverse_selection(server)

    server.controller.align_to_interface()
    start = list(server.state.mpr_origin)

    with server.state:
        server.state.snap_traverse = 100
    end = list(server.state.mpr_origin)

    assert not server.state.snap_no_interface
    assert end != start
    assert [step["name"] for step in server.state.mpr_rotation_data["angles_list"]] == [
        "Interface plane"
    ]


def test_traverse_slider_listener_is_wired(app):
    """The slider must move the views on its own, without pressing Align."""
    server, _, _, _ = app
    traverse_selection(server)

    with server.state:
        server.state.snap_traverse = 60

    assert server.state.mpr_rotation_data["angles_list"]


def test_swap_reverses_traverse_without_losing_the_middle_group(app):
    server, _, _, _ = app
    traverse_selection(server)

    with server.state:
        server.state.snap_traverse = 25
    server.controller.swap_snap_groups()

    assert server.state.snap_labels_a == [3]
    assert server.state.snap_labels_b == [2]
    assert server.state.snap_labels_c == [1]
    assert server.state.snap_traverse == 75


# --- tile mode runs through the real server state -----------------------------


def tile_props(scene) -> list[int]:
    return [r.GetViewProps().GetNumberOfItems() for r in scene.tile_views.renderers]


def test_the_tile_window_is_built_with_the_layout(read_only_app):
    _, scene, _, _ = read_only_app

    assert scene.tile_views is not None
    assert scene.tile_views.window.GetOffScreenRendering() == 1
    assert len(scene.tile_views) == scene.tile_rows * scene.tile_cols


def test_entering_tile_mode_fills_the_grid(app):
    server, scene, _, _ = app
    traverse_selection(server)

    with server.state:
        server.state.maximized_view = "tile"

    assert not server.state.snap_no_interface
    assert tile_props(scene) == [1] * len(scene.tile_views)


def test_reshaping_the_grid_resamples_the_path(app):
    server, scene, _, _ = app
    traverse_selection(server)

    with server.state:
        server.state.maximized_view = "tile"
    with server.state:
        server.state.tile_rows = 2
        server.state.tile_cols = 4

    assert len(scene.tile_views) == 8
    assert tile_props(scene) == [1] * 8


def test_the_grid_stays_empty_without_a_path(app):
    server, scene, _, _ = app
    connect(server)

    with server.state:
        server.state.maximized_view = "tile"

    assert tile_props(scene) == [0] * len(scene.tile_views)


def test_leaving_tile_mode_stops_retiling(app):
    server, scene, _, _ = app
    traverse_selection(server)

    with server.state:
        server.state.maximized_view = "tile"
    with server.state:
        server.state.maximized_view = ""
        server.state.tile_rows = 6

    assert len(scene.tile_views) == scene.tile_rows * scene.tile_cols


# --- Reset returns the snap panel to its defaults -----------------------------


def test_reset_undoes_a_locked_traverse_alignment(app):
    server, _, _, _ = app
    traverse_selection(server)

    with server.state:
        server.state.snap_traverse = 40
        server.state.snap_locked = True
        server.state.snap_orientation_locked = True
    server.controller.align_to_interface()

    assert server.state.mpr_rotation_data["angles_list"]
    moved = list(server.state.mpr_origin)

    with server.state:
        server.controller.reset_snap()

    state = server.state
    assert state.snap_mode == "label"
    assert state.snap_labels_a == [] and state.snap_labels_b == []
    assert state.snap_labels_c == []
    assert state.snap_traverse == 0
    assert state.snap_locked is False
    assert state.snap_orientation_locked is False
    assert state.mpr_rotation_data["angles_list"] == []
    assert state.mpr_origin != moved


def test_reset_recentres_on_the_volume(app):
    server, scene, logic, _ = app
    connect(server)

    with server.state:
        server.state.mpr_origin = [99.0, 99.0, 99.0]
    with server.state:
        server.controller.reset_snap()

    centre = logic.mpr.convention.point_from_itk(
        scene.volumes[0].mpr_image_data(0).GetCenter()
    )
    assert server.state.mpr_origin == pytest.approx(centre)


# --- the generated vue expressions ---------------------------------------------


def test_no_binding_negates_a_variable_it_then_compares(read_only_app):
    """Catch ``!x === 'y'``, which JS reads as ``(!x) === 'y'`` and never fires.

    An expression like this raises no error anywhere: the element simply never
    renders. Building the negation by prefixing "!" to a constant holding a
    comparison is the easy way to write one by accident.
    """
    _, _, _, ui = read_only_app

    trap = re.compile(r"![A-Za-z_$][\w$.]*\s*[=!]==")
    offenders = [
        binding
        for binding in re.findall(r'(?:v-if|:disabled)="([^"]*)"', ui.layout.html)
        if trap.search(binding)
    ]

    assert offenders == []


def test_the_traverse_slider_is_reachable_in_the_quad_view(read_only_app):
    """It went missing once behind exactly the negation above."""
    _, _, _, ui = read_only_app

    slider = re.search(r'<VSlider[^>]*v-model="snap_traverse"[^>]*>', ui.layout.html)
    assert slider is not None
    condition = re.search(r'v-if="([^"]*)"', slider.group(0)).group(1)
    assert "maximized_view !== 'tile'" in condition


def build_ready(tmp_path, **overrides):
    """A built app taken through the startup a running server performs.

    ``state.ready()`` rather than ``state.flush()``: flushing is skipped until
    the state is marked ready, so a test that only flushes fires no change
    listener at all and proves nothing about what they do.
    ``finalize_mpr_initialization`` is the initialization the UI defers to the
    client's mount, which is what fills in the active volume.
    """
    server, scene, logic, ui = build_app(build_scene(tmp_path, **overrides))
    server.state.ready()
    server.controller.finalize_mpr_initialization()
    server.state.flush()
    return server, scene, logic, ui


def test_configured_snap_survives_startup(tmp_path):
    """The configured selection is what the panel holds once the state settles.

    The unit tests drive the controllers directly; only here does the
    ``snap_seg_label`` listener actually fire, which is what used to clear the
    groups out from under the config.
    """
    server, _, _, _ = build_ready(
        tmp_path,
        snap={
            "mode": "traverse",
            "labels_a": [1],
            "labels_b": [2],
            "labels_c": [3],
            "traverse": 50,
            "locked": True,
            "orientation_locked": True,
        },
    )

    assert server.state.snap_mode == "traverse"
    assert server.state.snap_labels_a == [1]
    assert server.state.snap_labels_b == [2]
    assert server.state.snap_labels_c == [3]
    assert server.state.snap_traverse == 50
    assert server.state.snap_locked is True
    assert server.state.snap_orientation_locked is True
    assert server.state.snap_no_interface is False
    assert ALIGN_STEP_NAME in [
        step["name"] for step in server.state.mpr_rotation_data["angles_list"]
    ]


def test_configured_layout_opens_maximized(tmp_path):
    server, _, _, _ = build_ready(tmp_path, view={"layout": "sagittal"})
    assert server.state.maximized_view == "sagittal"


def test_the_quad_layout_leaves_maximized_view_empty(tmp_path):
    server, _, _, _ = build_ready(tmp_path)
    assert server.state.maximized_view == ""


def test_opening_in_tile_mode_builds_the_grid(tmp_path):
    """The riskiest of these: the tile listener fires on the first flush.

    Tiles are only drawn for a traverse path, so the layout and the snap
    selection have to be configured together to reach the grid at all.
    """
    server, scene, logic, _ = build_ready(
        tmp_path,
        view={"layout": "tile"},
        snap={
            "mode": "traverse",
            "labels_a": [1],
            "labels_b": [2],
            "labels_c": [3],
        },
    )

    assert server.state.maximized_view == "tile"
    assert logic.tiles.active
    assert server.state.snap_no_interface is False
    # Every tile actually carries a cut, rather than the grid merely existing:
    # the render windows are built whichever layout is on screen.
    assert all(
        renderer.GetViewProps().GetNumberOfItems() > 0
        for renderer in scene.tile_views.renderers
    )


def test_configured_theme_reaches_the_renderer(tmp_path):
    server, scene, _, _ = build_ready(tmp_path, view={"theme": "light"})

    assert server.state.theme_mode == "light"
    assert scene.renderer.GetBackground() == pytest.approx(scene.background.light)


def test_configured_segmentation_overlay_starts_on(tmp_path):
    server, scene, _, _ = build_ready(
        tmp_path, segmentation_overrides={"mpr_overlay": True}
    )

    for seg in scene.segmentations:
        assert server.state[ObjectState.of(seg).mpr_overlay] is True


def test_configured_overlay_opacity_reaches_state(tmp_path):
    server, _, _, _ = build_ready(tmp_path, mpr_segmentation_opacity=0.25)
    assert server.state.mpr_segmentation_opacity == 0.25


def test_the_quad_layout_leaves_the_tiles_empty(tmp_path):
    """The counterpart to the tile test: an empty grid is what "not built" looks like."""
    _, scene, logic, _ = build_ready(
        tmp_path,
        snap={"mode": "traverse", "labels_a": [1], "labels_b": [2], "labels_c": [3]},
    )

    assert not logic.tiles.active
    assert all(
        renderer.GetViewProps().GetNumberOfItems() == 0
        for renderer in scene.tile_views.renderers
    )


def drawn(scene) -> dict[str, int]:
    """How many props each MPR view is holding."""
    return {
        view: scene.mpr_views.renderer(view).GetViewProps().GetNumberOfItems()
        for view in scene.mpr_views
    }


def slice_origin(scene, view: str = "axial") -> list[float]:
    """Where a view's reslice is currently aimed, in ITK coordinates."""
    axes = (
        scene.volumes[0].get_mpr_actors_for_frame(0)[view]["reslice"].GetResliceAxes()
    )
    return [axes.GetElement(row, 3) for row in range(3)]


@pytest.mark.parametrize("layout", ["volume", "tile"])
def test_a_layout_that_hides_the_slices_does_not_reslice(tmp_path, layout):
    """Three resampled planes per frame, for views nobody is looking at."""
    _, scene, logic, _ = build_ready(tmp_path, view={"layout": layout})

    assert not logic.mpr.active
    assert set(drawn(scene).values()) == {0}


@pytest.mark.parametrize("layout", ["quad", "axial"])
def test_a_layout_that_shows_the_slices_draws_them(tmp_path, layout):
    _, scene, logic, _ = build_ready(tmp_path, view={"layout": layout})

    assert logic.mpr.active
    assert all(count > 0 for count in drawn(scene).values())


def test_returning_to_a_slice_layout_draws_it(tmp_path):
    server, scene, _, _ = build_ready(tmp_path, view={"layout": "volume"})

    with server.state:
        server.state.maximized_view = ""
    server.state.flush()

    assert all(count > 0 for count in drawn(scene).values())


def test_the_views_come_back_at_the_origin_they_missed(tmp_path):
    """The point of the catch-up: the origin moved with nothing listening."""
    server, scene, logic, _ = build_ready(tmp_path)
    moved = [3.0, -2.0, 1.0]

    for key, value in (("maximized_view", "volume"), ("mpr_origin", moved)):
        with server.state:
            server.state[key] = value
        server.state.flush()

    assert slice_origin(scene) != pytest.approx(
        logic.mpr.convention.point_to_itk(moved)
    )

    with server.state:
        server.state.maximized_view = ""
    server.state.flush()

    assert slice_origin(scene) == pytest.approx(
        logic.mpr.convention.point_to_itk(moved)
    )


def test_a_locked_volume_camera_follows_while_the_slices_are_hidden(tmp_path):
    """The volume rendering is on screen in exactly the layout the slices are not."""
    _, scene, logic, _ = build_ready(
        tmp_path, view={"layout": "volume", "camera_lock": "LL"}
    )
    camera = scene.renderer.GetActiveCamera()
    camera.SetViewUp(1.0, 0.0, 0.0)

    logic.mpr.update_mpr_rotation()

    assert camera.GetViewUp() == pytest.approx((0.0, 0.0, 1.0))


def test_configured_camera_lock_reaches_state(tmp_path):
    """Seeding this fires a camera sync on the first flush, so it is built here."""
    server, _, _, _ = build_ready(tmp_path, view={"camera_lock": "LL"})

    assert server.state.camera_lock == "LL"
    assert [item["value"] for item in server.state.camera_lock_items] == [
        "free",
        "UL",
        "LL",
        "LR",
    ]


def test_configured_drawer_sections_open(tmp_path):
    server, _, _, _ = build_ready(
        tmp_path, view={"drawer_sections": ["orientation", "export"]}
    )
    assert server.state.drawer_sections == ["orientation", "export"]


def test_the_help_dialog_can_open_with_the_app(tmp_path):
    server, _, _, _ = build_ready(tmp_path, view={"help_visible": True})
    assert server.state.help_overlay_visible is True


def test_the_help_dialog_is_closed_by_default(tmp_path):
    server, _, _, _ = build_ready(tmp_path)
    assert server.state.help_overlay_visible is False


def test_the_metadata_sheet_can_open_with_the_app(tmp_path):
    server, _, _, _ = build_ready(tmp_path, view={"metadata_visible": True})
    assert server.state.metadata_overlay_visible is True


def test_the_metadata_sheet_is_closed_by_default(tmp_path):
    server, _, _, _ = build_ready(tmp_path)
    assert server.state.metadata_overlay_visible is False


def test_the_toolbar_opens_both_reference_sheets(read_only_app):
    """The buttons toggle in the browser, so a misspelled key fails silently."""
    _, _, _, ui = read_only_app

    assert "metadata_overlay_visible = !metadata_overlay_visible" in ui.layout.html
    assert "help_overlay_visible = !help_overlay_visible" in ui.layout.html


def test_the_metadata_sheet_holds_a_page_for_every_object(read_only_app):
    """One dropdown branch per renderable; a missed kind shows as a missing key."""
    _, scene, _, ui = read_only_app

    for obj in scene.renderables:
        assert f"metadata_object === '{obj.kind}:{obj.label}'" in ui.layout.html
