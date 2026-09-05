"""The cameras, as part of what the app is showing.

A camera is the one thing on screen that never passed through state: the volume
view forwards its drags to VTK's trackball, which turns the camera and tells
nobody. What is checked here is that the mirror is right when it is looked at,
that it can point the cameras when told to, and -- the part that is easy to get
wrong -- that it does not point them when it has not been told to.
"""

# Third Party
import pytest
import vtk

# Internal
from cardio.camera import Cameras, Pose
from tests.test_app_smoke import build_app, build_scene, connect

# --------------------------------------------------------------- the model ---


def test_a_pose_reads_and_writes_a_camera():
    one, other = vtk.vtkCamera(), vtk.vtkCamera()
    one.SetPosition(1.0, 2.0, 3.0)
    one.SetFocalPoint(4.0, 5.0, 6.0)
    one.SetViewUp(0.0, 0.0, 1.0)
    one.SetParallelScale(7.0)

    Pose.of(one).apply_to(other)

    assert Pose.of(other) == Pose.of(one)


def test_a_pose_survives_being_written_down():
    camera = vtk.vtkCamera()
    camera.Azimuth(37.0)
    pose = Pose.of(camera)

    assert Pose(**pose.model_dump(mode="json")) == pose


def test_only_the_slice_poses_that_are_set_are_offered():
    cameras = Cameras(axial=Pose.of(vtk.vtkCamera()))

    assert list(cameras.slices) == ["axial"]


def test_an_unknown_camera_is_refused():
    with pytest.raises(Exception, match="tilt"):
        Cameras(tilt=1.0)


# ---------------------------------------------------------- in the running app ---


@pytest.fixture
def app(tmp_path):
    built = build_app(build_scene(tmp_path))
    connect(built[0])
    return built


def volume_camera(scene):
    return scene.renderer.GetActiveCamera()


def test_every_view_that_exists_is_mirrored(app):
    server, _, _, _ = app
    cameras = Cameras(**server.state.cameras)

    assert cameras.volume is not None
    assert set(cameras.slices) == {"axial", "coronal", "sagittal"}
    assert cameras.tile_scale is not None


def dragged(camera):
    """Turn the camera the way a drag on the volume view turns it.

    Not ``Azimuth``: the scene opens with the camera on its own view-up axis,
    which is the one axis an azimuth turns about to no effect.
    """
    camera.Elevation(25.0)
    camera.OrthogonalizeViewUp()


def test_the_mirror_says_where_the_camera_actually_is(app):
    server, scene, logic, _ = app
    dragged(volume_camera(scene))

    logic.camera.publish()

    assert Cameras(**server.state.cameras).volume == Pose.of(volume_camera(scene))


def test_a_camera_moved_behind_the_mirrors_back_is_not_snapped_home(app):
    """The trackball turns the camera without going through state.

    A listener that pointed the cameras wherever state last said would undo
    every drag on the next unrelated flush.
    """
    server, scene, _, _ = app
    dragged(volume_camera(scene))
    moved = volume_camera(scene).GetPosition()

    with server.state:
        server.state.mpr_segmentation_opacity = 0.33

    assert volume_camera(scene).GetPosition() == moved


def test_writing_a_pose_into_state_points_the_camera(app):
    """Which is what undoing a camera move comes down to."""
    server, scene, _, _ = app
    wanted = Pose(
        position=(10.0, 20.0, 30.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 0.0, 1.0),
        parallel_scale=3.0,
    )

    with server.state:
        server.state.cameras = {
            **server.state.cameras,
            "volume": wanted.model_dump(mode="json"),
        }

    assert Pose.of(volume_camera(scene)) == wanted


def test_a_trackball_drag_is_undone_like_anything_else(app):
    """The drag reaches VTK and not state, so nothing has changed to undo.

    Letting go is what writes it down, and that is an action like any other
    with a before to go back to. Without it a drag would be invisible to the
    journal, and writing the old pose back would be dropped as a no-op --
    state having never left it.
    """
    server, scene, logic, _ = app
    changes = []
    logic.journal.watch(changes.append)

    home = Pose.of(volume_camera(scene))
    dragged(volume_camera(scene))
    assert Pose.of(volume_camera(scene)) != home

    logic.dispatch("place_camera")
    assert "cameras" in changes[-1].before

    with server.state:
        for key, value in changes[-1].before.items():
            server.state[key] = value

    assert Pose.of(volume_camera(scene)) == home


def test_a_pose_may_be_placed_outright(app):
    """Which is how a script points a camera."""
    _, scene, logic, _ = app
    wanted = Pose(
        position=(1.0, 2.0, 3.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 0.0, 1.0),
        parallel_scale=9.0,
    )

    logic.dispatch("place_camera", cameras={"volume": wanted.model_dump(mode="json")})

    assert Pose.of(volume_camera(scene)) == wanted


def test_a_camera_move_is_reported_as_part_of_the_change(app):
    _, _, logic, _ = app
    changes = []
    logic.journal.watch(changes.append)

    logic.dispatch("zoom_views", factor=2.0)

    assert "cameras" in changes[-1].before


def test_undoing_a_zoom_puts_the_views_back(app):
    server, scene, logic, _ = app
    changes = []
    logic.journal.watch(changes.append)

    def placements():
        return [
            scene.mpr_views.renderer(view).GetActiveCamera().GetPosition()
            for view in scene.mpr_views
        ]

    before = placements()

    logic.dispatch("zoom_views", factor=2.0)
    assert placements() != before

    with server.state:
        for key, value in changes[-1].before.items():
            server.state[key] = value

    assert placements() == before


def test_a_configured_pose_points_the_camera(tmp_path):
    """The other half of a saved session: reopening it looks the same."""
    pose = Pose(
        position=(11.0, 22.0, 33.0),
        focal_point=(0.0, 1.0, 2.0),
        view_up=(0.0, 0.0, 1.0),
        parallel_scale=4.0,
    )
    scene = build_scene(tmp_path, view={"cameras": {"volume": pose.model_dump()}})
    server, scene, _, _ = build_app(scene)
    connect(server)

    assert Pose.of(volume_camera(scene)) == pose


def test_a_configured_slice_pose_waits_for_its_view_to_exist(tmp_path):
    """The MPR windows are built after Logic, so there is nothing to point yet."""
    pose = Pose(
        position=(0.0, 0.0, 90.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 1.0, 0.0),
        parallel_scale=5.0,
    )
    scene = build_scene(tmp_path, view={"cameras": {"axial": pose.model_dump()}})
    server, scene, _, _ = build_app(scene)
    connect(server)

    axial = scene.mpr_views.renderer("axial").GetActiveCamera()
    assert Pose.of(axial) == pose


def test_a_cine_does_not_fill_the_journal_with_a_change_per_frame(app):
    """The playback and capture loops turn the camera and step the frame by
    calling the methods rather than asking for the actions.

    Which is what keeps a three-hundred-and-sixty-frame rotating capture from
    being three hundred and sixty things to undo.
    """
    _, scene, logic, _ = app
    changes = []
    logic.journal.watch(changes.append)

    for _ in range(5):
        scene.renderer.GetActiveCamera().Azimuth(1.0)
        logic.playback.increment_frame()

    assert changes == []

    logic.dispatch("increment_frame")
    assert [change.action for change in changes] == ["increment_frame"]
