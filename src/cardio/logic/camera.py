"""The cameras, mirrored into state so they can be saved and put back."""

# Internal
from ..action import action
from ..camera import Cameras, Pose
from .base import Controller


class CameraController(Controller):
    """Reads where the cameras are looking, and points them where told.

    State is a mirror here rather than the source of truth, because a camera
    can move without anyone asking: the volume view forwards its drags to VTK's
    trackball, which turns the camera and tells nobody. So the mirror is
    refreshed at the moments it has to be right -- before anything looks at the
    document -- rather than kept right continuously, which would mean fighting
    the trackball for it.

    Telling the two directions apart is what ``_published`` is for. A value in
    state that is the one last published is this controller's own mirror and
    means nothing; any other value is somebody -- an undo, a restore, a config
    -- saying where the cameras should point.
    """

    def __init__(self, app):
        super().__init__(app)
        self._published = None
        self._configured = None

    def register(self):
        self.server.state.change("cameras")(self.apply)

    def seed(self):
        """Take the configured poses, to install once their views exist."""
        self._configured = self.scene.view.cameras
        self._published = None
        self.install_configured()

    def install_configured(self):
        """Point whichever cameras the config names and now exist.

        The MPR and tile windows are built after Logic is, so at seeding time
        there is usually nothing there to point. This runs again once the views
        are up, which is the moment ``finalize_mpr_initialization`` marks.
        """
        if self._configured is not None:
            self._configured = self._install(self._configured)
        self.publish()

    def read(self) -> Cameras:
        """Where the cameras are looking, as far as they exist."""
        cameras = Cameras(volume=Pose.of(self.scene.renderer.GetActiveCamera()))

        views = self.scene.mpr_views
        if views is not None:
            for view in views:
                setattr(cameras, view, Pose.of(views.renderer(view).GetActiveCamera()))

        tiles = self.scene.tile_views
        if tiles is not None and len(tiles):
            cameras.tile_scale = tiles.renderers[0].GetActiveCamera().GetParallelScale()

        return cameras

    def publish(self):
        """Write where the cameras are looking into state."""
        self._published = self.read().model_dump(mode="json")
        self.server.state.cameras = self._published

    @action("place_camera")
    def place_camera(self, cameras: dict | None = None):
        """Point the cameras where ``cameras`` says, or write down where they are.

        The volume view's drags reach VTK's trackball rather than any of this,
        so the camera can be somewhere new with no action having been asked
        for. Asking for this one on the release is what makes that a change
        like any other, with something to undo it to. Given a pose instead, it
        is how a script points a camera.
        """
        if cameras is not None:
            self._install(Cameras(**cameras))
            self.server.controller.view_update()
        self.publish()

    def apply(self, cameras, **kwargs):
        """Point the cameras where state says, unless state is only the mirror."""
        if cameras is None or cameras == self._published:
            return

        self._install(Cameras(**cameras))
        self.publish()
        self.server.controller.view_update()

    def _install(self, cameras: Cameras) -> Cameras | None:
        """Point every camera ``cameras`` names and that exists.

        Returns what is left over -- poses for views not built yet -- or None
        once there is nothing still waiting.
        """
        waiting = Cameras()

        if cameras.volume is not None:
            cameras.volume.apply_to(self.scene.renderer.GetActiveCamera())

        views = self.scene.mpr_views
        for view, pose in cameras.slices.items():
            if views is None:
                setattr(waiting, view, pose)
            else:
                pose.apply_to(views.renderer(view).GetActiveCamera())
                views.renderer(view).ResetCameraClippingRange()

        tiles = self.scene.tile_views
        if cameras.tile_scale is not None:
            if tiles is None or not len(tiles):
                waiting.tile_scale = cameras.tile_scale
            else:
                for renderer in tiles.renderers:
                    renderer.GetActiveCamera().SetParallelScale(cameras.tile_scale)

        return waiting if waiting.model_dump(exclude_none=True) else None
