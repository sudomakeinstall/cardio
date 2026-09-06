"""Per-object clipping planes and the camera's depth clipping range."""

# Internal
from ..state import ObjectState
from .base import Controller


class ClippingController(Controller):
    """Clip boxes for every renderable, plus the shared near/far range."""

    object_seeds = ("clipping", "preset")

    def register(self):
        state = self.server.state

        camera = self.scene.renderer.GetActiveCamera()

        def reapply_clip(obj, event):
            near, far = state.clip_depth
            if camera.GetClippingRange() != (near, far):
                camera.SetClippingRange(near, far)

        # Held so the observer is not garbage collected
        self._clip_observer = reapply_clip
        self.scene.renderWindow.AddObserver("StartEvent", self._clip_observer)

        state.change("clip_depth")(self.sync_clip_depth)

        clipping_keys = [
            key
            for obj in self.scene.renderables
            for key in ObjectState.of(obj).clip_controls
        ]
        if clipping_keys:
            state.change(*clipping_keys)(self.sync_clipping)

    def sync_clipping(self, **kwargs):
        """Apply each object's clipping toggle and bounds from the UI controls."""
        for obj in self.scene.renderables:
            keys = ObjectState.of(obj)
            enabled = self.server.state[keys.clipping]
            obj.toggle_clipping(enabled)

            if not enabled:
                continue

            ranges = [getattr(self.server.state, key, None) for key in keys.clip_bounds]
            if not all(ranges):
                continue

            obj.update_clipping_bounds(
                [bound for axis in ranges for bound in (axis[0], axis[1])]
            )

        self.server.controller.view_update()

    def sync_clip_depth(self, **kwargs):
        near, far = self.server.state.clip_depth
        self.scene.renderer.GetActiveCamera().SetClippingRange(near, far)
        self.server.controller.view_update()

    def seed(self):
        """The depth range, each object's toggle, and its bounds from geometry.

        The depth range has no ``Scene`` field: it is where the camera's own
        clipping range sits once the pipeline is built, which is the only
        sensible place for the slider to open.
        """
        super().seed()
        state = self.server.state
        state.clip_depth = list(
            self.scene.renderer.GetActiveCamera().GetClippingRange()
        )

        for obj in self.scene.renderables:
            keys = ObjectState.of(obj)
            state[keys.clip_panel] = []

            if not obj.actors:
                continue

            # The configured crop if there is one, else the object's own extent,
            # which is the whole of it and so crops nothing.
            bounds = obj.crop or obj.combined_bounds
            for key, low in zip(keys.clip_bounds, (0, 2, 4)):
                state[key] = [bounds[low], bounds[low + 1]]

        for volume in self.scene.volumes:
            state[ObjectState.of(volume).preset_panel] = []
