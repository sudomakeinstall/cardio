"""Per-object clipping planes and the camera's depth clipping range."""

# Internal
from ..state import ObjectState
from .base import Controller


class ClippingController(Controller):
    """Clip boxes for every renderable, plus the shared near/far range."""

    def register(self):
        state = self.server.state

        camera = self.scene.renderer.GetActiveCamera()
        state.clip_depth = list(camera.GetClippingRange())

        def reapply_clip(obj, event):
            near, far = state.clip_depth
            if camera.GetClippingRange() != (near, far):
                camera.SetClippingRange(near, far)

        # Held so the observer is not garbage collected
        self._clip_observer = reapply_clip
        self.scene.renderWindow.AddObserver("StartEvent", self._clip_observer)

        state.change("clip_depth")(self.sync_clip_depth)

        for obj in self.scene.renderables:
            state[ObjectState.of(obj).clipping] = obj.clipping_enabled

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

    def _initialize_clipping_state(self):
        """Seed the clip panels and range sliders from each object's bounds."""
        for obj in self.scene.renderables:
            keys = ObjectState.of(obj)
            self.server.state[keys.clip_panel] = []

            if not obj.actors:
                continue

            bounds = obj.combined_bounds
            for key, low in zip(keys.clip_bounds, (0, 2, 4)):
                self.server.state[key] = [bounds[low], bounds[low + 1]]

        for volume in self.scene.volumes:
            keys = ObjectState.of(volume)
            self.server.state[keys.preset] = volume.transfer_function_preset
            self.server.state[keys.preset_panel] = []
