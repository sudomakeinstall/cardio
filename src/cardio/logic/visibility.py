"""Per-object visibility, transfer function presets and the background."""

# Internal
from ..state import ObjectState
from ..volume_property_presets import load_volume_property_preset
from .base import Controller


class VisibilityController(Controller):
    """What is drawn, and in what colours."""

    def register(self):
        state = self.server.state
        state.change("theme_mode")(self.sync_background_color)

        for obj in self.scene.renderables:
            state[ObjectState.of(obj).visibility] = obj.visible

        visibility_keys = [
            ObjectState.of(obj).visibility for obj in self.scene.renderables
        ]
        if visibility_keys:
            state.change(*visibility_keys)(self.sync_visibility)

        preset_keys = [ObjectState.of(v).preset for v in self.scene.volumes]
        if preset_keys:
            state.change(*preset_keys)(self.sync_volume_presets)

    def sync_visibility(self, **kwargs):
        """Show or hide each object's current frame, per its visibility toggle."""
        frame = self.server.state.frame
        for obj in self.scene.renderables:
            actor = obj.frame_actor(frame)
            if actor is not None:
                actor.SetVisibility(self.server.state[ObjectState.of(obj).visibility])
        self.server.controller.view_update()

    def sync_volume_presets(self, **kwargs):
        """Update volume transfer function presets based on UI selection."""
        for v in self.scene.volumes:
            preset_name = self.server.state[ObjectState.of(v).preset]
            preset = load_volume_property_preset(preset_name)

            # Apply preset to all actors
            for actor in v.actors:
                actor.SetProperty(preset.vtk_property)

        self.server.controller.view_update()

    def sync_background_color(self, theme_mode, **kwargs):
        """Sync VTK renderer background with dark mode."""
        if theme_mode == "dark":
            # Dark mode: use dark background from config
            self.scene.renderer.SetBackground(
                *self.scene.background.dark,
            )
        else:
            # Light mode: use light background from config
            self.scene.renderer.SetBackground(
                *self.scene.background.light,
            )
        self.server.controller.view_update()
