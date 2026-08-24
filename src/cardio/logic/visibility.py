"""Per-object visibility, transfer function presets and the background."""

# Internal
from ..state import DEFAULT_THEME_MODE, THEME_DARK, ObjectState
from ..volume_property_presets import load_volume_property_preset
from .base import Controller


class VisibilityController(Controller):
    """What is drawn, and in what colours."""

    def register(self):
        state = self.server.state
        state.setdefault("theme_mode", DEFAULT_THEME_MODE)
        state.change("theme_mode")(self.sync_background_color)
        self.apply_background_color(state.theme_mode)

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

    def background_color(self, theme_mode) -> tuple[float, float, float]:
        background = self.scene.background
        return background.dark if theme_mode == THEME_DARK else background.light

    def apply_background_color(self, theme_mode):
        """Set the VR renderer background for a theme, without a re-render.

        Called at registration, before any view exists to update.
        """
        self.scene.renderer.SetBackground(*self.background_color(theme_mode))

    def sync_background_color(self, theme_mode, **kwargs):
        """Sync VTK renderer background with dark mode."""
        self.apply_background_color(theme_mode)
        self.server.controller.view_update()
