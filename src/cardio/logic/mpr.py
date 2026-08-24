"""The MPR views: active volume, slice pose, window/level, crosshairs, overlays."""

# Third Party
import numpy as np

# Internal
from ..orientation import cumulative_rotation_matrix
from ..state import ObjectState
from ..window_level import presets
from .base import Controller


class MPRController(Controller):
    """Everything that decides what the three MPR views show."""

    def __init__(self, app):
        super().__init__(app)
        self._updating_from_preset = False
        self._pending_active_volume = None

    def register(self):
        state = self.server.state
        state.mpr_presets = []
        state.volume_items = [
            {"text": volume.label, "value": volume.label}
            for volume in self.scene.volumes
        ]
        state.camera_lock = "free"
        state.camera_lock_items = [
            {"title": "Free", "value": "free"},
            {"title": "UL (Axial)", "value": "UL"},
            {"title": "LL (Coronal)", "value": "LL"},
            {"title": "LR (Sagittal)", "value": "LR"},
        ]
        state.mpr_origin = [0.0, 0.0, 0.0]
        state.mpr_crosshairs_enabled = self.scene.mpr_crosshairs_enabled

        state.change("active_volume_label")(self.sync_active_volume)
        state.change("mpr_origin")(self.update_slice_positions)
        state.change("mpr_crosshairs_enabled")(self.sync_crosshairs_visibility)
        state.change("mpr_window", "mpr_level")(self.update_mpr_window_level)
        state.change("mpr_window_level_preset")(self.update_mpr_preset)
        state.change("mpr_rotation_data")(self.update_mpr_rotation)
        state.change("camera_lock")(self._on_camera_lock_change)
        state.change("mpr_segmentation_opacity")(self.update_segmentation_opacity)
        for seg in self.scene.segmentations:
            state.change(ObjectState.of(seg).mpr_overlay)(
                self.sync_segmentation_overlays
            )

        for seg in self.scene.segmentations:
            state[ObjectState.of(seg).mpr_overlay] = False
        state.mpr_segmentation_opacity = 0.7

        self.server.controller.reset_mpr_origin = self.reset_mpr_origin
        self.server.controller.finalize_mpr_initialization = (
            self.finalize_mpr_initialization
        )

    def register_initial_view(self):
        """Seed the MPR view state, once the other controllers have registered.

        ``active_volume_label`` is deliberately left empty until the UI is up;
        finalize_mpr_initialization sets it, avoiding a race with trame's
        listener bookkeeping.
        """
        state = self.server.state
        state.mpr_enabled = self.scene.mpr_enabled
        state.active_volume_label = ""
        self._pending_active_volume = (
            self.scene.volumes[0].label
            if self.scene.volumes and not self.scene.active_volume_label
            else self.scene.active_volume_label
        )
        state.mpr_origin = list(self.scene.mpr_origin)
        state.mpr_window = self.scene.mpr_window
        state.mpr_level = self.scene.mpr_level
        state.mpr_window_level_preset = self.scene.mpr_window_level_preset

        self.app.rotations.publish(self.scene.mpr_rotation_sequence)

        state.mpr_presets = [{"text": "Select W/L...", "value": None}] + [
            {"text": preset.name, "value": key} for key, preset in presets.items()
        ]

        # Set the values the preset implies without driving the views, which
        # may not exist yet.
        if self.scene.mpr_window_level_preset in presets:
            preset = presets[self.scene.mpr_window_level_preset]
            state.mpr_window = preset.window
            state.mpr_level = preset.level

    def update_mpr_frame(self, frame):
        """Update MPR views to show the specified frame."""
        if not getattr(self.server.state, "mpr_enabled", False):
            return

        active_volume_label = getattr(self.server.state, "active_volume_label", "")
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        # Get or create MPR actors for the new frame
        mpr_actors = active_volume.get_mpr_actors_for_frame(frame)

        # CRITICAL: Sync slice positions IMMEDIATELY after creation
        # This ensures actors have correct origin before being added to renderers

        origin = getattr(self.server.state, "mpr_origin", [0.0, 0.0, 0.0])
        rotation_sequence, rotation_angles = self.app.rotations.visible_rotation_data()

        origin = self.convention.point_to_itk(origin)

        active_volume.update_slice_positions(
            frame,
            origin,
            rotation_sequence,
            rotation_angles,
            self.scene.mpr_rotation_sequence.metadata.angle_units,
        )

        for seg in self.scene.segmentations:
            if self.server.state[ObjectState.of(seg).mpr_overlay]:
                seg.update_slice_positions(
                    frame,
                    origin,
                    rotation_sequence,
                    rotation_angles,
                    self.scene.mpr_rotation_sequence.metadata.angle_units,
                )

        views = self.scene.mpr_views
        if views is not None:
            views.show(
                mpr_actors,
                active_volume.crosshair_actors,
                getattr(self.server.state, "mpr_crosshairs_enabled", True),
            )

        # Add segmentation overlays
        self._add_segmentation_overlays_to_mpr(frame)

        # Apply current window/level settings to the MPR actors
        window = getattr(self.server.state, "mpr_window", 400.0)
        level = getattr(self.server.state, "mpr_level", 40.0)
        active_volume.update_mpr_window_level(frame, window, level)

    def sync_active_volume(self, active_volume_label, **kwargs):
        """Handle active volume selection for MPR."""

        if not active_volume_label or not self.server.state.mpr_enabled:
            return

        # Find the selected volume
        active_volume = None
        for volume in self.scene.volumes:
            if volume.label == active_volume_label:
                active_volume = volume
                break

        if not active_volume:
            return

        # Initialize origin to volume center (in LPS coordinates)
        try:
            current_frame = getattr(self.server.state, "frame", 0)
            volume_actor = active_volume.actors[current_frame]
            image_data = volume_actor.GetMapper().GetInput()
            center = image_data.GetCenter()

            # Set origin to volume center if it's at default [0,0,0]
            current_origin = getattr(self.server.state, "mpr_origin", [0.0, 0.0, 0.0])
            if current_origin == [0.0, 0.0, 0.0]:
                # VTK reports the centre in ITK; state holds the user's order
                self.server.state.mpr_origin = self.convention.point_from_itk(center)
        except (RuntimeError, IndexError) as e:
            print(f"Error: Cannot get center for volume '{active_volume_label}': {e}")
            return

        # Create MPR actors for current frame
        current_frame = getattr(self.server.state, "frame", 0)
        mpr_actors = active_volume.get_mpr_actors_for_frame(current_frame)

        # Create crosshair actors
        crosshairs = active_volume.create_crosshair_actors(
            colors=self.scene.mpr_crosshair_colors,
            line_width=self.scene.mpr_crosshair_width,
        )
        crosshairs_visible = getattr(self.server.state, "mpr_crosshairs_enabled", True)

        # A new volume gets the cameras refit; frame changes deliberately do not
        views = self.scene.mpr_views
        if views is not None:
            views.show(mpr_actors, crosshairs, crosshairs_visible, reset_camera=True)

        # Add segmentation overlays
        self._add_segmentation_overlays_to_mpr(current_frame)

        # Apply current window/level settings to the MPR actors
        window = getattr(self.server.state, "mpr_window", 800.0)
        level = getattr(self.server.state, "mpr_level", 200.0)
        active_volume.update_mpr_window_level(current_frame, window, level)

        # Update all views
        self.server.controller.view_update()

    def update_slice_positions(self, **kwargs):
        """Update MPR slice positions when sliders change."""

        if not getattr(self.server.state, "mpr_enabled", False):
            return

        active_volume_label = getattr(self.server.state, "active_volume_label", "")
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        # Get current origin
        origin = getattr(self.server.state, "mpr_origin", [0.0, 0.0, 0.0])
        rotation_sequence, rotation_angles = self.app.rotations.visible_rotation_data()

        origin = self.convention.point_to_itk(origin)

        # CRITICAL FIX: Update ALL cached frames, not just current frame
        # This ensures all frames use the same global origin when switching
        for frame in active_volume._mpr_actors.keys():
            active_volume.update_slice_positions(
                frame,
                origin,
                rotation_sequence,
                rotation_angles,
                self.scene.mpr_rotation_sequence.metadata.angle_units,
            )

        # Update segmentation overlay positions for all cached frames
        for seg in self.scene.segmentations:
            if self.server.state[ObjectState.of(seg).mpr_overlay]:
                for frame in seg._mpr_actors.keys():
                    seg.update_slice_positions(
                        frame,
                        origin,
                        rotation_sequence,
                        rotation_angles,
                        self.scene.mpr_rotation_sequence.metadata.angle_units,
                    )

        self.server.controller.view_update()

    def update_mpr_window_level(self, **kwargs):
        """Update MPR window/level when sliders change."""

        if not getattr(self.server.state, "mpr_enabled", False):
            return

        active_volume_label = getattr(self.server.state, "active_volume_label", "")
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        # Get current window/level values
        window = getattr(self.server.state, "mpr_window", 400.0)
        level = getattr(self.server.state, "mpr_level", 40.0)
        current_frame = getattr(self.server.state, "frame", 0)

        # Check if this change is from manual adjustment (not from preset)
        # by checking if we're not in the middle of a preset update
        if not getattr(self, "_updating_from_preset", False):
            # Reset preset selection when manually adjusting window/level
            current_preset = getattr(self.server.state, "mpr_window_level_preset", None)
            if current_preset is not None:
                self.server.state.mpr_window_level_preset = None

        # Update window/level for MPR actors
        active_volume.update_mpr_window_level(current_frame, window, level)

        # Update all views
        self.server.controller.view_update()

    def update_mpr_preset(self, mpr_window_level_preset, **kwargs):
        """Update MPR window/level when preset changes."""
        # Handle None value (Select W/L... option) - do nothing
        if mpr_window_level_preset is None:
            return

        if mpr_window_level_preset in presets:
            preset = presets[mpr_window_level_preset]

            # Set flag to indicate we're updating from preset
            self._updating_from_preset = True
            try:
                self.server.state.mpr_window = preset.window
                self.server.state.mpr_level = preset.level

                # Update the actual MPR views with new window/level
                self.update_mpr_window_level()
            finally:
                # Always clear the flag
                self._updating_from_preset = False

    def update_mpr_rotation(self, **kwargs):
        """Update MPR views when rotation changes."""

        if self.server.state.rotations_saved_at:
            self.server.state.rotations_stale = True

        if not getattr(self.server.state, "mpr_enabled", False):
            return

        active_volume_label = getattr(self.server.state, "active_volume_label", "")
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        # Get current origin and frame
        origin = getattr(self.server.state, "mpr_origin", [0.0, 0.0, 0.0])
        current_frame = getattr(self.server.state, "frame", 0)

        rotation_sequence, rotation_angles = self.app.rotations.visible_rotation_data()

        origin = self.convention.point_to_itk(origin)

        # Update slice positions with rotation
        active_volume.update_slice_positions(
            current_frame,
            origin,
            rotation_sequence,
            rotation_angles,
            self.scene.mpr_rotation_sequence.metadata.angle_units,
        )

        # Update segmentation overlay positions
        for seg in self.scene.segmentations:
            if self.server.state[ObjectState.of(seg).mpr_overlay]:
                seg.update_slice_positions(
                    current_frame,
                    origin,
                    rotation_sequence,
                    rotation_angles,
                    self.scene.mpr_rotation_sequence.metadata.angle_units,
                )

        self._sync_vr_camera_to_mpr()
        self.server.controller.view_update()

    def _sync_vr_camera_to_mpr(self):
        lock = self.server.state.camera_lock
        if lock == "free":
            return

        orientation = {"UL": "axial", "LL": "coronal", "LR": "sagittal"}[lock]

        # Base slice normals and up vectors in LPS coordinates.
        # Normal = out-of-plane direction; up = Y axis of the reslice frame.
        base_normals = {
            "axial": np.array([0.0, 0.0, 1.0]),  # Superior (Z in LAS)
            "sagittal": np.array([1.0, 0.0, 0.0]),  # Left (Z in ASL)
            "coronal": np.array([0.0, 1.0, 0.0]),  # Posterior (Z in LSA)
        }
        base_ups = {
            "axial": np.array([0.0, -1.0, 0.0]),  # Anterior (Y in LAS)
            "sagittal": np.array([0.0, 0.0, 1.0]),  # Superior (Y in ASL)
            "coronal": np.array([0.0, 0.0, 1.0]),  # Superior (Y in LSA)
        }

        normal = base_normals[orientation]
        up = base_ups[orientation]

        active_volume = self._active_volume()
        if active_volume is not None:
            rotation_sequence, rotation_angles = (
                self.app.rotations.visible_rotation_data()
            )
            rotation = cumulative_rotation_matrix(
                rotation_sequence,
                rotation_angles,
                self.scene.mpr_rotation_sequence.metadata.angle_units,
            )
            normal = rotation @ normal
            up = rotation @ up

        vr_renderer = self.scene.renderer
        vr_cam = vr_renderer.GetActiveCamera()
        vr_fp = np.array(vr_cam.GetFocalPoint())
        vr_pos = np.array(vr_cam.GetPosition())
        vr_dist = np.linalg.norm(vr_fp - vr_pos)

        vr_cam.SetPosition(*(vr_fp - normal * vr_dist))
        vr_cam.SetViewUp(*up)
        vr_renderer.ResetCameraClippingRange()

    def _on_camera_lock_change(self, camera_lock, **kwargs):
        self._sync_vr_camera_to_mpr()
        self.server.controller.view_update()

    def sync_crosshairs_visibility(self, **kwargs):
        """Toggle crosshair visibility on all MPR views."""
        if not getattr(self.server.state, "mpr_enabled", False):
            return

        active_volume_label = getattr(self.server.state, "active_volume_label", "")
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        visible = getattr(self.server.state, "mpr_crosshairs_enabled", True)
        active_volume.set_crosshairs_visible(visible)
        self.server.controller.view_update()

    def reset_mpr_origin(self):
        active_volume_label = getattr(self.server.state, "active_volume_label", "")
        active_volume = self._active_volume()
        if not active_volume:
            return
        current_frame = getattr(self.server.state, "frame", 0)
        image_data = active_volume.actors[current_frame].GetMapper().GetInput()
        self.server.state.mpr_origin = self.convention.point_from_itk(
            image_data.GetCenter()
        )

    def set_origin(self, center):
        """Write a centroid to mpr_origin, converting out of ITK if needed."""
        self.server.state.mpr_origin = self.convention.point_from_itk(center)

    def finalize_mpr_initialization(self, **kwargs):
        """Set the active volume label after UI is ready to avoid race condition."""
        if hasattr(self, "_pending_active_volume") and self._pending_active_volume:
            # Flush immediately so trame-server clears its listener-key accumulator
            # (_listener_keys in 3.12+). Without the flush, active_volume_label
            # remains pending past state.initial (called when the client connects),
            # causing sync_active_volume to fire spuriously on the next unrelated
            # state update (e.g. label selection in the snap UI).
            with self.server.state:
                self.server.state.active_volume_label = self._pending_active_volume
            delattr(self, "_pending_active_volume")

        # Apply loaded rotation data to MPR views
        self.update_mpr_rotation()

    def _add_segmentation_overlays_to_mpr(self, frame: int):
        """Pose and add the enabled segmentation overlays on top of the views."""
        views = self.scene.mpr_views
        opacity = self.server.state.mpr_segmentation_opacity
        origin = self.convention.point_to_itk(
            getattr(self.server.state, "mpr_origin", [0.0, 0.0, 0.0])
        )
        rotation_sequence, rotation_angles = self.app.rotations.visible_rotation_data()

        for seg in self.scene.segmentations:
            if not self.server.state[ObjectState.of(seg).mpr_overlay]:
                continue

            overlay = seg.get_mpr_actors_for_frame(frame)
            seg.update_mpr_opacity(frame, opacity)
            seg.update_slice_positions(
                frame,
                origin,
                rotation_sequence,
                rotation_angles,
                self.scene.mpr_rotation_sequence.metadata.angle_units,
            )

            if views is not None:
                views.add_overlay(overlay)

    def sync_segmentation_overlays(self, **kwargs):
        """Toggle segmentation overlay visibility on MPR views."""
        if not self.server.state.mpr_enabled:
            return

        views = self.scene.mpr_views
        if views is None:
            return

        current_frame = self.server.state.frame
        active_volume = self._active_volume()

        views.clear()
        if active_volume:
            views.set_image(active_volume.get_mpr_actors_for_frame(current_frame))
            views.add_crosshairs(
                active_volume.crosshair_actors,
                getattr(self.server.state, "mpr_crosshairs_enabled", True),
            )

        # Poses the overlays as it adds them, so no second pass is needed here
        self._add_segmentation_overlays_to_mpr(current_frame)

        self.server.controller.view_update()

    def scroll_vector(self, view_name: str):
        """The out-of-plane direction of a view, in the user's index order.

        Composed in ITK -- the only order the rotation math is defined in --
        then handed back in the convention ``mpr_origin`` is stored in.
        """
        base_normals = {
            "axial": np.array([0.0, 0.0, 1.0]),
            "sagittal": np.array([1.0, 0.0, 0.0]),
            "coronal": np.array([0.0, 1.0, 0.0]),
        }
        if view_name not in base_normals:
            return np.array([0.0, 0.0, 1.0])

        convention = self.convention
        sequence, angles = self.app.rotations.visible_rotation_data()
        rotation = cumulative_rotation_matrix(sequence, angles, convention.angle_units)

        return np.array(convention.point_from_itk(rotation @ base_normals[view_name]))

    def scroll_slice(self, view_name: str, distance: float):
        """Move the shared origin along ``view_name``'s out-of-plane direction."""
        if view_name not in ("axial", "sagittal", "coronal"):
            return

        origin = getattr(self.server.state, "mpr_origin", [0.0, 0.0, 0.0])
        step = self.scroll_vector(view_name)
        self.server.state.mpr_origin = [
            origin[i] + distance * step[i] for i in range(3)
        ]

    def adjust_window_level(self, window_delta: float, level_delta: float):
        """Nudge the MPR window and level, keeping the window positive."""
        state = self.server.state
        state.mpr_window = max(1.0, state.mpr_window + window_delta)
        state.mpr_level = state.mpr_level + level_delta

    def update_segmentation_opacity(self, **kwargs):
        """Update segmentation overlay opacity."""
        if not self.server.state.mpr_enabled:
            return

        current_frame = self.server.state.frame
        opacity = self.server.state.mpr_segmentation_opacity

        for seg in self.scene.segmentations:
            if self.server.state[ObjectState.of(seg).mpr_overlay]:
                seg.update_mpr_opacity(current_frame, opacity)

        self.server.controller.view_update()
