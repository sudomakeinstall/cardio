import asyncio
import datetime as dt

import numpy as np
from trame.app import asynchronous

from .convention import Convention, exchange_point, exchange_step
from .orientation import cumulative_rotation_matrix
from .scene import Scene
from .screenshot import Screenshot

ALIGN_STEP_NAME = "Interface plane"


class Logic:
    def _active_volume(self):
        """The volume the MPR views are showing, or None if there isn't one."""
        label = getattr(self.server.state, "active_volume_label", "")
        return next((v for v in self.scene.volumes if v.label == label), None)

    @property
    def convention(self) -> Convention:
        """The index order and angle units the MPR state is currently written in."""
        return Convention.from_metadata(self.scene.mpr_rotation_sequence.metadata)

    def _get_visible_rotation_data(self):
        """Rotation sequence and angles for the visible steps, in ITK for VTK."""
        rotation_data = getattr(
            self.server.state, "mpr_rotation_data", {"angles_list": []}
        )
        return self.convention.visible_sequence_to_itk(
            rotation_data.get("angles_list", [])
        )

    def __init__(self, server, scene: Scene):
        self.server = server
        self.scene = scene
        self._invalidate_lock_cache()

        # Playback timing state
        self._playback_start_time = None
        self._last_render_duration = 0.0
        self._last_target_frame = None
        self._is_rendering = False
        self._playback_task = None

        # Initialize mpr_presets early to avoid undefined errors
        self.server.state.mpr_presets = []

        # Initialize volume items for MPR dropdown
        self.server.state.volume_items = [
            {"text": volume.label, "value": volume.label}
            for volume in self.scene.volumes
        ]

        # Initialize angle units items for dropdown
        self.server.state.angle_units_items = [
            {"text": "Degrees", "value": "degrees"},
            {"text": "Radians", "value": "radians"},
        ]

        # Initialize axis convention items for dropdown
        self.server.state.index_order_items = [
            {"text": "ITK (X=L, Y=P, Z=S)", "value": "itk"},
            {"text": "Roma (X=S, Y=P, Z=L)", "value": "roma"},
        ]

        self.server.state.camera_lock = "free"
        self.server.state.camera_lock_items = [
            {"title": "Free", "value": "free"},
            {"title": "UL (Axial)", "value": "UL"},
            {"title": "LL (Coronal)", "value": "LL"},
            {"title": "LR (Sagittal)", "value": "LR"},
        ]

        # Initialize MPR origin (will be updated when active volume changes)
        self.server.state.mpr_origin = [0.0, 0.0, 0.0]

        camera = self.scene.renderer.GetActiveCamera()
        self.server.state.clip_depth = list(camera.GetClippingRange())

        def _reapply_clip(obj, event):
            near, far = self.server.state.clip_depth
            if camera.GetClippingRange() != (near, far):
                camera.SetClippingRange(near, far)

        self._clip_observer = _reapply_clip
        self.scene.renderWindow.AddObserver("StartEvent", self._clip_observer)

        self.server.state.mpr_crosshairs_enabled = self.scene.mpr_crosshairs_enabled

        self.server.state.change("frame")(self.update_frame)
        self.server.state.change("playing")(self._handle_playing_change)
        self.server.state.change("theme_mode")(self.sync_background_color)
        self.server.state.change("active_volume_label")(self.sync_active_volume)
        self.server.state.change("mpr_origin")(self.update_slice_positions)
        self.server.state.change("mpr_crosshairs_enabled")(
            self.sync_crosshairs_visibility
        )
        self.server.state.change("mpr_window", "mpr_level")(
            self.update_mpr_window_level
        )
        self.server.state.change("mpr_window_level_preset")(self.update_mpr_preset)
        self.server.state.change("mpr_rotation_data")(self.update_mpr_rotation)
        self.server.state.change("angle_units")(self.sync_angle_units)
        self.server.state.change("index_order")(self.sync_index_order)
        self.server.state.change("clip_depth")(self.sync_clip_depth)
        self.server.state.change("camera_lock")(self._on_camera_lock_change)
        self.server.state.change("mpr_segmentation_opacity")(
            self.update_segmentation_opacity
        )
        for s in self.scene.segmentations:
            self.server.state.change(f"mpr_segmentation_overlay_{s.label}")(
                self.sync_segmentation_overlays
            )

        if self.scene.segmentations:
            self.server.state.change("snap_seg_label")(self._on_snap_seg_changed)
            self.server.state.change("snap_mode", "snap_labels_a", "snap_labels_b")(
                self._on_snap_selection_changed
            )
            self.server.state.change("snap_locked")(self._on_snap_lock_changed)
            self.server.state.change("snap_orientation_locked")(
                self._on_snap_orientation_lock_changed
            )

        # Initialize visibility state variables
        for m in self.scene.meshes:
            self.server.state[f"mesh_visibility_{m.label}"] = m.visible
        for v in self.scene.volumes:
            self.server.state[f"volume_visibility_{v.label}"] = v.visible
        for s in self.scene.segmentations:
            self.server.state[f"segmentation_visibility_{s.label}"] = s.visible

        # Initialize MPR overlay state variables
        for s in self.scene.segmentations:
            self.server.state[f"mpr_segmentation_overlay_{s.label}"] = False
        self.server.state.mpr_segmentation_opacity = 0.7

        if self.scene.segmentations:
            self.server.state.snap_mode = "label"
            self.server.state.snap_seg_label = self.scene.segmentations[0].label
            self.server.state.snap_labels_a = []
            self.server.state.snap_labels_b = []
            self.server.state.snap_available_labels = []
            self.server.state.snap_seg_items = [
                {"title": s.label, "value": s.label} for s in self.scene.segmentations
            ]
            self.server.state.snap_no_interface = False
            self.server.state.snap_locked = False
            self.server.state.interface_flatness = 0.0
            self.server.state.snap_orientation_locked = False

        # Initialize preset state variables
        for v in self.scene.volumes:
            self.server.state[f"volume_preset_{v.label}"] = v.transfer_function_preset

        # Initialize clipping state variables
        for m in self.scene.meshes:
            self.server.state[f"mesh_clipping_{m.label}"] = m.clipping_enabled
        for v in self.scene.volumes:
            self.server.state[f"volume_clipping_{v.label}"] = v.clipping_enabled
        for s in self.scene.segmentations:
            self.server.state[f"segmentation_clipping_{s.label}"] = s.clipping_enabled
        self.server.state.change(
            *[f"mesh_visibility_{m.label}" for m in self.scene.meshes]
        )(self.sync_mesh_visibility)
        self.server.state.change(
            *[f"volume_visibility_{v.label}" for v in self.scene.volumes]
        )(self.sync_volume_visibility)
        self.server.state.change(
            *[f"segmentation_visibility_{s.label}" for s in self.scene.segmentations]
        )(self.sync_segmentation_visibility)
        self.server.state.change(
            *[f"volume_preset_{v.label}" for v in self.scene.volumes]
        )(self.sync_volume_presets)

        # Set up mesh clipping controls
        mesh_clipping_controls = []
        for m in self.scene.meshes:
            mesh_clipping_controls.extend(
                [
                    f"mesh_clipping_{m.label}",
                    f"clip_x_{m.label}",
                    f"clip_y_{m.label}",
                    f"clip_z_{m.label}",
                ]
            )
        if mesh_clipping_controls:
            self.server.state.change(*mesh_clipping_controls)(self.sync_mesh_clipping)

        # Set up volume clipping controls
        volume_clipping_controls = []
        for v in self.scene.volumes:
            volume_clipping_controls.extend(
                [
                    f"volume_clipping_{v.label}",
                    f"clip_x_{v.label}",
                    f"clip_y_{v.label}",
                    f"clip_z_{v.label}",
                ]
            )
        if volume_clipping_controls:
            self.server.state.change(*volume_clipping_controls)(
                self.sync_volume_clipping
            )

        # Set up segmentation clipping controls
        segmentation_clipping_controls = []
        for s in self.scene.segmentations:
            segmentation_clipping_controls.extend(
                [
                    f"segmentation_clipping_{s.label}",
                    f"clip_x_{s.label}",
                    f"clip_y_{s.label}",
                    f"clip_z_{s.label}",
                ]
            )
        if segmentation_clipping_controls:
            self.server.state.change(*segmentation_clipping_controls)(
                self.sync_segmentation_clipping
            )

        self.server.controller.increment_frame = self.increment_frame
        self.server.controller.decrement_frame = self.decrement_frame
        self.server.controller.screenshot = self.screenshot
        self.server.controller.save_rotation_angles = self.save_rotation_angles
        self.server.controller.reset_all = self.reset_all
        self.server.controller.close_application = self.close_application
        self.server.controller.finalize_mpr_initialization = (
            self.finalize_mpr_initialization
        )

        # MPR rotation controllers
        self.server.controller.add_x_rotation = lambda: self.add_mpr_rotation("X")
        self.server.controller.add_y_rotation = lambda: self.add_mpr_rotation("Y")
        self.server.controller.add_z_rotation = lambda: self.add_mpr_rotation("Z")
        self.server.controller.remove_rotation_event = self.remove_mpr_rotation
        self.server.controller.reset_rotation_angle = self.reset_rotation_angle
        self.server.controller.reset_rotations = self.reset_mpr_rotations
        self.server.controller.reset_mpr_origin = self.reset_mpr_origin
        self.server.controller.snap_to_centroid = self.snap_to_centroid
        self.server.controller.align_to_interface = self.align_to_interface

        # Initialize screenshot viewport state
        for viewport in ("vr", "axial", "coronal", "sagittal"):
            self.server.state[f"screenshot_viewport_{viewport}"] = (
                viewport in self.scene.screenshot_viewports
            )

        # Initialize MPR state
        self.server.state.mpr_enabled = self.scene.mpr_enabled
        # Initialize active_volume_label to empty string (actual value set after UI ready)
        self.server.state.active_volume_label = ""
        # Store the actual volume label to set later, avoiding race condition
        self._pending_active_volume = (
            self.scene.volumes[0].label
            if self.scene.volumes and not self.scene.active_volume_label
            else self.scene.active_volume_label
        )
        self.server.state.mpr_origin = list(self.scene.mpr_origin)
        self.server.state.mpr_window = self.scene.mpr_window
        self.server.state.mpr_level = self.scene.mpr_level
        self.server.state.mpr_window_level_preset = self.scene.mpr_window_level_preset

        # Initialize rotation data from RotationSequence (includes metadata)
        self.server.state.mpr_rotation_data = (
            self.scene.mpr_rotation_sequence.model_dump(mode="json")
        )

        # Keep mirror variables for UI binding convenience
        self.server.state.angle_units = self.server.state.mpr_rotation_data["metadata"][
            "angle_units"
        ]
        self.server.state.index_order = self.server.state.mpr_rotation_data["metadata"][
            "index_order"
        ]

        # Initialize MPR presets data
        try:
            from .window_level import presets

            self.server.state.mpr_presets = [
                {"text": "Select W/L...", "value": None}
            ] + [{"text": preset.name, "value": key} for key, preset in presets.items()]
        except Exception as e:
            print(f"Error initializing MPR presets: {e}")
            self.server.state.mpr_presets = []

        # Apply initial preset to ensure window/level values are set correctly
        # Only update state values, don't call update methods yet since MPR may not be enabled
        from .window_level import presets

        if self.scene.mpr_window_level_preset in presets:
            preset = presets[self.scene.mpr_window_level_preset]
            self.server.state.mpr_window = preset.window
            self.server.state.mpr_level = preset.level

        # Initialize clipping state variables
        self._initialize_clipping_state()

        if self.scene.segmentations:
            self._on_snap_seg_changed()

    def update_frame(self, frame, **kwargs):
        # Before update_mpr_frame reads mpr_origin, so the new frame is
        # positioned on its own centroid rather than the previous frame's.
        self.apply_frame_lock(frame)

        self.scene.hide_all_frames()

        # Show frame with server state visibility
        for mesh in self.scene.meshes:
            visible = self.server.state[f"mesh_visibility_{mesh.label}"]
            if visible:
                mesh.actors[frame % len(mesh.actors)].SetVisibility(True)

        for volume in self.scene.volumes:
            visible = self.server.state[f"volume_visibility_{volume.label}"]
            if visible:
                volume.actors[frame % len(volume.actors)].SetVisibility(True)

        for segmentation in self.scene.segmentations:
            visible = self.server.state[f"segmentation_visibility_{segmentation.label}"]
            if visible:
                actor = segmentation.actors[frame % len(segmentation.actors)]
                actor.SetVisibility(True)

        # Update MPR views if MPR is enabled
        self.update_mpr_frame(frame)

        self.server.controller.view_update()

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
        rotation_sequence, rotation_angles = self._get_visible_rotation_data()

        origin = self.convention.point_to_itk(origin)

        active_volume.update_slice_positions(
            frame,
            origin,
            rotation_sequence,
            rotation_angles,
            self.scene.mpr_rotation_sequence.metadata.angle_units,
        )

        for seg in self.scene.segmentations:
            if self.server.state[f"mpr_segmentation_overlay_{seg.label}"]:
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

    def _handle_playing_change(self, playing, **kwargs):
        """Handle playback state changes with task cancellation support."""
        from trame.app import asynchronous

        # Cancel existing playback task if any
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            self._playback_task = None

        # Start new playback task if playing
        if playing:
            self._playback_task = asynchronous.create_task(
                self._play_loop(playing, **kwargs)
            )

    def _calculate_target_frame(self, elapsed_seconds, bpm, nframes):
        """Calculate target frame based on elapsed time.

        Args:
            elapsed_seconds: Time since playback started
            bpm: Beats per minute (playback speed)
            nframes: Total number of frames

        Returns:
            Target frame index (0 to nframes-1)
        """
        cycle_duration = 60.0 / bpm
        cycles_elapsed = elapsed_seconds / cycle_duration
        fractional_frame = (cycles_elapsed * nframes) % nframes
        return int(fractional_frame)

    async def _play_loop(self, playing, **kwargs):
        """Robust cine playback loop with adaptive timing and frame skipping.

        Addresses:
        - Time-based frame calculation (not frame-increment based)
        - Render time accounting with adaptive sleep
        - Frequent cancellation checks for responsiveness
        - Frame skipping when behind schedule
        - Render debouncing to prevent concurrent renders
        - Task cancellation support for immediate pause
        """
        # Validate that at least one playback mode is active
        if not (self.server.state.incrementing or self.server.state.rotating):
            self.server.state.playing = False
            return

        # Initialize playback timing state
        import time

        self._playback_start_time = time.perf_counter()
        self._last_target_frame = self.server.state.frame
        self._last_render_duration = 0.0

        # Playback parameters
        CHECK_INTERVAL = 0.01  # Check pause flag every 10ms for responsiveness

        try:
            while self.server.state.playing:
                # Calculate elapsed time and target frame
                elapsed = time.perf_counter() - self._playback_start_time

                # Get current playback parameters from state
                with self.server.state as state:
                    bpm = state.bpm
                    nframes = self.scene.nframes
                    incrementing = state.incrementing
                    rotating = state.rotating
                    bpr = state.bpr

                # Calculate target frame from elapsed time (time-based, not frame-based)
                target_frame = self._calculate_target_frame(elapsed, bpm, nframes)

                # Determine if render is needed
                frame_changed = incrementing and (
                    target_frame != self._last_target_frame
                )
                needs_rotation = rotating
                needs_render = frame_changed or needs_rotation

                # Render only if needed and not already rendering (debouncing)
                if needs_render and not self._is_rendering:
                    self._is_rendering = True
                    render_start = time.perf_counter()

                    try:
                        with self.server.state as state:
                            # Update frame if incrementing
                            if incrementing and frame_changed:
                                state.frame = target_frame
                                self._last_target_frame = target_frame

                            # Rotate camera if rotating
                            if rotating:
                                deg = 360 / (nframes * bpr)
                                self.scene.renderer.GetActiveCamera().Azimuth(deg)

                            # Synchronous blocking render
                            self.server.controller.view_update()

                        # Track render duration for adaptive timing
                        self._last_render_duration = time.perf_counter() - render_start

                    finally:
                        self._is_rendering = False

                # Adaptive sleep interval calculation
                base_interval = 60.0 / bpm / nframes
                adjusted_interval = max(
                    CHECK_INTERVAL, base_interval - self._last_render_duration
                )

                # Sleep in small chunks to remain responsive to pause
                remaining_sleep = adjusted_interval
                while remaining_sleep > 0 and self.server.state.playing:
                    sleep_chunk = min(CHECK_INTERVAL, remaining_sleep)
                    await asyncio.sleep(sleep_chunk)
                    remaining_sleep -= sleep_chunk

                    # Early exit check after each sleep chunk
                    if not self.server.state.playing:
                        break

        except asyncio.CancelledError:
            # Task was cancelled (pause button pressed) - exit gracefully
            pass
        finally:
            # Clean up playback state
            self._playback_start_time = None
            self._last_target_frame = None
            self._is_rendering = False

    def sync_mesh_visibility(self, **kwargs):
        for m in self.scene.meshes:
            visible = self.server.state[f"mesh_visibility_{m.label}"]
            m.actors[self.server.state.frame % len(m.actors)].SetVisibility(visible)
        self.server.controller.view_update()

    def sync_volume_visibility(self, **kwargs):
        for v in self.scene.volumes:
            visible = self.server.state[f"volume_visibility_{v.label}"]
            v.actors[self.server.state.frame % len(v.actors)].SetVisibility(visible)
        self.server.controller.view_update()

    def sync_segmentation_visibility(self, **kwargs):
        for s in self.scene.segmentations:
            visible = self.server.state[f"segmentation_visibility_{s.label}"]
            actor = s.actors[self.server.state.frame % len(s.actors)]
            actor.SetVisibility(visible)
        self.server.controller.view_update()

    def sync_volume_presets(self, **kwargs):
        """Update volume transfer function presets based on UI selection."""
        from .volume_property_presets import load_volume_property_preset

        for v in self.scene.volumes:
            preset_name = self.server.state[f"volume_preset_{v.label}"]
            preset = load_volume_property_preset(preset_name)

            # Apply preset to all actors
            for actor in v.actors:
                actor.SetProperty(preset.vtk_property)

        self.server.controller.view_update()

    def sync_mesh_clipping(self, **kwargs):
        """Update mesh clipping based on UI controls."""
        for m in self.scene.meshes:
            # Toggle clipping on/off
            clipping_enabled = self.server.state[f"mesh_clipping_{m.label}"]
            m.toggle_clipping(clipping_enabled)

            # Update clipping bounds from sliders
            if clipping_enabled and hasattr(self.server.state, f"clip_x_{m.label}"):
                x_bounds = getattr(self.server.state, f"clip_x_{m.label}")
                y_bounds = getattr(self.server.state, f"clip_y_{m.label}")
                z_bounds = getattr(self.server.state, f"clip_z_{m.label}")

                bounds = [
                    x_bounds[0],
                    x_bounds[1],  # x_min, x_max
                    y_bounds[0],
                    y_bounds[1],  # y_min, y_max
                    z_bounds[0],
                    z_bounds[1],  # z_min, z_max
                ]

                m.update_clipping_bounds(bounds)

        self.server.controller.view_update()

    def sync_volume_clipping(self, **kwargs):
        """Update volume clipping based on UI controls."""
        for v in self.scene.volumes:
            # Toggle clipping on/off
            clipping_enabled = self.server.state[f"volume_clipping_{v.label}"]
            v.toggle_clipping(clipping_enabled)

            # Update clipping bounds if enabled
            if clipping_enabled:
                x_range = self.server.state[f"clip_x_{v.label}"]
                y_range = self.server.state[f"clip_y_{v.label}"]
                z_range = self.server.state[f"clip_z_{v.label}"]

                if x_range and y_range and z_range:
                    bounds = [
                        x_range[0],
                        x_range[1],
                        y_range[0],
                        y_range[1],
                        z_range[0],
                        z_range[1],
                    ]
                    v.update_clipping_bounds(bounds)

        self.server.controller.view_update()

    def sync_segmentation_clipping(self, **kwargs):
        """Update segmentation clipping based on UI controls."""
        for s in self.scene.segmentations:
            # Toggle clipping on/off
            clipping_enabled = self.server.state[f"segmentation_clipping_{s.label}"]
            s.toggle_clipping(clipping_enabled)

            # Update clipping bounds if enabled
            if clipping_enabled:
                x_range = self.server.state[f"clip_x_{s.label}"]
                y_range = self.server.state[f"clip_y_{s.label}"]
                z_range = self.server.state[f"clip_z_{s.label}"]

                if x_range and y_range and z_range:
                    bounds = [
                        x_range[0],
                        x_range[1],
                        y_range[0],
                        y_range[1],
                        z_range[0],
                        z_range[1],
                    ]
                    s.update_clipping_bounds(bounds)

        self.server.controller.view_update()

    def sync_clip_depth(self, **kwargs):
        near, far = self.server.state.clip_depth
        self.scene.renderer.GetActiveCamera().SetClippingRange(near, far)
        self.server.controller.view_update()

    def increment_frame(self):
        if not self.server.state.playing:
            self.server.state.frame = (self.server.state.frame + 1) % self.scene.nframes
            self.server.controller.view_update()

    def decrement_frame(self):
        if not self.server.state.playing:
            self.server.state.frame = (self.server.state.frame - 1) % self.scene.nframes
            self.server.controller.view_update()

    @asynchronous.task
    async def screenshot(self):
        dr = dt.datetime.now().strftime(self.scene.timestamp_format)
        dr = self.scene.screenshot_directory / dr

        mpr_enabled = getattr(self.server.state, "mpr_enabled", False)
        selected = {
            name
            for name in ("vr", "axial", "coronal", "sagittal")
            if getattr(self.server.state, f"screenshot_viewport_{name}", True)
        }
        render_windows = {}
        if "vr" in selected:
            render_windows["vr"] = self.scene.renderWindow
        if mpr_enabled:
            for name in ("axial", "coronal", "sagittal"):
                if name in selected:
                    render_windows[name] = self.scene.mpr_views[name]

        for folder in render_windows:
            (dr / folder).mkdir(parents=True, exist_ok=True)

        def _save_all(i):
            for folder, rw in render_windows.items():
                Screenshot(rw).save(str(dr / folder / f"{i}.png"))

        if not (self.server.state.incrementing or self.server.state.rotating):
            _save_all(0)
        else:
            n = self.scene.nframes
            if self.server.state.rotating:
                n *= self.server.state.bpr
            deg = 360 / (self.scene.nframes * self.server.state.bpr)
            for i in range(n):
                with self.server.state:
                    if self.server.state.rotating:
                        self.scene.renderer.GetActiveCamera().Azimuth(deg)
                    if self.server.state.incrementing:
                        self.increment_frame()
                    self.server.controller.view_update()
                    _save_all(i)
                    await asyncio.sleep(
                        1 / self.server.state.bpm * 60 / self.scene.nframes
                    )

    @asynchronous.task
    async def save_rotation_angles(self):
        """Save current rotation angles to TOML file."""
        from .rotation import RotationSequence

        timestamp = dt.datetime.now()
        timestamp_str = timestamp.strftime(self.scene.timestamp_format)
        active_volume_label = getattr(self.server.state, "active_volume_label", "")

        if not active_volume_label:
            print("Warning: No active volume selected")
            return

        save_dir = self.scene.rotations_directory / active_volume_label
        save_dir.mkdir(parents=True, exist_ok=True)

        rotation_data = getattr(self.server.state, "mpr_rotation_data", {})

        # Create RotationSequence directly from full data structure
        rotation_seq = RotationSequence(**rotation_data)

        # Update only timestamp and volume_label (rest already in metadata)
        rotation_seq.metadata.timestamp = timestamp.isoformat()
        rotation_seq.metadata.volume_label = active_volume_label

        output_path = save_dir / f"{timestamp_str}.toml"
        rotation_seq.to_file(output_path)

        with self.server.state as state:
            state.rotations_saved_at = timestamp.strftime("%H:%M:%S")
            state.rotations_stale = False

    def reset_all(self):
        self.server.state.frame = 0
        self.server.state.playing = False
        self.server.state.incrementing = True
        self.server.state.rotating = False
        self.server.state.bpm = 60
        self.server.state.bpr = 5
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

    def sync_angle_units(self, angle_units, **kwargs):
        """Sync angle units selection - updates the scene configuration."""
        import copy

        import numpy as np

        from .orientation import AngleUnits

        # Get current units before changing
        old_units = self.scene.mpr_rotation_sequence.metadata.angle_units

        # Update based on UI selection
        new_units = None
        if angle_units == "degrees":
            new_units = AngleUnits.DEGREES
        elif angle_units == "radians":
            new_units = AngleUnits.RADIANS

        if new_units is None or old_units == new_units:
            return

        # Get current rotation data (now includes metadata)
        rotation_data = copy.deepcopy(
            getattr(self.server.state, "mpr_rotation_data", {})
        )

        # Convert all existing rotation angles (skip quaternion steps)
        if rotation_data.get("angles_list"):
            for rotation in rotation_data["angles_list"]:
                if rotation.get("quaternion") is not None:
                    continue
                current_angle = rotation.get("angle", 0)

                # Convert based on old -> new units
                if old_units == AngleUnits.DEGREES and new_units == AngleUnits.RADIANS:
                    rotation["angle"] = np.radians(current_angle)
                elif (
                    old_units == AngleUnits.RADIANS and new_units == AngleUnits.DEGREES
                ):
                    rotation["angle"] = np.degrees(current_angle)

        # Update nested metadata
        rotation_data["metadata"]["angle_units"] = angle_units

        # Update state (triggers re-render)
        self.server.state.mpr_rotation_data = rotation_data

        # Update scene
        self.scene.mpr_rotation_sequence.metadata.angle_units = new_units

    def sync_index_order(self, index_order, **kwargs):
        """Sync index order selection - converts existing rotations and updates scene."""
        import copy

        from .orientation import IndexOrder

        old_convention = self.scene.mpr_rotation_sequence.metadata.index_order

        # Convert string input to enum
        if isinstance(index_order, str):
            match index_order.lower():
                case "itk":
                    new_convention = IndexOrder.ITK
                case "roma":
                    new_convention = IndexOrder.ROMA
                case _:
                    raise ValueError(f"Unrecognized index order: {index_order}")
        else:
            new_convention = index_order

        if old_convention == new_convention:
            return

        # Get current rotation data (now includes metadata)
        rotation_data = copy.deepcopy(
            getattr(self.server.state, "mpr_rotation_data", {})
        )

        # Re-express every step in the new order. The exchange is its own
        # inverse, so one mapping serves both ITK<->ROMA directions -- and it
        # always applies here, since the two conventions are known to differ.
        rotation_data["angles_list"] = [
            exchange_step(step) for step in rotation_data.get("angles_list") or []
        ]

        # Update nested metadata
        rotation_data["metadata"]["index_order"] = index_order

        # Update state (triggers re-render)
        self.server.state.mpr_rotation_data = rotation_data

        mpr_origin = getattr(self.server.state, "mpr_origin", None)
        if mpr_origin is not None and len(mpr_origin) == 3:
            self.server.state.mpr_origin = exchange_point(mpr_origin)

        # Update scene
        self.scene.mpr_rotation_sequence.metadata.index_order = new_convention

    def _initialize_clipping_state(self):
        """Initialize clipping state variables for all objects."""
        # Initialize mesh clipping state
        for m in self.scene.meshes:
            # Initialize panel state
            setattr(self.server.state, f"clip_panel_{m.label}", [])

            # Initialize range sliders with mesh bounds if available
            if m.actors:
                bounds = m.combined_bounds
                setattr(self.server.state, f"clip_x_{m.label}", [bounds[0], bounds[1]])
                setattr(self.server.state, f"clip_y_{m.label}", [bounds[2], bounds[3]])
                setattr(self.server.state, f"clip_z_{m.label}", [bounds[4], bounds[5]])

        # Initialize volume clipping state
        for v in self.scene.volumes:
            preset_key = getattr(v, "transfer_function_preset", "cardiac")
            setattr(self.server.state, f"volume_preset_{v.label}", preset_key)

            # Initialize preset panel state (collapsed by default)
            setattr(self.server.state, f"preset_panel_{v.label}", [])
            if hasattr(v, "clipping_enabled"):
                # Initialize clipping checkbox state
                setattr(
                    self.server.state, f"volume_clipping_{v.label}", v.clipping_enabled
                )

                # Initialize panel state
                setattr(self.server.state, f"clip_panel_{v.label}", [])

                # Initialize range sliders with volume bounds if available
                if v.actors:
                    bounds = v.combined_bounds
                    setattr(
                        self.server.state, f"clip_x_{v.label}", [bounds[0], bounds[1]]
                    )
                    setattr(
                        self.server.state, f"clip_y_{v.label}", [bounds[2], bounds[3]]
                    )
                    setattr(
                        self.server.state, f"clip_z_{v.label}", [bounds[4], bounds[5]]
                    )

        # Initialize segmentation clipping state
        for s in self.scene.segmentations:
            # Initialize panel state
            setattr(self.server.state, f"clip_panel_{s.label}", [])

            # Initialize range sliders with segmentation bounds if available
            if s.actors:
                bounds = s.combined_bounds
                setattr(self.server.state, f"clip_x_{s.label}", [bounds[0], bounds[1]])
                setattr(self.server.state, f"clip_y_{s.label}", [bounds[2], bounds[3]])
                setattr(self.server.state, f"clip_z_{s.label}", [bounds[4], bounds[5]])

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
        rotation_sequence, rotation_angles = self._get_visible_rotation_data()

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
            if self.server.state[f"mpr_segmentation_overlay_{seg.label}"]:
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
        from .window_level import presets

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

        rotation_sequence, rotation_angles = self._get_visible_rotation_data()

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
            if self.server.state[f"mpr_segmentation_overlay_{seg.label}"]:
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
            rotation_sequence, rotation_angles = self._get_visible_rotation_data()
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

    def add_mpr_rotation(self, axis):
        """Add a new rotation to the MPR rotation sequence."""
        import copy

        current_data = copy.deepcopy(
            getattr(self.server.state, "mpr_rotation_data", {"angles_list": []})
        )
        angles_list = current_data["angles_list"]
        new_index = len(angles_list)

        angles_list.append(
            {
                "axis": axis,
                "angle": 0,
                "visible": True,
                "name": "",
                "name_editable": True,
                "deletable": True,
            }
        )

        self.server.state.mpr_rotation_data = current_data

    def remove_mpr_rotation(self, index):
        """Remove a rotation at given index."""
        import copy

        current_data = copy.deepcopy(
            getattr(self.server.state, "mpr_rotation_data", {"angles_list": []})
        )
        angles_list = current_data["angles_list"]

        if 0 <= index < len(angles_list):
            angles_list.pop(index)
            current_data["angles_list"] = angles_list
            self.server.state.mpr_rotation_data = current_data

    def reset_rotation_angle(self, index):
        """Reset the angle of a rotation at given index to zero."""
        import copy

        current_data = copy.deepcopy(
            getattr(self.server.state, "mpr_rotation_data", {"angles_list": []})
        )
        angles_list = current_data["angles_list"]

        if 0 <= index < len(angles_list):
            angles_list[index]["angle"] = 0
            current_data["angles_list"] = angles_list
            self.server.state.mpr_rotation_data = current_data

    def reset_mpr_rotations(self):
        """Reset all MPR rotations."""
        from .rotation import RotationSequence

        # Create fresh RotationSequence and serialize
        new_rotation_seq = RotationSequence()
        self.server.state.mpr_rotation_data = new_rotation_seq.model_dump(mode="json")

        # Update mirror variables
        self.server.state.angle_units = "radians"
        self.server.state.index_order = "itk"

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

    def _on_snap_seg_changed(self, snap_seg_label=None, **kwargs):
        label = snap_seg_label or getattr(self.server.state, "snap_seg_label", "")
        seg = next((s for s in self.scene.segmentations if s.label == label), None)
        if not seg:
            self.server.state.snap_available_labels = []
            return
        frame = getattr(self.server.state, "frame", 0) or 0
        raw_labels = seg.get_labels(frame)
        self.server.state.snap_available_labels = [
            {"title": str(lv), "value": lv} for lv in raw_labels
        ]
        self.server.state.snap_labels_a = []
        self.server.state.snap_labels_b = []
        self.server.state.snap_no_interface = False
        self._invalidate_lock_cache()

    def _snap_selection(self):
        """Current snap selection as (segmentation, mode, labels_a, labels_b).

        Returns None when the selection cannot produce a centroid.
        """
        state = self.server.state
        seg_label = getattr(state, "snap_seg_label", "")
        seg = next((s for s in self.scene.segmentations if s.label == seg_label), None)
        if seg is None:
            return None
        mode = getattr(state, "snap_mode", "label")
        labels_a = list(getattr(state, "snap_labels_a", []))
        labels_b = list(getattr(state, "snap_labels_b", []))
        if not labels_a or (mode == "interface" and not labels_b):
            return None
        return seg, mode, labels_a, labels_b

    def _snap_centroid(self, frame: int) -> list[float] | None:
        """Centroid of the current snap selection at ``frame``."""
        selection = self._snap_selection()
        if selection is None:
            return None
        seg, mode, labels_a, labels_b = selection
        if mode == "label":
            return seg.label_centroid(labels_a, frame)
        return seg.interface_centroid(labels_a, labels_b, frame)

    def _set_mpr_origin(self, center):
        """Write a centroid to mpr_origin, converting out of ITK if needed."""
        self.server.state.mpr_origin = self.convention.point_from_itk(center)

    def snap_to_centroid(self, **kwargs):
        if getattr(self.server.state, "snap_mode", "label") == "reset":
            self.reset_mpr_origin()
            return
        if self._snap_selection() is None:
            return

        frame = getattr(self.server.state, "frame", 0)
        center = self._snap_centroid(frame)
        if center is None:
            self.server.state.snap_no_interface = True
            return

        self.server.state.snap_no_interface = False
        self._set_mpr_origin(center)

    def _locked_centroid(self, frame: int) -> list[float] | None:
        """Memoized per-frame centroid, so locked playback recomputes once."""
        if frame not in self._lock_centroids:
            self._lock_centroids[frame] = self._snap_centroid(frame)
        return self._lock_centroids[frame]

    def apply_frame_lock(self, frame: int):
        """Re-apply the locked position and orientation for ``frame``.

        Position and orientation lock independently; this is a no-op unless at
        least one is enabled.
        """
        state = self.server.state
        mode = getattr(state, "snap_mode", "label")
        if mode == "reset":
            return

        lock_position = getattr(state, "snap_locked", False)
        lock_orientation = (
            getattr(state, "snap_orientation_locked", False) and mode == "interface"
        )
        if not (lock_position or lock_orientation):
            return
        if self._snap_selection() is None:
            return

        missing = False

        if lock_position:
            center = self._locked_centroid(frame)
            if center is None:
                missing = True
            else:
                self._set_mpr_origin(center)

        if lock_orientation:
            plane = self._interface_plane(frame)
            if plane is None:
                missing = True
            else:
                self._apply_alignment(plane)

        state.snap_no_interface = missing

    def _invalidate_lock_cache(self):
        """Drop the memoized per-frame centroids and interface planes."""
        self._lock_centroids: dict[int, list[float] | None] = {}
        self._lock_planes: dict[int, tuple | None] = {}
        self._align_reference = None

    def _on_snap_selection_changed(self, **kwargs):
        """Re-snap when the selection changes while locked."""
        self._invalidate_lock_cache()
        if getattr(self.server.state, "snap_locked", False):
            self.snap_to_centroid()

    def _on_snap_lock_changed(self, snap_locked=None, **kwargs):
        self._invalidate_lock_cache()
        if snap_locked:
            self.snap_to_centroid()

    def _on_snap_orientation_lock_changed(self, snap_orientation_locked=None, **kwargs):
        self._invalidate_lock_cache()
        if snap_orientation_locked:
            self.align_to_interface()

    def _interface_plane(self, frame: int):
        """Memoized dominant plane of the selected interface at ``frame``."""
        if frame not in self._lock_planes:
            selection = self._snap_selection()
            if selection is None:
                self._lock_planes[frame] = None
            else:
                seg, _, labels_a, labels_b = selection
                plane = seg.interface_plane(
                    labels_a, labels_b, frame, anchor=self._align_reference
                )
                # Carry the first plane's basis onto later frames, so the view
                # does not spin as the normal moves.
                if plane is not None and self._align_reference is None:
                    self._align_reference = plane[1]
                self._lock_planes[frame] = plane
        return self._lock_planes[frame]

    def _apply_alignment(self, plane):
        """Replace the alignment step so the views are based on ``plane``.

        The step goes first in the sequence, so any rotations the user has added
        are applied on top of it and keep their meaning relative to the plane: a
        Z rotation spins within it, X and Y tilt out of it.
        """
        import copy

        from .orientation import (
            axcode_transform_matrix,
            rotation_matrix_to_quaternion,
        )

        state = self.server.state
        _, axes, flatness = plane
        state.interface_flatness = flatness

        # The reslice matrix is cumulative @ view_transform, so the rotation that
        # puts the axial view in the interface plane satisfies R @ T_axial = axes.
        target = axes @ axcode_transform_matrix("LPS", "LAS").T
        quaternion = rotation_matrix_to_quaternion(target)

        quaternion = self.convention.quaternion_from_itk(quaternion)

        current_data = copy.deepcopy(
            getattr(state, "mpr_rotation_data", {"angles_list": []})
        )
        angles_list = [
            step
            for step in current_data["angles_list"]
            if step.get("name") != ALIGN_STEP_NAME
        ]
        angles_list.insert(
            0,
            {
                "axis": None,
                "angle": None,
                "quaternion": quaternion,
                "visible": True,
                "name": ALIGN_STEP_NAME,
                "name_editable": False,
                "deletable": True,
            },
        )
        current_data["angles_list"] = angles_list
        state.mpr_rotation_data = current_data

    def align_to_interface(self, **kwargs):
        """Rotate the MPR views into the dominant plane of the selected interface."""
        state = self.server.state
        if getattr(state, "snap_mode", "label") != "interface":
            return
        if self._snap_selection() is None:
            return

        frame = getattr(state, "frame", 0)
        plane = self._interface_plane(frame)
        if plane is None:
            state.snap_no_interface = True
            state.interface_flatness = 0.0
            return

        state.snap_no_interface = False
        self._apply_alignment(plane)
        self._set_mpr_origin(self._snap_centroid(frame) or plane[0])

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
        rotation_sequence, rotation_angles = self._get_visible_rotation_data()

        for seg in self.scene.segmentations:
            if not self.server.state[f"mpr_segmentation_overlay_{seg.label}"]:
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

    def update_segmentation_opacity(self, **kwargs):
        """Update segmentation overlay opacity."""
        if not self.server.state.mpr_enabled:
            return

        current_frame = self.server.state.frame
        opacity = self.server.state.mpr_segmentation_opacity

        for seg in self.scene.segmentations:
            if self.server.state[f"mpr_segmentation_overlay_{seg.label}"]:
                seg.update_mpr_opacity(current_frame, opacity)

        self.server.controller.view_update()

    @asynchronous.task
    async def close_application(self):
        """Close the application by stopping the server."""
        await self.server.stop()
