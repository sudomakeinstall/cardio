import asyncio
import datetime as dt

from trame.app import asynchronous

from .scene import Scene
from .screenshot import Screenshot


class Logic:
    def __init__(self, server, scene: Scene):
        self.server = server
        self.scene = scene

        # Initialize mpr_presets early to avoid undefined errors
        self.server.state.mpr_presets = []

        # Initialize volume items for MPR dropdown
        self.server.state.volume_items = [
            {"text": volume.label, "value": volume.label}
            for volume in self.scene.volumes
        ]

        self.server.state.change("frame")(self.update_frame)
        self.server.state.change("playing")(self.play)
        self.server.state.change("theme_mode")(self.sync_background_color)
        self.server.state.change("mpr_enabled")(self.sync_mpr_mode)
        self.server.state.change("active_volume_label")(self.sync_active_volume)
        self.server.state.change("axial_slice", "sagittal_slice", "coronal_slice")(
            self.update_slice_positions
        )
        self.server.state.change("mpr_window", "mpr_level")(
            self.update_mpr_window_level
        )
        self.server.state.change("mpr_window_level_preset")(self.update_mpr_preset)
        self.server.state.change("mpr_rotation_sequence")(self.update_mpr_rotation)

        # Add handlers for individual rotation angles and visibility
        for i in range(self.scene.max_mpr_rotations):
            self.server.state.change(f"mpr_rotation_angle_{i}")(
                self.update_mpr_rotation
            )
            self.server.state.change(f"mpr_rotation_visible_{i}")(
                self.update_mpr_rotation
            )

        # Initialize visibility state variables
        for m in self.scene.meshes:
            self.server.state[f"mesh_visibility_{m.label}"] = m.visible
        for v in self.scene.volumes:
            self.server.state[f"volume_visibility_{v.label}"] = v.visible
        for s in self.scene.segmentations:
            self.server.state[f"segmentation_visibility_{s.label}"] = s.visible

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
        self.server.controller.reset_all = self.reset_all
        self.server.controller.close_application = self.close_application

        # MPR rotation controllers
        self.server.controller.add_x_rotation = lambda: self.add_mpr_rotation("X")
        self.server.controller.add_y_rotation = lambda: self.add_mpr_rotation("Y")
        self.server.controller.add_z_rotation = lambda: self.add_mpr_rotation("Z")
        self.server.controller.remove_rotation_event = self.remove_mpr_rotation
        self.server.controller.reset_rotations = self.reset_mpr_rotations

        # Initialize MPR state
        self.server.state.mpr_enabled = self.scene.mpr_enabled
        self.server.state.active_volume_label = self.scene.active_volume_label
        self.server.state.axial_slice = self.scene.axial_slice
        self.server.state.sagittal_slice = self.scene.sagittal_slice
        self.server.state.coronal_slice = self.scene.coronal_slice
        self.server.state.mpr_window = self.scene.mpr_window
        self.server.state.mpr_level = self.scene.mpr_level
        self.server.state.mpr_window_level_preset = self.scene.mpr_window_level_preset
        self.server.state.mpr_rotation_sequence = self.scene.mpr_rotation_sequence

        # Initialize MPR presets data
        try:
            from .window_level import presets

            self.server.state.mpr_presets = [
                {"text": preset.name, "value": key} for key, preset in presets.items()
            ]
        except Exception as e:
            print(f"Error initializing MPR presets: {e}")
            self.server.state.mpr_presets = []

        # Initialize rotation angle states
        for i in range(self.scene.max_mpr_rotations):
            setattr(self.server.state, f"mpr_rotation_angle_{i}", 0)
            setattr(self.server.state, f"mpr_rotation_axis_{i}", f"Rotation {i + 1}")
            setattr(self.server.state, f"mpr_rotation_visible_{i}", True)

        # Apply initial preset to ensure window/level values are set correctly
        # Only update state values, don't call update methods yet since MPR may not be enabled
        from .window_level import presets

        if self.scene.mpr_window_level_preset in presets:
            preset = presets[self.scene.mpr_window_level_preset]
            self.server.state.mpr_window = preset.window
            self.server.state.mpr_level = preset.level

        # Initialize clipping state variables
        self._initialize_clipping_state()

    def update_frame(self, frame, **kwargs):
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

        # Find the active volume
        active_volume = None
        for volume in self.scene.volumes:
            if volume.label == active_volume_label:
                active_volume = volume
                break

        if not active_volume:
            return

        # Get or create MPR actors for the new frame
        mpr_actors = active_volume.get_mpr_actors_for_frame(frame)

        # Update each MPR renderer with the new frame's actors
        if self.scene.axial_renderWindow:
            axial_renderer = (
                self.scene.axial_renderWindow.GetRenderers().GetFirstRenderer()
            )
            if axial_renderer:
                axial_renderer.RemoveAllViewProps()
                axial_renderer.AddActor(mpr_actors["axial"]["actor"])
                mpr_actors["axial"]["actor"].SetVisibility(True)
                # Apply current slice position and window/level
                self._apply_current_mpr_settings(active_volume, frame)
                axial_renderer.ResetCamera()

        if self.scene.coronal_renderWindow:
            coronal_renderer = (
                self.scene.coronal_renderWindow.GetRenderers().GetFirstRenderer()
            )
            if coronal_renderer:
                coronal_renderer.RemoveAllViewProps()
                coronal_renderer.AddActor(mpr_actors["coronal"]["actor"])
                mpr_actors["coronal"]["actor"].SetVisibility(True)
                coronal_renderer.ResetCamera()

        if self.scene.sagittal_renderWindow:
            sagittal_renderer = (
                self.scene.sagittal_renderWindow.GetRenderers().GetFirstRenderer()
            )
            if sagittal_renderer:
                sagittal_renderer.RemoveAllViewProps()
                sagittal_renderer.AddActor(mpr_actors["sagittal"]["actor"])
                mpr_actors["sagittal"]["actor"].SetVisibility(True)
                sagittal_renderer.ResetCamera()

    def _apply_current_mpr_settings(self, active_volume, frame):
        """Apply current slice positions and window/level to MPR actors."""
        # Apply slice positions
        axial_slice = getattr(self.server.state, "axial_slice", 0.5)
        sagittal_slice = getattr(self.server.state, "sagittal_slice", 0.5)
        coronal_slice = getattr(self.server.state, "coronal_slice", 0.5)

        # Get rotation data
        rotation_sequence = getattr(self.server.state, "mpr_rotation_sequence", [])
        rotation_angles = {}
        for i in range(len(rotation_sequence)):
            rotation_angles[i] = getattr(
                self.server.state, f"mpr_rotation_angle_{i}", 0
            )

        active_volume.update_slice_positions(
            frame,
            axial_slice,
            sagittal_slice,
            coronal_slice,
            rotation_sequence,
            rotation_angles,
        )

        # Apply window/level
        window = getattr(self.server.state, "mpr_window", 400.0)
        level = getattr(self.server.state, "mpr_level", 40.0)
        active_volume.update_mpr_window_level(frame, window, level)

    @asynchronous.task
    async def play(self, playing, **kwargs):
        if not (self.server.state.incrementing or self.server.state.rotating):
            self.server.state.playing = False
        while self.server.state.playing:
            with self.server.state as state:
                if state.incrementing:
                    state.frame = (state.frame + 1) % self.scene.nframes
                if state.rotating:
                    deg = 360 / (self.scene.nframes * state.bpr)
                    self.scene.renderer.GetActiveCamera().Azimuth(deg)
                self.server.controller.view_update()
            await asyncio.sleep(1 / state.bpm * 60 / self.scene.nframes)

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
        dr = dt.datetime.now().strftime(self.scene.screenshot_subdirectory_format)
        dr = self.scene.screenshot_directory / dr
        dr.mkdir(parents=True, exist_ok=True)

        if not (self.server.state.incrementing or self.server.state.rotating):
            ss = Screenshot(self.scene.renderWindow)
            ss.save(f"{dr}/0.png")
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
                    ss = Screenshot(self.scene.renderWindow)
                    ss.save(f"{dr}/{i}.png")
                    await asyncio.sleep(
                        1 / self.server.state.bpm * 60 / self.scene.nframes
                    )

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

    def sync_mpr_mode(self, mpr_enabled, **kwargs):
        """Handle MPR mode toggle."""
        if (
            mpr_enabled
            and self.scene.volumes
            and not self.server.state.active_volume_label
        ):
            # Auto-select first volume when MPR is enabled and no volume is selected
            self.server.state.active_volume_label = self.scene.volumes[0].label

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

        # Create MPR actors for current frame
        current_frame = getattr(self.server.state, "frame", 0)
        mpr_actors = active_volume.get_mpr_actors_for_frame(current_frame)

        # Add MPR actors to their respective renderers
        if self.scene.axial_renderWindow:
            axial_renderer = (
                self.scene.axial_renderWindow.GetRenderers().GetFirstRenderer()
            )
            if axial_renderer:
                axial_renderer.RemoveAllViewProps()  # Clear existing actors
                axial_renderer.AddActor(mpr_actors["axial"]["actor"])
                mpr_actors["axial"]["actor"].SetVisibility(True)
                axial_renderer.ResetCamera()

        if self.scene.coronal_renderWindow:
            coronal_renderer = (
                self.scene.coronal_renderWindow.GetRenderers().GetFirstRenderer()
            )
            if coronal_renderer:
                coronal_renderer.RemoveAllViewProps()
                coronal_renderer.AddActor(mpr_actors["coronal"]["actor"])
                mpr_actors["coronal"]["actor"].SetVisibility(True)
                coronal_renderer.ResetCamera()

        if self.scene.sagittal_renderWindow:
            sagittal_renderer = (
                self.scene.sagittal_renderWindow.GetRenderers().GetFirstRenderer()
            )
            if sagittal_renderer:
                sagittal_renderer.RemoveAllViewProps()
                sagittal_renderer.AddActor(mpr_actors["sagittal"]["actor"])
                mpr_actors["sagittal"]["actor"].SetVisibility(True)
                sagittal_renderer.ResetCamera()

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

        # Find the active volume
        active_volume = None
        for volume in self.scene.volumes:
            if volume.label == active_volume_label:
                active_volume = volume
                break

        if not active_volume:
            return

        # Get current slice positions
        axial_slice = getattr(self.server.state, "axial_slice", 0.5)
        sagittal_slice = getattr(self.server.state, "sagittal_slice", 0.5)
        coronal_slice = getattr(self.server.state, "coronal_slice", 0.5)
        current_frame = getattr(self.server.state, "frame", 0)

        # Get rotation data
        rotation_sequence = getattr(self.server.state, "mpr_rotation_sequence", [])
        rotation_angles = {}
        for i in range(len(rotation_sequence)):
            rotation_angles[i] = getattr(
                self.server.state, f"mpr_rotation_angle_{i}", 0
            )

        # Update slice positions with rotation
        active_volume.update_slice_positions(
            current_frame,
            axial_slice,
            sagittal_slice,
            coronal_slice,
            rotation_sequence,
            rotation_angles,
        )

        # Update all views
        self.server.controller.view_update()

    def update_mpr_window_level(self, **kwargs):
        """Update MPR window/level when sliders change."""

        if not getattr(self.server.state, "mpr_enabled", False):
            return

        active_volume_label = getattr(self.server.state, "active_volume_label", "")
        if not active_volume_label:
            return

        # Find the active volume
        active_volume = None
        for volume in self.scene.volumes:
            if volume.label == active_volume_label:
                active_volume = volume
                break

        if not active_volume:
            return

        # Get current window/level values
        window = getattr(self.server.state, "mpr_window", 400.0)
        level = getattr(self.server.state, "mpr_level", 40.0)
        current_frame = getattr(self.server.state, "frame", 0)

        # Update window/level for MPR actors
        active_volume.update_mpr_window_level(current_frame, window, level)

        # Update all views
        self.server.controller.view_update()

    def update_mpr_preset(self, mpr_window_level_preset, **kwargs):
        """Update MPR window/level when preset changes."""
        from .window_level import presets

        if mpr_window_level_preset in presets:
            preset = presets[mpr_window_level_preset]
            self.server.state.mpr_window = preset.window
            self.server.state.mpr_level = preset.level

            # Update the actual MPR views with new window/level
            self.update_mpr_window_level()

    def update_mpr_rotation(self, **kwargs):
        """Update MPR views when rotation changes."""
        if not getattr(self.server.state, "mpr_enabled", False):
            return

        active_volume_label = getattr(self.server.state, "active_volume_label", "")
        if not active_volume_label:
            return

        # Find the active volume
        active_volume = None
        for volume in self.scene.volumes:
            if volume.label == active_volume_label:
                active_volume = volume
                break

        if not active_volume:
            return

        # Get current slice positions
        axial_slice = getattr(self.server.state, "axial_slice", 0.5)
        sagittal_slice = getattr(self.server.state, "sagittal_slice", 0.5)
        coronal_slice = getattr(self.server.state, "coronal_slice", 0.5)
        current_frame = getattr(self.server.state, "frame", 0)

        # Get rotation data - include all visible rotations
        rotation_sequence = getattr(self.server.state, "mpr_rotation_sequence", [])
        rotation_angles = {}

        # Include all visible rotations regardless of position
        for i in range(len(rotation_sequence)):
            is_visible = getattr(self.server.state, f"mpr_rotation_visible_{i}", True)
            if is_visible:
                rotation_angles[i] = getattr(
                    self.server.state, f"mpr_rotation_angle_{i}", 0
                )

        # Update slice positions with rotation
        active_volume.update_slice_positions(
            current_frame,
            axial_slice,
            sagittal_slice,
            coronal_slice,
            rotation_sequence,
            rotation_angles,
        )

        # Update all views
        self.server.controller.view_update()

    def add_mpr_rotation(self, axis):
        """Add a new rotation to the MPR rotation sequence."""
        import copy

        current_sequence = copy.deepcopy(
            getattr(self.server.state, "mpr_rotation_sequence", [])
        )
        current_sequence.append({"axis": axis, "angle": 0})
        self.server.state.mpr_rotation_sequence = current_sequence
        self.update_mpr_rotation_labels()

    def remove_mpr_rotation(self, index):
        """Remove a rotation at given index and all subsequent rotations."""
        sequence = list(getattr(self.server.state, "mpr_rotation_sequence", []))
        if 0 <= index < len(sequence):
            sequence = sequence[:index]
            self.server.state.mpr_rotation_sequence = sequence

            # Reset angle states for all removed rotations
            for i in range(index, self.scene.max_mpr_rotations):
                setattr(self.server.state, f"mpr_rotation_angle_{i}", 0)

            self.update_mpr_rotation_labels()

    def reset_mpr_rotations(self):
        """Reset all MPR rotations."""
        self.server.state.mpr_rotation_sequence = []
        for i in range(self.scene.max_mpr_rotations):
            setattr(self.server.state, f"mpr_rotation_angle_{i}", 0)
            setattr(self.server.state, f"mpr_rotation_visible_{i}", True)
        self.update_mpr_rotation_labels()

    def update_mpr_rotation_labels(self):
        """Update the rotation axis labels for display."""
        rotation_sequence = getattr(self.server.state, "mpr_rotation_sequence", [])
        for i, rotation in enumerate(rotation_sequence):
            setattr(
                self.server.state,
                f"mpr_rotation_axis_{i}",
                f"{rotation['axis']} ({i + 1})",
            )
        # Clear unused labels
        for i in range(len(rotation_sequence), self.scene.max_mpr_rotations):
            setattr(self.server.state, f"mpr_rotation_axis_{i}", f"Rotation {i + 1}")

    @asynchronous.task
    async def close_application(self):
        """Close the application by stopping the server."""
        await self.server.stop()
