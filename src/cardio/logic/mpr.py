"""The MPR views: active volume, slice pose, window/level, crosshairs, overlays."""

# System
import math

# Third Party
import numpy as np

# Internal
from ..action import action
from ..reslice import VIEW_TRANSFORMS, ResliceSet
from ..state import ObjectState
from ..view import CameraLock, Layout
from ..window_level import presets
from .base import Controller

ITK_AXIS_NAMES = ("X", "Y", "Z")

# A spoke shorter than this has no reliable direction, so a drag through the
# origin reads as no rotation rather than a wild spin.
MIN_SPOKE_PIXELS = 8.0


def _signed_axis(vector) -> tuple[str, float]:
    """A signed coordinate direction as an ITK axis name and a sign."""
    index = int(np.argmax(np.abs(vector)))
    return ITK_AXIS_NAMES[index], float(np.sign(vector[index]))


def _swept_degrees(centre, start, end) -> float:
    """The angle the cursor sweeps about ``centre``, clockwise positive.

    Zero when either spoke is too short to have a direction worth trusting.
    """
    ax, ay = start[0] - centre[0], start[1] - centre[1]
    bx, by = end[0] - centre[0], end[1] - centre[1]

    if min(math.hypot(ax, ay), math.hypot(bx, by)) < MIN_SPOKE_PIXELS:
        return 0.0

    # Display y points up, so a positive cross product sweeps anticlockwise
    return -math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by))


# The camera lock choices as the picker labels them. Keyed by the enum, so a
# new member without a label fails here rather than silently missing from it.
CAMERA_LOCK_TITLES = {
    CameraLock.FREE: "Free",
    CameraLock.UL: "UL (Axial)",
    CameraLock.LL: "LL (Coronal)",
    CameraLock.LR: "LR (Sagittal)",
}


class MPRController(Controller):
    """Everything that decides what the three MPR views show."""

    seeds = (
        "camera_lock",
        "mpr_origin",
        "mpr_crosshairs_enabled",
        # Already reconciled with the preset by the scene, so these are copies
        # rather than decisions, and the views need not exist to make them.
        "mpr_window",
        "mpr_level",
        "mpr_window_level_preset",
        "mpr_segmentation_opacity",
    )

    def __init__(self, app):
        super().__init__(app)
        self._pending_active_volume = None
        self._missed_volume_change = False

    @property
    def active(self) -> bool:
        """Whether the layout on screen draws the MPR views.

        Every reslice here costs three resampled planes per frame, so the work
        is skipped while the volume rendering or the tile grid is up and the
        views are caught up on the way back.
        """
        return Layout.from_state(
            getattr(self.server.state, "maximized_view", None)
        ).shows_slices

    def register(self):
        state = self.server.state
        state.change("active_volume_label")(self.sync_active_volume)
        state.change("mpr_origin")(self.update_slice_positions)
        state.change("mpr_crosshairs_enabled")(self.sync_crosshairs_visibility)
        state.change("mpr_window", "mpr_level")(self.update_mpr_window_level)
        state.change("mpr_window_level_preset")(self.update_mpr_preset)
        state.change("mpr_rotation_data")(self.update_mpr_rotation)
        state.change("camera_lock")(self._on_camera_lock_change)
        state.change("maximized_view")(self._on_layout_changed)
        state.change("mpr_segmentation_opacity")(self.update_segmentation_opacity)
        for seg in self.scene.segmentations:
            state.change(ObjectState.of(seg).mpr_overlay)(
                self.sync_segmentation_overlays
            )

        self.server.controller.finalize_mpr_initialization = (
            self.finalize_mpr_initialization
        )

    def seed(self):
        """The cuts' pose, their window and level, and the pickers' contents."""
        super().seed()
        state = self.server.state

        state.volume_items = [
            {"text": volume.label, "value": volume.label}
            for volume in self.scene.volumes
        ]
        state.camera_lock_items = [
            {"title": CAMERA_LOCK_TITLES[lock], "value": lock.value}
            for lock in CameraLock
        ]
        state.mpr_presets = [{"text": "Select W/L...", "value": None}] + [
            {"text": preset.name, "value": key} for key, preset in presets.items()
        ]

        for seg in self.scene.segmentations:
            state[ObjectState.of(seg).mpr_overlay] = seg.mpr_overlay

        self._seed_active_volume()

    def _seed_active_volume(self):
        """Choose the volume the cuts are taken from, now or when the UI is up.

        Before the server is ready nothing flushes, and ``state.initial`` moves
        the write through to the client without firing a listener -- so a label
        written now would arrive already applied, and the views would never be
        built for it. It waits for ``finalize_mpr_initialization`` instead.
        Seeding a running server has no such problem and no reason to wait.
        """
        label = (
            self.scene.volumes[0].label
            if self.scene.volumes and not self.scene.active_volume_label
            else self.scene.active_volume_label
        )

        if getattr(self.server.state, "is_ready", False):
            self.server.state.active_volume_label = label
            return

        self.server.state.active_volume_label = ""
        self._pending_active_volume = label

    def update_mpr_frame(self, frame):
        """Update MPR views to show the specified frame."""
        if not self.active:
            return

        active_volume_label = self.server.state.active_volume_label
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        mpr_actors = self._reslices(active_volume, frame)

        views = self.scene.mpr_views
        if views is not None:
            views.show(
                mpr_actors,
                active_volume.crosshair_actors,
                self.server.state.mpr_crosshairs_enabled,
            )

        # Add segmentation overlays
        self._add_segmentation_overlays_to_mpr(frame)

        # Apply current window/level settings to the MPR actors
        window = self.server.state.mpr_window
        level = self.server.state.mpr_level
        active_volume.update_mpr_window_level(frame, window, level)

    def sync_active_volume(self, active_volume_label, **kwargs):
        """Handle active volume selection for MPR."""

        if not active_volume_label:
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
            current_frame = self._frame
            volume_actor = active_volume.actors[current_frame]
            image_data = volume_actor.GetMapper().GetInput()
            center = image_data.GetCenter()

            # Set origin to volume center if it's at default [0,0,0]
            current_origin = self.server.state.mpr_origin
            if current_origin == [0.0, 0.0, 0.0]:
                # VTK reports the centre in ITK; state holds the user's order
                self.server.state.mpr_origin = self.convention.point_from_itk(center)
        except (RuntimeError, IndexError) as e:
            print(f"Error: Cannot get center for volume '{active_volume_label}': {e}")
            return

        # The origin is centred above whatever the layout is, because snapping
        # and the tile grid read it; only the views themselves can wait.
        if not self.active:
            self._missed_volume_change = True
            return

        mpr_actors = self._reslices(active_volume, current_frame)

        # Create crosshair actors
        crosshairs = active_volume.create_crosshair_actors(
            colors=self.scene.mpr_crosshair_colors,
            line_width=self.scene.mpr_crosshair_width,
        )
        crosshairs_visible = self.server.state.mpr_crosshairs_enabled

        # A new volume gets the cameras refit; frame changes deliberately do not
        views = self.scene.mpr_views
        if views is not None:
            views.show(mpr_actors, crosshairs, crosshairs_visible, reset_camera=True)

        # Add segmentation overlays
        self._add_segmentation_overlays_to_mpr(current_frame)

        # Apply current window/level settings to the MPR actors
        window = self.server.state.mpr_window
        level = self.server.state.mpr_level
        active_volume.update_mpr_window_level(current_frame, window, level)

        # Update all views
        self.server.controller.view_update()

    def update_slice_positions(self, **kwargs):
        """Update MPR slice positions when sliders change."""

        if not self.active:
            return

        active_volume_label = self.server.state.active_volume_label
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        # Every cut already built, not just the current frame's, so that
        # stepping between frames lands on the same origin.
        pose = self._current_pose()
        for obj in [active_volume, *self._overlaid_segmentations()]:
            for reslices in obj._mpr_actors.values():
                reslices.set_pose(*pose)

        self.server.controller.view_update()

    def update_mpr_window_level(self, **kwargs):
        """Update MPR window/level when sliders change."""

        if not self.active:
            return

        active_volume_label = self.server.state.active_volume_label
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        # Get current window/level values
        window = self.server.state.mpr_window
        level = self.server.state.mpr_level
        current_frame = self._frame

        self._clear_preset_if_departed(window, level)

        # Update window/level for MPR actors
        active_volume.update_mpr_window_level(current_frame, window, level)

        # Update all views
        self.server.controller.view_update()

    def _clear_preset_if_departed(self, window, level):
        """Drop the preset selection once the values stop matching it.

        Window and level are dragged, but they are also written by the preset
        itself and by seeding, neither of which is a departure from the preset.
        Comparing against what the preset implies tells those apart without
        anyone having to announce which kind of write they are -- which a flag
        could not do anyway, this being a listener that runs at the flush
        rather than at the write.
        """
        preset = presets.get(self.server.state.mpr_window_level_preset)
        if preset is not None and not preset.matches(window, level):
            self.server.state.mpr_window_level_preset = None

    def update_mpr_preset(self, mpr_window_level_preset, **kwargs):
        """Update MPR window/level when preset changes."""
        # Handle None value (Select W/L... option) - do nothing
        if mpr_window_level_preset is None:
            return

        if mpr_window_level_preset in presets:
            preset = presets[mpr_window_level_preset]
            self.server.state.mpr_window = preset.window
            self.server.state.mpr_level = preset.level
            self.update_mpr_window_level()

    def update_mpr_rotation(self, **kwargs):
        """Update MPR views when rotation changes."""

        if self.server.state.rotations_saved_at:
            self.server.state.rotations_stale = True

        if not self.active:
            # A locked volume camera follows the slice pose even while the
            # slices themselves are off screen, so that much still has to run.
            self._sync_vr_camera_to_mpr()
            self.server.controller.view_update()
            return

        active_volume_label = self.server.state.active_volume_label
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        current_frame = self._frame

        for obj in [active_volume, *self._overlaid_segmentations()]:
            self._reslices(obj, current_frame)

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
            rotation = self.app.rotations.rotation_matrix()
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

    def _on_layout_changed(self, **kwargs):
        """Catch the views up on what changed while they were off screen.

        Nothing reposes them while another layout is up, so coming back has to
        redraw from the state as it now stands rather than trust the renderers.
        ``update_mpr_frame`` rebuilds all three without refitting the cameras,
        which would throw away the pan and zoom the views were left at; a
        volume changed in the meantime does want that refit, and says so.
        """
        if not self.active:
            return

        if self._missed_volume_change:
            self._missed_volume_change = False
            self.sync_active_volume(self.server.state.active_volume_label)
            return

        self.update_mpr_frame(self._frame)
        self.server.controller.view_update()

    def _on_camera_lock_change(self, camera_lock, **kwargs):
        self._sync_vr_camera_to_mpr()
        self.server.controller.view_update()

    def sync_crosshairs_visibility(self, **kwargs):
        """Toggle crosshair visibility on all MPR views."""
        if not self.active:
            return

        active_volume_label = self.server.state.active_volume_label
        if not active_volume_label:
            return

        active_volume = self._active_volume()

        if not active_volume:
            return

        visible = self.server.state.mpr_crosshairs_enabled
        active_volume.set_crosshairs_visible(visible)
        self.server.controller.view_update()

    @action("reset_mpr_origin")
    def reset_mpr_origin(self):
        active_volume_label = self.server.state.active_volume_label
        active_volume = self._active_volume()
        if not active_volume:
            return
        current_frame = self._frame
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

        # The moment the views are real, which is when a configured camera pose
        # finally has something to point.
        self.app.camera.install_configured()

    def _current_pose(self):
        """The origin and rotation the cuts are aimed by, both in ITK."""
        return (
            self.convention.point_to_itk(self.server.state.mpr_origin),
            self.app.rotations.rotation_matrix(),
        )

    def _reslices(self, obj, frame: int) -> ResliceSet:
        """``obj``'s cuts for ``frame``, aimed where the views are looking.

        Building and aiming are one step, so the class has no way to hold a
        set nothing has posed: a fresh pipeline is centred on its own image and
        unrotated, which is what it would draw.
        """
        reslices = obj.get_mpr_actors_for_frame(frame)
        reslices.set_pose(*self._current_pose())
        return reslices

    def _add_segmentation_overlays_to_mpr(self, frame: int):
        """Pose and add the enabled segmentation overlays on top of the views."""
        views = self.scene.mpr_views
        opacity = self.server.state.mpr_segmentation_opacity

        for seg in self._overlaid_segmentations():
            overlay = self._reslices(seg, frame)
            seg.update_mpr_opacity(frame, opacity)

            if views is not None:
                views.add_overlay(overlay)

    def sync_segmentation_overlays(self, **kwargs):
        """Toggle segmentation overlay visibility on MPR views."""
        if not self.active:
            return

        views = self.scene.mpr_views
        if views is None:
            return

        current_frame = self.server.state.frame
        active_volume = self._active_volume()

        views.clear()
        if active_volume:
            views.set_image(self._reslices(active_volume, current_frame))
            views.add_crosshairs(
                active_volume.crosshair_actors,
                self.server.state.mpr_crosshairs_enabled,
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

        rotation = self.app.rotations.rotation_matrix()

        return np.array(
            self.convention.point_from_itk(rotation @ base_normals[view_name])
        )

    @action("scroll_slice")
    def scroll_slice(self, view_name: str, distance: float):
        """Travel ``distance`` out of the plane, or along the traverse path.

        Traverse mode names a line through the volume, and travelling it is
        what scrolling there is for; every other mode leaves the slice along
        its own normal.
        """
        if view_name not in ("axial", "sagittal", "coronal"):
            return

        # Traverse mode has a line to travel, and it is the more useful one:
        # scrolling scrubs the path rather than leaving it through the slice.
        if self.app.snap.travel(distance):
            return

        if self.app.snap.position_locked:
            return

        origin = self.server.state.mpr_origin
        step = self.scroll_vector(view_name)
        self.server.state.mpr_origin = [
            origin[i] + distance * step[i] for i in range(3)
        ]

    @action("rotate_view")
    def rotate_view(self, view_name: str, start: list[float], end: list[float]):
        """Spin the slice frame by the angle the cursor sweeps about the origin.

        The origin sits at the centre of every reslice and the camera looks
        straight at it, so the line from it to the cursor is a spoke: the angle
        between the spoke where the drag was and the spoke where it is now is
        the angle to turn. The grab is direct, with no sensitivity to tune --
        the image keeps up with the cursor at any radius.

        The turn is a roll about the view's own normal, the axis
        ``scroll_slice`` travels along, so the axial view turns about the
        craniocaudal axis and goes on showing the same cut. That normal is a
        signed L/A/S direction before any rotation is applied, so the drag
        lands as a plain Euler step about X, Y or Z rather than a quaternion.

        ``start`` and ``end`` are display positions, as the view events carry.
        """
        if view_name not in VIEW_TRANSFORMS or self.app.snap.orientation_locked:
            return

        views = self.scene.mpr_views
        if views is None:
            return

        centre = views.origin_on_screen(view_name)
        if centre is None:
            return

        degrees = _swept_degrees(centre, start, end)
        if not degrees:
            return

        frame = VIEW_TRANSFORMS[view_name]

        # Which way a turn reads on screen flips with the frame's handedness,
        # and the axcode frames do not agree on it: coronal's is the odd one,
        # the same disagreement scroll_vector settles by taking P rather than
        # the axcode's A as its normal.
        hand = float(np.linalg.det(frame))

        axis, sign = _signed_axis(frame[:, 2])
        self.app.rotations.turn_mouse(axis, sign * hand * degrees)

    @action("zoom_views")
    def zoom_views(self, factor: float):
        """Zoom all three MPR views by ``factor``."""
        views = self.scene.mpr_views
        if views is None:
            return

        views.zoom(factor)
        self.server.controller.view_update()

    def pan_vectors(self, view_name: str):
        """A view's in-plane right and up directions, in the user's index order.

        Columns 0 and 1 of the reslice frame, which is the basis the view is
        actually cut in; column 2 is the normal ``scroll_vector`` walks along.
        Composed in ITK and handed back in the convention ``mpr_origin`` is
        stored in, exactly as the normal is.
        """
        frame = self.app.rotations.rotation_matrix() @ VIEW_TRANSFORMS[view_name]
        return tuple(
            np.array(self.convention.point_from_itk(frame[:, axis])) for axis in (0, 1)
        )

    @action("pan_view")
    def pan_view(self, view_name: str, dx: float, dy: float):
        """Slide the shared origin within ``view_name``'s own plane.

        The in-plane sibling of ``scroll_slice``. The dragged view translates
        under its crosshair, which stays put because the origin is always what
        sits at the centre of a reslice; the other two views re-cut, one view's
        in-plane axes being the axes the others scroll along.
        """
        if self.app.snap.position_locked:
            return

        views = self.scene.mpr_views
        if views is None or view_name not in VIEW_TRANSFORMS:
            return

        scale = views.world_per_pixel(view_name)
        if not scale:
            return

        right, up = self.pan_vectors(view_name)
        # Against the drag, so the image travels with the cursor
        step = -(dx * right + dy * up) * scale

        origin = self.server.state.mpr_origin
        self.server.state.mpr_origin = [origin[i] + step[i] for i in range(3)]

    @action("adjust_window_level")
    def adjust_window_level(self, window_delta: float, level_delta: float):
        """Nudge the MPR window and level, keeping the window positive."""
        state = self.server.state
        state.mpr_window = max(1.0, state.mpr_window + window_delta)
        state.mpr_level = state.mpr_level + level_delta

    @action("set_window_level_preset")
    def set_window_level_preset(self, preset: int):
        """Choose one of the named window/level pairs."""
        self.server.state.mpr_window_level_preset = preset

    @action("toggle_crosshairs")
    def toggle_crosshairs(self):
        """Show or hide the lines marking where the other two cuts fall."""
        state = self.server.state
        state.mpr_crosshairs_enabled = not state.mpr_crosshairs_enabled

    def update_segmentation_opacity(self, **kwargs):
        """Update segmentation overlay opacity."""
        if not self.active:
            return

        current_frame = self.server.state.frame
        opacity = self.server.state.mpr_segmentation_opacity

        for seg in self._overlaid_segmentations():
            seg.update_mpr_opacity(current_frame, opacity)

        self.server.controller.view_update()
