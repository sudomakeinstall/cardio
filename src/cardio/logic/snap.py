"""Snapping the MPR origin and orientation to a segmentation feature."""

# System
import logging

# Third Party
import numpy as np

# Internal
from ..orientation import (
    axcode_transform_matrix,
    rotation_matrix_to_quaternion,
)
from ..rotation import RotationStep
from ..segmentation import interpolate_planes
from ..snap import PLANAR_MODES, SnapMode
from .base import Controller

ALIGN_STEP_NAME = "Interface plane"

# The two interfaces traverse mode travels between: A|B, then B|C.
INTERFACE_AB = 0
INTERFACE_BC = 1


def _without_alignment(steps):
    """``steps`` with the alignment step removed."""
    return [step for step in steps if step.name != ALIGN_STEP_NAME]


def alignment_rotation(axes: np.ndarray) -> np.ndarray:
    """The rotation that puts the axial view in the plane ``axes`` describes.

    The reslice matrix is cumulative @ view_transform, so the rotation the
    views need satisfies R @ T_axial = axes. In ITK, as all the rotation math
    is.
    """
    return axes @ axcode_transform_matrix("LPS", "LAS").T


class SnapController(Controller):
    """Centroid snapping, interface alignment and the per-frame locks."""

    def __init__(self, app):
        super().__init__(app)
        self._published_seg_label = None
        self._invalidate_lock_cache()

    def register(self):
        if not self.scene.segmentations:
            return

        state = self.server.state
        state.change("snap_seg_label")(self._on_snap_seg_changed)
        state.change("snap_mode", "snap_labels_a", "snap_labels_b", "snap_labels_c")(
            self._on_snap_selection_changed
        )
        state.change("snap_traverse")(self._on_snap_traverse_changed)
        state.change("snap_locked")(self._on_snap_lock_changed)
        state.change("snap_orientation_locked")(self._on_snap_orientation_lock_changed)

        self._publish_configured_selection()
        state.snap_available_labels = []
        state.snap_seg_items = [
            {"title": s.label, "value": s.label} for s in self.scene.segmentations
        ]
        state.snap_no_interface = False
        state.snap_locked = False
        state.interface_flatness = 0.0
        state.snap_orientation_locked = False

        self.server.controller.snap_to_centroid = self.snap_to_centroid
        self.server.controller.align_to_interface = self.align_to_interface
        self.server.controller.swap_snap_groups = self.swap_groups
        self.server.controller.reset_snap = self.reset

    def register_initial_labels(self):
        """Populate the pickers and apply the configured selection.

        Called once every controller has registered and the MPR view exists,
        which is what the configured groups have to be validated against and
        what a configured lock has to snap against.
        """
        if not self.scene.segmentations:
            return
        seg = self._selected_segmentation()
        if seg is None:
            return
        self._publish_available_labels(seg)
        self._publish_configured_groups(seg)
        self._apply_configured_locks()

    def _apply_configured_locks(self):
        """Turn on the locks the config asks for, and act on them now.

        The change listeners would do the same work on the next flush, but a
        lock has to hold from the first frame rather than the first flush, and
        the tests drive the controllers without a flush at all. Doing it twice
        costs nothing: both answers come from the memoized caches.
        """
        snap = self.scene.snap
        state = self.server.state
        state.snap_locked = snap.locked
        state.snap_orientation_locked = snap.orientation_locked
        if snap.locked:
            self.snap_to_centroid()
        if snap.orientation_locked:
            self.align_to_interface()

    @property
    def _frame(self) -> int:
        """The frame being shown.

        ``frame`` reaches state only when the playback slider is built, which
        happens after Logic; until then the scene's configured frame is the one
        that will be shown.
        """
        frame = getattr(self.server.state, "frame", None)
        return self.scene.current_frame if frame is None else frame

    def _selected_segmentation(self, label: str | None = None):
        """The segmentation ``label`` names, defaulting to the selected one."""
        if label is None:
            label = getattr(self.server.state, "snap_seg_label", "")
        return next((s for s in self.scene.segmentations if s.label == label), None)

    def _publish_configured_selection(self):
        """Write the configured mode, segmentation and groups to state.

        The groups are written unfiltered here: the pickers are not populated
        until ``register_initial_labels``, which filters them against the
        labels the segmentation actually holds.
        """
        state = self.server.state
        snap = self.scene.snap
        state.snap_mode = snap.mode.value
        state.snap_seg_label = (
            snap.segmentation_label or self.scene.segmentations[0].label
        )
        state.snap_labels_a = list(snap.labels_a)
        state.snap_labels_b = list(snap.labels_b)
        state.snap_labels_c = list(snap.labels_c)
        state.snap_traverse = snap.traverse

    def _publish_available_labels(self, seg) -> bool:
        """Publish the label picker options for ``seg`` at the current frame.

        Returns whether this is a different segmentation from the one the
        pickers were last built for.
        """
        frame = self._frame
        self.server.state.snap_available_labels = [
            {"title": str(lv), "value": lv} for lv in seg.get_labels(frame)
        ]
        changed = seg.label != self._published_seg_label
        self._published_seg_label = seg.label
        return changed

    def _publish_configured_groups(self, seg):
        """Apply the configured groups to state, dropping labels ``seg`` lacks.

        A typo leaves the mode short of a group, which the panel already shows
        as a selection that cannot snap; that is friendlier than refusing to
        launch over one bad value.
        """
        state = self.server.state
        frame = self._frame
        present = set(seg.get_labels(frame))
        snap = self.scene.snap

        missing = sorted({lv for group in snap.groups for lv in group} - present)
        if missing:
            logging.warning(
                f"Snap labels not present in '{seg.label}' at frame {frame}: {missing}"
            )

        state.snap_labels_a = [lv for lv in snap.labels_a if lv in present]
        state.snap_labels_b = [lv for lv in snap.labels_b if lv in present]
        state.snap_labels_c = [lv for lv in snap.labels_c if lv in present]
        state.snap_no_interface = False
        self._invalidate_lock_cache()

    def _on_snap_seg_changed(self, snap_seg_label=None, **kwargs):
        """Repopulate the pickers, and clear groups that no longer apply.

        The groups index into the segmentation being left, so they cannot
        survive a change of segmentation. The listener also fires on the first
        flush, for the value the configured selection wrote -- which is the
        segmentation the pickers were just built for, so nothing is cleared.
        """
        seg = self._selected_segmentation(snap_seg_label or None)
        if seg is None:
            self.server.state.snap_available_labels = []
            return
        if not self._publish_available_labels(seg):
            return
        self.server.state.snap_labels_a = []
        self.server.state.snap_labels_b = []
        self.server.state.snap_labels_c = []
        self.server.state.snap_no_interface = False
        self._invalidate_lock_cache()

    def _snap_selection(self):
        """Current selection as (segmentation, mode, labels_a, labels_b, labels_c).

        Returns None when the selection cannot produce a centroid: every mode
        needs group A, the interface modes need B, and traverse needs C as well.
        """
        state = self.server.state
        seg_label = getattr(state, "snap_seg_label", "")
        seg = next((s for s in self.scene.segmentations if s.label == seg_label), None)
        if seg is None:
            return None
        mode = getattr(state, "snap_mode", SnapMode.LABEL)
        labels_a = list(getattr(state, "snap_labels_a", []))
        labels_b = list(getattr(state, "snap_labels_b", []))
        labels_c = list(getattr(state, "snap_labels_c", []))
        if not labels_a:
            return None
        if mode in PLANAR_MODES and not labels_b:
            return None
        if mode == SnapMode.TRAVERSE and not labels_c:
            return None
        return seg, mode, labels_a, labels_b, labels_c

    def _traverse_fraction(self) -> float:
        """The traverse slider as a fraction of the way from A|B to B|C."""
        return getattr(self.server.state, "snap_traverse", 0) / 100.0

    def _interface_labels(self, selection, interface: int):
        """The label groups on either side of ``interface`` for ``selection``."""
        _, _, labels_a, labels_b, labels_c = selection
        if interface == INTERFACE_BC:
            return labels_b, labels_c
        return labels_a, labels_b

    def _endpoint_centroid(self, frame: int, interface: int) -> list[float] | None:
        """Memoized centroid of one snap endpoint at ``frame``.

        In label mode that is the centroid of group A; otherwise the centroid of
        the named interface. Keyed on the interface rather than the mode's final
        answer, so the memo survives the traverse slider moving.
        """
        key = (frame, interface)
        if key not in self._lock_centroids:
            selection = self._snap_selection()
            if selection is None:
                self._lock_centroids[key] = None
            else:
                seg, mode, labels_a, _, _ = selection
                if mode == SnapMode.LABEL:
                    self._lock_centroids[key] = seg.label_centroid(labels_a, frame)
                else:
                    left, right = self._interface_labels(selection, interface)
                    self._lock_centroids[key] = seg.interface_centroid(
                        left, right, frame
                    )
        return self._lock_centroids[key]

    def _snap_centroid(
        self, frame: int, fraction: float | None = None
    ) -> list[float] | None:
        """Centroid the current snap selection asks for at ``frame``.

        In traverse mode the two interface centroids are blended by ``fraction``,
        which defaults to the slider. That is the same straight line
        ``interpolate_planes`` walks -- but taken over the centres of mass the
        other modes use, rather than the plane fit's centroid.
        """
        selection = self._snap_selection()
        if selection is None:
            return None
        if selection[1] != SnapMode.TRAVERSE:
            return self._endpoint_centroid(frame, INTERFACE_AB)

        start = self._endpoint_centroid(frame, INTERFACE_AB)
        end = self._endpoint_centroid(frame, INTERFACE_BC)
        if start is None or end is None:
            return None
        if fraction is None:
            fraction = self._traverse_fraction()
        fraction = min(1.0, max(0.0, float(fraction)))
        blend = (1.0 - fraction) * np.asarray(start) + fraction * np.asarray(end)
        return [float(v) for v in blend]

    def snap_to_centroid(self, **kwargs):
        if self._snap_selection() is None:
            return

        frame = self._frame
        center = self._snap_centroid(frame)
        if center is None:
            self.server.state.snap_no_interface = True
            return

        self.server.state.snap_no_interface = False
        self.app.mpr.set_origin(center)

    def reset(self, **kwargs):
        """Put the panel back the way the config asks for, and undo what it did.

        Reset returns to the configured selection rather than to a blank panel,
        so the state it lands in is the state the app launches in.

        The locks are released first: rebuilding the groups while one is still
        on would send the views chasing a selection that is being taken apart.
        They go back on last, once there is a settled scene to snap against.

        Only the alignment step is dropped from the rotation sequence. Steps the
        user added are theirs, and the rotations panel has its own button for
        deleting those.
        """
        if not self.scene.segmentations:
            return

        state = self.server.state
        state.snap_locked = False
        state.snap_orientation_locked = False

        self._publish_configured_selection()
        state.snap_no_interface = False
        state.interface_flatness = 0.0

        self._invalidate_lock_cache()
        self.app.rotations.edit_steps(_without_alignment)
        self.app.mpr.reset_mpr_origin()

        seg = self._selected_segmentation()
        if seg is not None:
            self._publish_available_labels(seg)
            self._publish_configured_groups(seg)
            self._apply_configured_locks()

    def travel(self, steps: float) -> bool:
        """Move along the traverse path by ``steps`` percent of its length.

        The slider is the path, so travelling it is scrubbing the slider: the
        mode already turns a fraction into an origin and an orientation, and a
        lock derives its answer from the same fraction, so this drives the mode
        rather than fighting it.

        Returns whether there was a path to travel at all, so a caller can fall
        back to what scrolling otherwise means. Travelling into either end is
        still travelling -- it holds there rather than reverting to moving the
        origin through the slice.
        """
        if getattr(self.server.state, "snap_mode", SnapMode.LABEL) != SnapMode.TRAVERSE:
            return False
        if self._snap_selection() is None:
            return False

        # A trackpad sends fractions of a step; without carrying the remainder
        # a slow scroll would round to nothing every time and never move.
        total = self._travel_remainder + steps
        whole = int(total)
        self._travel_remainder = total - whole

        current = getattr(self.server.state, "snap_traverse", 0)
        moved = min(100, max(0, current + whole))
        if moved != current:
            self.server.state.snap_traverse = moved
        return True

    @property
    def position_locked(self) -> bool:
        """Whether snap is the standing owner of ``mpr_origin``.

        A lock re-applies the centroid on every frame, so a gesture that moves
        the origin would be overwritten the next time the frame changed.
        """
        return bool(getattr(self.server.state, "snap_locked", False))

    @property
    def orientation_locked(self) -> bool:
        """Whether snap is the standing owner of the alignment step.

        Only the planar modes fit a plane, so only they can hold an
        orientation to begin with.
        """
        state = self.server.state
        return (
            bool(getattr(state, "snap_orientation_locked", False))
            and getattr(state, "snap_mode", SnapMode.LABEL) in PLANAR_MODES
        )

    def apply_frame_lock(self, frame: int):
        """Re-apply the locked position and orientation for ``frame``.

        Position and orientation lock independently; this is a no-op unless at
        least one is enabled.
        """
        state = self.server.state

        lock_position = self.position_locked
        lock_orientation = self.orientation_locked
        if not (lock_position or lock_orientation):
            return
        if self._snap_selection() is None:
            return

        missing = False

        if lock_position:
            center = self._snap_centroid(frame)
            if center is None:
                missing = True
            else:
                self.app.mpr.set_origin(center)

        if lock_orientation:
            plane = self._aligned_plane(frame)
            if plane is None:
                missing = True
            else:
                self._apply_alignment(plane)

        state.snap_no_interface = missing

    def _invalidate_lock_cache(self):
        """Drop the memoized per-frame centroids and interface planes."""
        self._lock_centroids: dict[tuple[int, int], list[float] | None] = {}
        self._lock_planes: dict[tuple[int, int], tuple | None] = {}
        self._travel_remainder = 0.0
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

    def _interface_plane(self, frame: int, interface: int = INTERFACE_AB):
        """Memoized dominant plane of one selected interface at ``frame``.

        The B|C plane is anchored on the A|B plane of the same frame. Fitted
        independently the two would not share in-plane axes -- ``plane_basis``
        derives them from a fixed anatomical reference -- and travelling between
        them would spin the view rather than simply tilt it. The anchor also
        settles B|C's normal sign from A|B's, which is the direction of travel.
        """
        key = (frame, interface)
        if key not in self._lock_planes:
            selection = self._snap_selection()
            if selection is None:
                self._lock_planes[key] = None
            else:
                seg = selection[0]
                left, right = self._interface_labels(selection, interface)
                anchor = self._align_reference
                if interface == INTERFACE_BC:
                    start = self._interface_plane(frame, INTERFACE_AB)
                    anchor = None if start is None else start[1]
                plane = seg.interface_plane(left, right, frame, anchor=anchor)
                # Carry the first plane's basis onto later frames, so the view
                # does not spin as the normal moves.
                if (
                    plane is not None
                    and interface == INTERFACE_AB
                    and self._align_reference is None
                ):
                    self._align_reference = plane[1]
                self._lock_planes[key] = plane
        return self._lock_planes[key]

    def _traverse_plane(self, frame: int, fraction: float | None = None):
        """The plane at ``fraction`` along the path, defaulting to the slider."""
        start = self._interface_plane(frame, INTERFACE_AB)
        end = self._interface_plane(frame, INTERFACE_BC)
        if start is None or end is None:
            return None
        if fraction is None:
            fraction = self._traverse_fraction()
        return interpolate_planes(start, end, fraction)

    def _aligned_plane(self, frame: int, fraction: float | None = None):
        """The plane the current mode aligns to at ``frame``."""
        if getattr(self.server.state, "snap_mode", SnapMode.LABEL) == SnapMode.TRAVERSE:
            return self._traverse_plane(frame, fraction)
        return self._interface_plane(frame)

    def traverse_pose(
        self, frame: int, fraction: float
    ) -> tuple[list[float], np.ndarray] | None:
        """Origin and plane basis ``fraction`` of the way along the path.

        The origin is in ITK, as ``interface_centroid`` returns it. None when
        the selection is incomplete or either interface is missing, so a caller
        sampling several fractions can give up on the first gap.
        """
        plane = self._traverse_plane(frame, fraction)
        if plane is None:
            return None
        centroid = self._snap_centroid(frame, fraction) or plane[0]
        return [float(v) for v in centroid], plane[1]

    def _apply_alignment(self, plane):
        """Replace the alignment step so the views are based on ``plane``.

        The step goes first in the sequence, so any rotations the user has added
        are applied on top of it and keep their meaning relative to the plane: a
        Z rotation spins within it, X and Y tilt out of it.
        """

        state = self.server.state
        _, axes, flatness = plane
        state.interface_flatness = flatness

        quaternion = rotation_matrix_to_quaternion(alignment_rotation(axes))

        quaternion = self.convention.quaternion_from_itk(quaternion)

        def with_alignment(steps):
            kept = _without_alignment(steps)
            return [
                RotationStep(
                    quaternion=quaternion,
                    visible=True,
                    name=ALIGN_STEP_NAME,
                    name_editable=False,
                    deletable=True,
                ),
                *kept,
            ]

        self.app.rotations.edit_steps(with_alignment)

    def swap_groups(self, **kwargs):
        """Reverse the selection, and with it the interface normal.

        In interface mode that exchanges the two groups, viewing the interface
        from the other side. In traverse mode it exchanges the outer two, which
        keeps both interfaces but reverses the direction of travel, so the
        slider starts at the landmark it used to end on.

        Only the selection changes; the views move when the user aligns, the
        same as any other edit to the groups. The exception is an orientation
        lock, which is a standing request to follow the interface.

        The cache is dropped here rather than left to the selection listener,
        which only fires once this returns: the stale anchor would otherwise
        flip the recomputed normal straight back.
        """
        state = self.server.state
        mode = getattr(state, "snap_mode", SnapMode.LABEL)
        if mode not in PLANAR_MODES:
            return

        if mode == SnapMode.TRAVERSE:
            state.snap_labels_a, state.snap_labels_c = (
                list(getattr(state, "snap_labels_c", [])),
                list(getattr(state, "snap_labels_a", [])),
            )
            state.snap_traverse = 100 - getattr(state, "snap_traverse", 0)
        else:
            state.snap_labels_a, state.snap_labels_b = (
                list(getattr(state, "snap_labels_b", [])),
                list(getattr(state, "snap_labels_a", [])),
            )
        self._invalidate_lock_cache()

        if getattr(state, "snap_orientation_locked", False):
            self.align_to_interface()

    def _on_snap_traverse_changed(self, **kwargs):
        """Follow the slider as it is dragged, rather than waiting for Align."""
        if getattr(self.server.state, "snap_mode", SnapMode.LABEL) != SnapMode.TRAVERSE:
            return
        if self._snap_selection() is None:
            return
        self.align_to_interface()

    def align_to_interface(self, **kwargs):
        """Rotate the MPR views into the plane the current selection names."""
        state = self.server.state
        if getattr(state, "snap_mode", SnapMode.LABEL) not in PLANAR_MODES:
            return
        if self._snap_selection() is None:
            return

        frame = self._frame
        plane = self._aligned_plane(frame)
        if plane is None:
            state.snap_no_interface = True
            state.interface_flatness = 0.0
            return

        state.snap_no_interface = False
        self._apply_alignment(plane)
        self.app.mpr.set_origin(self._snap_centroid(frame) or plane[0])
