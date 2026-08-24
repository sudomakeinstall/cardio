"""Snapping the MPR origin and orientation to a segmentation feature."""

# Internal
from ..orientation import (
    axcode_transform_matrix,
    rotation_matrix_to_quaternion,
)
from ..rotation import RotationStep
from .base import Controller

ALIGN_STEP_NAME = "Interface plane"


class SnapController(Controller):
    """Centroid snapping, interface alignment and the per-frame locks."""

    def __init__(self, app):
        super().__init__(app)
        self._invalidate_lock_cache()

    def register(self):
        if not self.scene.segmentations:
            return

        state = self.server.state
        state.change("snap_seg_label")(self._on_snap_seg_changed)
        state.change("snap_mode", "snap_labels_a", "snap_labels_b")(
            self._on_snap_selection_changed
        )
        state.change("snap_locked")(self._on_snap_lock_changed)
        state.change("snap_orientation_locked")(self._on_snap_orientation_lock_changed)

        state.snap_mode = "label"
        state.snap_seg_label = self.scene.segmentations[0].label
        state.snap_labels_a = []
        state.snap_labels_b = []
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

    def register_initial_labels(self):
        """Populate the label pickers, once a segmentation is selected."""
        if self.scene.segmentations:
            self._on_snap_seg_changed()

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

    def snap_to_centroid(self, **kwargs):
        if getattr(self.server.state, "snap_mode", "label") == "reset":
            self.app.mpr.reset_mpr_origin()
            return
        if self._snap_selection() is None:
            return

        frame = getattr(self.server.state, "frame", 0)
        center = self._snap_centroid(frame)
        if center is None:
            self.server.state.snap_no_interface = True
            return

        self.server.state.snap_no_interface = False
        self.app.mpr.set_origin(center)

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
                self.app.mpr.set_origin(center)

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

        state = self.server.state
        _, axes, flatness = plane
        state.interface_flatness = flatness

        # The reslice matrix is cumulative @ view_transform, so the rotation that
        # puts the axial view in the interface plane satisfies R @ T_axial = axes.
        target = axes @ axcode_transform_matrix("LPS", "LAS").T
        quaternion = rotation_matrix_to_quaternion(target)

        quaternion = self.convention.quaternion_from_itk(quaternion)

        def with_alignment(steps):
            kept = [step for step in steps if step.name != ALIGN_STEP_NAME]
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
        """Exchange the two label groups, reversing the interface normal.

        Only the selection changes; the views move when the user aligns, the
        same as any other edit to the groups. The exception is an orientation
        lock, which is a standing request to follow the interface.

        The cache is dropped here rather than left to the selection listener,
        which only fires once this returns: the stale anchor would otherwise
        flip the recomputed normal straight back.
        """
        state = self.server.state
        if getattr(state, "snap_mode", "label") != "interface":
            return

        state.snap_labels_a, state.snap_labels_b = (
            list(getattr(state, "snap_labels_b", [])),
            list(getattr(state, "snap_labels_a", [])),
        )
        self._invalidate_lock_cache()

        if getattr(state, "snap_orientation_locked", False):
            self.align_to_interface()

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
        self.app.mpr.set_origin(self._snap_centroid(frame) or plane[0])
