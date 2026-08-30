"""Lightweight stand-ins for the trame server and Scene used by the controllers.

``make_logic`` wires the real controllers over those stand-ins, so a test drives
the same code the application does with only the state it cares about named.
"""

# Third Party
import numpy as np

# Internal
from cardio.logic.mpr import MPRController
from cardio.logic.rotations import RotationController
from cardio.logic.snap import SnapController
from cardio.logic.tiles import TileController
from cardio.orientation import (
    AngleUnits,
    IndexOrder,
    axcode_transform_matrix,
    cumulative_rotation_matrix,
)
from cardio.rotation import RotationMetadata, RotationSequence
from cardio.tile_views import TileViews


class FakeState(dict):
    """Stand-in for trame's state: attribute access over a plain dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeController:
    """Stand-in for trame's controller: any attribute is a recording no-op."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)

        return record


class FakeScene:
    """Only the Scene attributes the snap/lock/align/tile paths touch."""

    def __init__(
        self,
        segmentations,
        index_order=IndexOrder.ITK,
        angle_units=AngleUnits.DEGREES,
        volumes=None,
        tile_rows=3,
        tile_cols=3,
        mpr_views=None,
    ):
        self.segmentations = segmentations
        self.volumes = list(volumes or [])
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.tile_views = None
        self.mpr_views = mpr_views
        self.mpr_rotation_sequence = RotationSequence(
            metadata=RotationMetadata(index_order=index_order, angle_units=angle_units)
        )

    def setup_tile_render_window(self):
        if self.tile_views is None:
            self.tile_views = TileViews()
            self.tile_views.set_grid(self.tile_rows, self.tile_cols)


class FakeApp:
    """The parts of Logic a controller reaches through ``self.app``.

    Builds the real controllers rather than stubbing them, so the tests drive
    the same code the application does.
    """

    def __init__(self, scene, **state):
        self.scene = scene
        self.server = type(
            "Server",
            (),
            {"state": FakeState(state), "controller": FakeController()},
        )()
        self.rotations = RotationController(self)
        self.mpr = MPRController(self)
        self.snap = SnapController(self)
        self.tiles = TileController(self)


def snap_state(**overrides) -> dict:
    """The snap panel's state, as Logic registers it, with two groups chosen.

    Returned fresh each call: the values are mutable and the controllers write
    through them, so two apps must not share one.
    """
    state = dict(
        snap_seg_label="s",
        snap_mode="interface",
        snap_labels_a=[1],
        snap_labels_b=[2],
        snap_labels_c=[],
        snap_traverse=0,
        snap_locked=False,
        snap_orientation_locked=False,
        snap_no_interface=False,
        interface_flatness=0.0,
        frame=0,
        mpr_origin=[0.0, 0.0, 0.0],
        mpr_rotation_data={"angles_list": []},
    )
    state.update(overrides)
    return state


def make_logic(segmentation, index_order=IndexOrder.ITK, **overrides) -> FakeApp:
    """Real controllers over a scene holding one segmentation, in interface mode."""
    state = snap_state(snap_seg_label=segmentation.label, **overrides)
    return FakeApp(FakeScene([segmentation], index_order=index_order), **state)


def traverse_logic(segmentation, **overrides) -> FakeApp:
    """``make_logic`` in traverse mode, with the third group chosen as well."""
    overrides.setdefault("snap_labels_c", [3])
    overrides.setdefault("snap_mode", "traverse")
    return make_logic(segmentation, **overrides)


def align_at(logic: FakeApp, traverse: int):
    """Move the slider and align, as dragging it does."""
    logic.server.state.snap_traverse = traverse
    logic.snap.align_to_interface()


def axial_normal(logic: FakeApp) -> np.ndarray:
    """The axial slice normal the reslice pipeline will actually use."""
    sequence, angles = logic.rotations.visible_rotation_data()
    cumulative = cumulative_rotation_matrix(sequence, angles, AngleUnits.DEGREES)
    return (cumulative @ axcode_transform_matrix("LPS", "LAS"))[:, 2]
