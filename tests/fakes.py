"""Lightweight stand-ins for the trame server and Scene used by the controllers."""

# Internal
from cardio.logic.mpr import MPRController
from cardio.logic.rotations import RotationController
from cardio.logic.snap import SnapController
from cardio.logic.tiles import TileController
from cardio.orientation import AngleUnits, IndexOrder
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
    ):
        self.segmentations = segmentations
        self.volumes = list(volumes or [])
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.tile_views = None
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
