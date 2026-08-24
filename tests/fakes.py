"""Lightweight stand-ins for the trame server and Scene used by Logic."""

from cardio.orientation import AngleUnits, IndexOrder
from cardio.rotation import RotationMetadata, RotationSequence


class FakeState(dict):
    """Stand-in for trame's state: attribute access over a plain dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeScene:
    """Only the Scene attributes the snap/lock/align paths touch."""

    def __init__(
        self,
        segmentations,
        index_order=IndexOrder.ITK,
        angle_units=AngleUnits.DEGREES,
    ):
        self.segmentations = segmentations
        self.mpr_rotation_sequence = RotationSequence(
            metadata=RotationMetadata(index_order=index_order, angle_units=angle_units)
        )
