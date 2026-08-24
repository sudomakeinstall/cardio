"""The single boundary between the user's index order and the one VTK is given.

``mpr_origin`` and the steps in ``mpr_rotation_data`` are stored in whichever
index order the user selected. VTK is only ever handed ITK. ``Convention`` is
the one place that crosses that boundary: no other module should permute a
coordinate, an axis or a quaternion by hand.
"""

# System
import dataclasses as dc

# Internal
from .orientation import AngleUnits, IndexOrder

# ROMA (X=S, Y=P, Z=L) and ITK (X=L, Y=P, Z=S) differ by exchanging the first
# and last index. That exchange is a reflection, so a rotation carried across it
# keeps its magnitude but reverses its sense -- hence the negated angles and the
# negated quaternion components below. The mapping is its own inverse, so one
# implementation serves both directions.
AXIS_EXCHANGE = {"X": "Z", "Y": "Y", "Z": "X"}


def exchange_point(point) -> list[float]:
    """Swap the first and last index of a point or vector."""
    return [point[2], point[1], point[0]]


def exchange_axis(axis: str) -> str:
    """Swap a rotation axis name between the two orders."""
    return AXIS_EXCHANGE[axis]


def exchange_angle(angle: float) -> float:
    """Reverse a rotation's sense, as the reflected axes require."""
    return -angle


def exchange_quaternion(quaternion) -> list[float]:
    """Swap a quaternion [x, y, z, w] between the two orders."""
    x, y, z, w = quaternion
    return [-z, -y, -x, w]


def exchange_step(step: dict) -> dict:
    """A rotation step re-expressed in the other index order.

    Used when the user switches convention, where the exchange always applies --
    unlike the ``Convention`` methods, which apply it only when the current
    order is not already the one being converted to.
    """
    converted = dict(step)
    if converted.get("quaternion") is not None:
        converted["quaternion"] = exchange_quaternion(converted["quaternion"])
    else:
        converted["axis"] = exchange_axis(converted["axis"])
        converted["angle"] = exchange_angle(converted.get("angle", 0))
    return converted


@dc.dataclass(frozen=True)
class Convention:
    """The index order and angle units the user-facing MPR state is written in."""

    index_order: IndexOrder = IndexOrder.ITK
    angle_units: AngleUnits = AngleUnits.RADIANS

    @classmethod
    def from_metadata(cls, metadata) -> "Convention":
        """Read the convention off a ``RotationMetadata``."""
        return cls(index_order=metadata.index_order, angle_units=metadata.angle_units)

    @property
    def is_itk(self) -> bool:
        return self.index_order == IndexOrder.ITK

    def point_to_itk(self, point) -> list[float]:
        """A point or vector in this convention, expressed in ITK."""
        if self.is_itk:
            return list(point)
        return exchange_point(point)

    def point_from_itk(self, point) -> list[float]:
        """An ITK point or vector, expressed in this convention.

        The index exchange is an involution, so this is ``point_to_itk`` again.
        """
        return self.point_to_itk(point)

    def axis_to_itk(self, axis: str) -> str:
        """A rotation axis name in this convention, expressed in ITK."""
        if self.is_itk:
            return axis
        return exchange_axis(axis)

    def angle_to_itk(self, angle: float) -> float:
        """A rotation angle in this convention, expressed in ITK."""
        if self.is_itk:
            return angle
        return exchange_angle(angle)

    def quaternion_to_itk(self, quaternion) -> list[float]:
        """A quaternion [x, y, z, w] in this convention, expressed in ITK."""
        if self.is_itk:
            return list(quaternion)
        return exchange_quaternion(quaternion)

    def quaternion_from_itk(self, quaternion) -> list[float]:
        """An ITK quaternion, expressed in this convention (an involution)."""
        return self.quaternion_to_itk(quaternion)

    def sequence_to_itk(self, steps) -> tuple[list[dict], dict[int, float]]:
        """Convert rotation steps into the form VTK needs.

        Returns ``(sequence, angles)`` where ``sequence`` holds one
        ``{"axis": ...}`` or ``{"quaternion": ...}`` entry per step and
        ``angles`` maps each step's position to its angle -- the shape
        ``orientation.cumulative_rotation_matrix`` consumes. Quaternion steps
        carry their rotation entirely in the quaternion, so their angle is zero.
        """
        sequence: list[dict] = []
        angles: dict[int, float] = {}

        for index, step in enumerate(steps):
            quaternion = step.get("quaternion")
            if quaternion is not None:
                sequence.append({"quaternion": self.quaternion_to_itk(quaternion)})
                angles[index] = 0
            else:
                sequence.append({"axis": self.axis_to_itk(step["axis"])})
                angles[index] = self.angle_to_itk(step.get("angle", 0))

        return sequence, angles

    def visible_sequence_to_itk(self, steps) -> tuple[list[dict], dict[int, float]]:
        """``sequence_to_itk`` over only the steps the user has left visible."""
        return self.sequence_to_itk(
            [step for step in steps if step.get("visible", True)]
        )
