"""The MPR rotation sequence: the single source of rotation truth."""

# Internal
from ..action import action
from ..convention import exchange_point
from ..orientation import (
    AngleUnits,
    IndexOrder,
    angle_from_degrees,
    cumulative_rotation_matrix,
)
from ..rotation import RotationSequence, RotationStep
from .base import Controller

# The mouse-driven Euler steps, named by the physical axis each one turns
# about rather than by its letter, which changes with the index order.
MOUSE_STEP_NAMES = {"X": "Mouse L-R", "Y": "Mouse P-A", "Z": "Mouse S-I"}


class RotationController(Controller):
    """Owns ``mpr_rotation_data`` and the units/index-order mirrors."""

    # All three come off the same sequence, so they cannot fall out of step;
    # the two metadata paths nest inside mpr_rotation_data's on purpose.
    seeds = ("mpr_rotation_data", "angle_units", "index_order")

    def register(self):
        state = self.server.state
        state.change("angle_units")(self.sync_angle_units)
        state.change("index_order")(self.sync_index_order)

    def seed(self):
        super().seed()
        state = self.server.state
        state.angle_units_items = [
            {"text": "Degrees", "value": "degrees"},
            {"text": "Radians", "value": "radians"},
        ]
        state.index_order_items = [
            {"text": "ITK (X=L, Y=P, Z=S)", "value": "itk"},
            {"text": "Roma (X=S, Y=P, Z=L)", "value": "roma"},
        ]

    def rotation_sequence(self) -> RotationSequence:
        """The rotation state as its model, validated on the way in.

        The steps come from trame state, which is what the UI edits. Metadata
        falls back to the scene's, so a state payload that predates a metadata
        write cannot silently reset the convention to the model defaults.
        """
        data = dict(self.server.state.mpr_rotation_data or {})
        data.setdefault("metadata", self.scene.mpr_rotation_sequence.metadata)
        data.setdefault("angles_list", [])
        return RotationSequence(**data)

    def publish(self, sequence: RotationSequence):
        """The only place rotation state is written back.

        Keeps the scene's metadata and the UI's mirror variables in step with
        the sequence, so the three representations cannot drift.
        """
        self.scene.mpr_rotation_sequence = sequence
        self.server.state.mpr_rotation_data = sequence.model_dump(mode="json")
        self.server.state.angle_units = sequence.metadata.angle_units.value
        self.server.state.index_order = sequence.metadata.index_order.value

    def edit_steps(self, edit):
        """Apply ``edit`` to the list of steps and publish the result."""
        sequence = self.rotation_sequence()
        sequence.angles_list = edit(list(sequence.angles_list))
        self.publish(sequence)

    def visible_rotation_data(self):
        """Rotation sequence and angles for the visible steps, in ITK for VTK."""
        rotation_data = self.server.state.mpr_rotation_data
        return self.convention.visible_sequence_to_itk(
            rotation_data.get("angles_list", [])
        )

    def rotation_matrix(self, exclude: str | None = None):
        """The visible rotation composed into one 3x3 matrix, in ITK.

        ``exclude`` drops every step of that name first. The steps are filtered
        before conversion because ``visible_sequence_to_itk`` keys its angles by
        position, so removing a step afterwards would misalign them.
        """
        steps = (self.server.state.mpr_rotation_data or {}).get("angles_list", [])
        if exclude is not None:
            steps = [step for step in steps if step.get("name") != exclude]
        sequence, angles = self.convention.visible_sequence_to_itk(steps)
        return cumulative_rotation_matrix(sequence, angles, self.convention.angle_units)

    def turn_mouse(self, axis: str, degrees: float):
        """Add ``degrees`` about ITK ``axis`` as the innermost rotation.

        The step goes last, where the sequence composes it first. That is the
        only place a turn about a fixed base axis is also a turn about the axis
        the view shows now, which is what keeps a roll in its own plane however
        much rotation is stacked above it.

        Dragging on about the same axis accumulates into that step. Changing
        axis starts a new one rather than reordering the sequence: reordering
        would recompose the rotation and move the views out from under the
        user, and putting the new axis anywhere but last would tilt the plane
        being rolled instead of spinning it.
        """
        sequence = self.rotation_sequence()
        steps = list(sequence.angles_list)
        name = MOUSE_STEP_NAMES[axis]

        angle = angle_from_degrees(
            self.convention.angle_from_itk(degrees), sequence.metadata.angle_units
        )

        innermost = steps[-1] if steps else None
        if innermost is not None and innermost.name == name and innermost.visible:
            innermost.angle = (innermost.angle or 0.0) + angle
        else:
            steps.append(
                RotationStep(
                    axis=self.convention.axis_from_itk(axis),
                    angle=angle,
                    name=name,
                    name_editable=False,
                )
            )

        sequence.angles_list = steps
        self.publish(sequence)

    @action("add_rotation")
    def add_mpr_rotation(self, axis: str):
        """Append a new Euler rotation about ``axis``."""

        def append(steps):
            return [*steps, RotationStep(axis=axis, angle=0)]

        self.edit_steps(append)

    @action("remove_rotation")
    def remove_mpr_rotation(self, index: int):
        """Remove the rotation at ``index``."""

        def without(steps):
            if 0 <= index < len(steps):
                steps.pop(index)
            return steps

        self.edit_steps(without)

    @action("reset_rotation_angle")
    def reset_rotation_angle(self, index: int):
        """Zero the angle of the rotation at ``index``."""

        def zeroed(steps):
            if 0 <= index < len(steps):
                steps[index].angle = 0.0
            return steps

        self.edit_steps(zeroed)

    @action("reset_rotations")
    def reset_mpr_rotations(self):
        """Drop every rotation, back to a default sequence."""
        self.publish(RotationSequence())

    def sync_angle_units(self, angle_units, **kwargs):
        """Re-express the stored angles when the user switches units."""
        try:
            units = AngleUnits(angle_units)
        except ValueError:
            return

        sequence = self.rotation_sequence()
        if units == sequence.metadata.angle_units:
            return

        self.publish(sequence.with_units(units))

    def sync_index_order(self, index_order, **kwargs):
        """Re-express the stored rotations and origin when the order switches."""
        if isinstance(index_order, IndexOrder):
            order = index_order
        else:
            try:
                order = IndexOrder(str(index_order).lower())
            except ValueError as error:
                raise ValueError(f"Unrecognized index order: {index_order}") from error

        sequence = self.rotation_sequence()
        if order == sequence.metadata.index_order:
            return

        self.publish(sequence.with_index_order(order))

        # mpr_origin lives in state rather than in the sequence, so it is
        # exchanged here rather than by with_index_order.
        mpr_origin = self.server.state.mpr_origin
        if mpr_origin is not None and len(mpr_origin) == 3:
            self.server.state.mpr_origin = exchange_point(mpr_origin)
