"""The MPR rotation sequence: the single source of rotation truth."""

# Internal
from ..convention import exchange_point
from ..orientation import AngleUnits, IndexOrder
from ..rotation import RotationSequence, RotationStep
from .base import Controller


class RotationController(Controller):
    """Owns ``mpr_rotation_data`` and the units/index-order mirrors."""

    def register(self):
        state = self.server.state
        state.angle_units_items = [
            {"text": "Degrees", "value": "degrees"},
            {"text": "Radians", "value": "radians"},
        ]
        state.index_order_items = [
            {"text": "ITK (X=L, Y=P, Z=S)", "value": "itk"},
            {"text": "Roma (X=S, Y=P, Z=L)", "value": "roma"},
        ]

        state.change("angle_units")(self.sync_angle_units)
        state.change("index_order")(self.sync_index_order)

        controller = self.server.controller
        controller.add_x_rotation = lambda: self.add_mpr_rotation("X")
        controller.add_y_rotation = lambda: self.add_mpr_rotation("Y")
        controller.add_z_rotation = lambda: self.add_mpr_rotation("Z")
        controller.remove_rotation_event = self.remove_mpr_rotation
        controller.reset_rotation_angle = self.reset_rotation_angle
        controller.reset_rotations = self.reset_mpr_rotations

    def rotation_sequence(self) -> RotationSequence:
        """The rotation state as its model, validated on the way in.

        The steps come from trame state, which is what the UI edits. Metadata
        falls back to the scene's, so a state payload that predates a metadata
        write cannot silently reset the convention to the model defaults.
        """
        data = dict(getattr(self.server.state, "mpr_rotation_data", None) or {})
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
        rotation_data = getattr(
            self.server.state, "mpr_rotation_data", {"angles_list": []}
        )
        return self.convention.visible_sequence_to_itk(
            rotation_data.get("angles_list", [])
        )

    def add_mpr_rotation(self, axis):
        """Append a new Euler rotation about ``axis``."""

        def append(steps):
            return [*steps, RotationStep(axis=axis, angle=0)]

        self.edit_steps(append)

    def remove_mpr_rotation(self, index):
        """Remove the rotation at ``index``."""

        def without(steps):
            if 0 <= index < len(steps):
                steps.pop(index)
            return steps

        self.edit_steps(without)

    def reset_rotation_angle(self, index):
        """Zero the angle of the rotation at ``index``."""

        def zeroed(steps):
            if 0 <= index < len(steps):
                steps[index].angle = 0.0
            return steps

        self.edit_steps(zeroed)

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
        mpr_origin = getattr(self.server.state, "mpr_origin", None)
        if mpr_origin is not None and len(mpr_origin) == 3:
            self.server.state.mpr_origin = exchange_point(mpr_origin)
