"""Every trame state variable the app declares, in one place.

The keys used to be declared in three: a controller's ``register()``, the
``UI`` constructor, and the ``(key, default)`` tuple on a widget. Nothing
enumerated them, so nothing could ask the questions the app now needs to ask in
bulk -- which keys describe what is on screen, which are browsing state, and
which ``Scene`` field each of the first kind is seeded from.

``tests/test_config_coverage.py`` used to carry that enumeration as test data.
It still walks the source to keep the registry honest; the registry itself now
lives here, where the application can read it too.
"""

# System
import dataclasses as dc
import enum


class Scope(enum.StrEnum):
    """What a variable is, for the parts of the app that treat state in bulk."""

    DOCUMENT = "document"
    """What is on screen: seeded from a ``Scene`` field, and written back to one."""

    SESSION = "session"
    """Browsing state, which starts fresh every session and is never saved."""

    ITEMS = "items"
    """Picker contents and other values derived from the scene or the layout."""


@dc.dataclass(frozen=True)
class Variable:
    """One state key, and what the app is allowed to assume about it."""

    key: str
    scope: Scope
    source: str = ""
    """For a document variable, the dotted ``Scene`` field it is seeded from."""

    reason: str = ""
    """For the other two scopes, why this is not something a config decides."""

    def __post_init__(self):
        if self.scope is Scope.DOCUMENT:
            if not self.source:
                raise ValueError(f"{self.key} is document state with no Scene field")
            if self.reason:
                raise ValueError(f"{self.key} is document state; a reason says why not")
        else:
            if not self.reason.strip():
                raise ValueError(f"{self.key} is {self.scope} with no reason given")
            if self.source:
                raise ValueError(
                    f"{self.key} is {self.scope}; a source would be unused"
                )


def _document(**sources: str) -> list[Variable]:
    return [
        Variable(key, Scope.DOCUMENT, source=source) for key, source in sources.items()
    ]


def _session(**reasons: str) -> list[Variable]:
    return [
        Variable(key, Scope.SESSION, reason=reason) for key, reason in reasons.items()
    ]


def _items(**reasons: str) -> list[Variable]:
    return [
        Variable(key, Scope.ITEMS, reason=reason) for key, reason in reasons.items()
    ]


# The dotted paths are resolved against Scene by the coverage test, so a field
# renamed out from under one of these fails there rather than at runtime.
DOCUMENT = _document(
    active_volume_label="active_volume_label",
    angle_units="mpr_rotation_sequence.metadata.angle_units",
    bpm="playback.bpm",
    bpr="playback.bpr",
    camera_lock="view.camera_lock",
    capture_format="capture_format",
    drawer_sections="view.drawer_sections",
    frame="current_frame",
    help_overlay_visible="view.help_visible",
    incrementing="playback.incrementing",
    index_order="mpr_rotation_sequence.metadata.index_order",
    maximized_view="view.layout",
    metadata_overlay_visible="view.metadata_visible",
    mpr_crosshairs_enabled="mpr_crosshairs_enabled",
    mpr_level="mpr_level",
    mpr_origin="mpr_origin",
    mpr_rotation_data="mpr_rotation_sequence",
    mpr_segmentation_opacity="mpr_segmentation_opacity",
    mpr_window="mpr_window",
    mpr_window_level_preset="mpr_window_level_preset",
    playback_quality="playback.quality",
    playback_resolution="playback.resolution",
    rotating="playback.rotating",
    snap_labels_a="snap.labels_a",
    snap_labels_b="snap.labels_b",
    snap_labels_c="snap.labels_c",
    snap_locked="snap.locked",
    snap_mode="snap.mode",
    snap_orientation_locked="snap.orientation_locked",
    snap_seg_label="snap.segmentation_label",
    snap_traverse="snap.traverse",
    theme_mode="view.theme",
    tile_cols="tile_cols",
    tile_rows="tile_rows",
)

# A reason here is a decision, not an excuse: anything a user would want to
# open the app in, or to find again in a saved session, belongs in DOCUMENT.
SESSION = _session(
    capture_ok="whether the last capture wrote anything",
    capture_progress="how far the running capture has got",
    capture_running="a capture in flight does not survive the session",
    capture_saved_at="written when a capture finishes",
    capture_summary="the one line the drawer shows about the last capture",
    clip_depth="derived from the camera's clipping range at build time",
    interface_flatness="measured from the interface the current selection fits",
    metadata_object="which object's metadata sheet is showing is browsing state",
    playing="starting playback on launch is a behaviour, not view state",
    rotations_saved_at="written when a save happens",
    rotations_stale="derived from edits since the last save",
    snap_no_interface="derived from whether the selection has an interface",
    trame__busy="trame's own, raised while a round trip is in flight",
    trame__title="trame's own, set from the version",
)

ITEMS = _items(
    angle_units_items="the two angle units, spelled for the picker",
    camera_lock_items="the CameraLock members, spelled for the picker",
    capture_available="the viewports the layout currently draws",
    capture_formats="the CaptureFormat members, spelled for the picker",
    event_types="the interactor events the views forward, fixed at build time",
    index_order_items="the two index orders, spelled for the picker",
    metadata_pages="one page per object in the scene",
    mpr_presets="the window/level presets, spelled for the picker",
    snap_available_labels="the labels present in the chosen segmentation",
    snap_seg_items="one entry per segmentation in the scene",
    tile_sizes="the grid sizes offered, fixed at build time",
    volume_items="one entry per volume in the scene",
)

VARIABLES: dict[str, Variable] = {
    variable.key: variable for variable in (*DOCUMENT, *SESSION, *ITEMS)
}


def keys(scope: Scope) -> list[str]:
    """Every literal key in ``scope``, sorted."""
    return sorted(key for key, var in VARIABLES.items() if var.scope is scope)


def source_of(key: str) -> str:
    """The dotted ``Scene`` field ``key`` is seeded from, empty if it has none."""
    return VARIABLES[key].source


# Which field on the object model seeds each of ObjectState's per-object keys.
# The clip bounds are absent deliberately: they come from the object's geometry
# rather than from a field.
OBJECT_SOURCES: dict[str, str] = {
    "visibility": "visible",
    "clipping": "clipping_enabled",
    "preset": "transfer_function_preset",
    "mpr_overlay": "mpr_overlay",
}
