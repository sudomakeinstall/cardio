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
import functools as ft

# Third Party
import pydantic_core

# Internal
from .state import VIEWPORTS, ObjectState, screenshot_viewport
from .view import Layout


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

    seeded_by: str = ""
    """For a document variable its controller writes, why the table cannot.

    Every other document key is written from its ``source`` by the seeding
    pass. These are the ones where reading the field is not the whole story --
    a fallback, a deferral, or a value that mirrors VTK rather than the config.
    """

    def __post_init__(self):
        if self.scope is Scope.DOCUMENT:
            if not self.source:
                raise ValueError(f"{self.key} is document state with no Scene field")
            if self.reason:
                raise ValueError(f"{self.key} is document state; a reason says why not")
        else:
            if self.seeded_by:
                raise ValueError(
                    f"{self.key} is {self.scope}; only document state is seeded"
                )
            if not self.reason.strip():
                raise ValueError(f"{self.key} is {self.scope} with no reason given")
            if self.source:
                raise ValueError(
                    f"{self.key} is {self.scope}; a source would be unused"
                )


def _declare(scope: Scope, **entries: str | tuple[str, str]) -> list[Variable]:
    """Every key in ``scope``, the string filling whichever field it wants.

    A document key whose controller writes it gives a pair -- its source, and
    why the seeding pass cannot write it -- so that the exceptions sit in the
    table beside the rule they are exceptions to.
    """
    field = "source" if scope is Scope.DOCUMENT else "reason"
    declared = []
    for key, entry in entries.items():
        value, seeded_by = entry if isinstance(entry, tuple) else (entry, "")
        if seeded_by and not seeded_by.strip():
            raise ValueError(f"{key} says its controller writes it, but not why")
        declared.append(Variable(key, scope, seeded_by=seeded_by, **{field: value}))
    return declared


# The dotted paths are resolved against Scene by the coverage test, so a field
# renamed out from under one of these fails there rather than at runtime.
DOCUMENT = _declare(
    Scope.DOCUMENT,
    active_volume_label=(
        "active_volume_label",
        (
            "an empty one means the first volume, and until the render views "
            "exist there is nothing to make active -- so MPRController writes "
            "an empty string and finalize_mpr_initialization the real one"
        ),
    ),
    # Nested inside mpr_rotation_data's source deliberately: the sequence is
    # written whole and these two mirror part of it. document._mirrored is the
    # other side of that -- saving skips them, seeding writes them.
    angle_units="mpr_rotation_sequence.metadata.angle_units",
    bpm="playback.bpm",
    bpr="playback.bpr",
    camera_lock="view.camera_lock",
    console_visible="view.console_visible",
    cameras=(
        "view.cameras",
        (
            "the state mirrors where VTK's cameras actually ended up, which is "
            "where the configured poses put them only until something moves"
        ),
    ),
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
    snap_seg_label=(
        "snap.segmentation_label",
        "an empty one means the first segmentation, and a scene may have none",
    ),
    snap_traverse="snap.traverse",
    theme_mode="view.theme",
    tile_cols="tile_cols",
    tile_rows="tile_rows",
)

# A reason here is a decision, not an excuse: anything a user would want to
# open the app in, or to find again in a saved session, belongs in DOCUMENT.
SESSION = _declare(
    Scope.SESSION,
    capture_ok="whether the last capture wrote anything",
    capture_progress="how far the running capture has got",
    capture_running="a capture in flight does not survive the session",
    capture_saved_at="written when a capture finishes",
    capture_summary="the one line the drawer shows about the last capture",
    console_entries="the log of what was done starts empty every session",
    console_input="what is half-typed at the prompt is not a thing to save",
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

ITEMS = _declare(
    Scope.ITEMS,
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


def keys_in_scope(scope: Scope) -> list[str]:
    """Every literal key in ``scope``, sorted."""
    return sorted(key for key, var in VARIABLES.items() if var.scope is scope)


# The keys a config and the running state spell differently, and the way back.
# ``maximized_view`` is the only one: it carries an empty string for the quad
# view, which a config calls by its name like any other layout.
TO_CONFIG = {"maximized_view": lambda value: Layout.from_state(value).value}


def to_config(key: str, value):
    """``value`` as the ``Scene`` field behind ``key`` spells it."""
    convert = TO_CONFIG.get(key)
    return convert(value) if convert else value


# The way back in. Only the one key needs saying: everything else state holds
# is what ``mode="json"`` makes of the field, which is not a choice this app
# gets to make and so is not worth a table of its own.
TO_STATE = {"maximized_view": lambda layout: layout.state_value}


def to_state(key: str, value):
    """The ``Scene`` field behind ``key`` as the state variable spells it."""
    convert = TO_STATE.get(key)
    return convert(value) if convert else pydantic_core.to_jsonable_python(value)


def read(scene, path: str):
    """The value at the dotted ``Scene`` field ``path``.

    Every branch a path passes through is a model with a default, so a scene
    always has one to walk -- there is no missing middle to guard against.
    """
    return ft.reduce(getattr, path.split("."), scene)


def state_value(scene, key: str):
    """What ``key`` should hold, given ``scene``.

    The one way state is told what a config says, as ``to_config`` is the one
    way a config is told what state says.
    """
    return to_state(key, read(scene, source_of(key)))


def source_of(key: str) -> str:
    """The dotted ``Scene`` field ``key`` is seeded from, empty if it has none."""
    return VARIABLES[key].source


# Which field on the object model seeds each of ObjectState's per-object keys,
# one key to one field. The clip bounds are absent because they are not that
# shape: three range sliders make up the one ``crop`` box, so each direction of
# that mapping is spelled where it is used.
OBJECT_SOURCES: dict[str, str] = {
    "visibility": "visible",
    "clipping": "clipping_enabled",
    "preset": "transfer_function_preset",
    "mpr_overlay": "mpr_overlay",
}


# The one config field that fans out to a family of keys rather than to a
# single one: a tick per viewport going in, the list of the ticked ones coming
# back. Neither direction is a copy, so both are spelled here rather than
# separately at each end, where they used to name the field as a literal.
VIEWPORT_TICKS = "screenshot_viewports"


def viewport_tick_keys() -> list[str]:
    """The tick key for every viewport a capture can be taken from."""
    return [screenshot_viewport(viewport) for viewport in VIEWPORTS]


def viewport_ticks(scene) -> dict[str, bool]:
    """Each viewport's tick, from the list of the ones the config asks for."""
    chosen = read(scene, VIEWPORT_TICKS)
    return {screenshot_viewport(viewport): viewport in chosen for viewport in VIEWPORTS}


def ticked_viewports(state) -> list[str]:
    """The viewports whose ticks are on, as the config field spells them."""
    return [viewport for viewport in VIEWPORTS if state[screenshot_viewport(viewport)]]


def object_state_value(obj, prop: str):
    """``obj``'s key for ``prop`` and what it should hold, or ``(None, None)``.

    A renderable only has the key if it has the field behind it, which is the
    same partition the save direction makes coming back.
    """
    field = OBJECT_SOURCES[prop]
    if field not in type(obj).model_fields:
        return None, None

    key = getattr(ObjectState.of(obj), prop)
    return key, to_state(key, getattr(obj, field))


def object_document_keys(obj) -> list[str]:
    """Every key saying what ``obj`` is showing.

    The clip bounds are here despite having no field behind them: a crop is
    part of what is on screen whether it was configured or taken from the
    object's own extent.
    """
    named = [
        getattr(ObjectState.of(obj), prop)
        for prop in OBJECT_SOURCES
        if OBJECT_SOURCES[prop] in type(obj).model_fields
    ]
    return [*named, *ObjectState.of(obj).clip_bounds]


def document_keys(scene) -> list[str]:
    """Every key that says what ``scene`` is currently showing.

    What a saved session writes down and what an undo puts back. The literal
    keys are the same whatever the scene holds; the rest are per object, so the
    scene is the only thing that can say what they are.
    """
    keys = keys_in_scope(Scope.DOCUMENT)
    keys.extend(viewport_tick_keys())
    for obj in scene.renderables:
        keys.extend(object_document_keys(obj))
    return keys
