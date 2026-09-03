"""Every control the UI binds has a decided config story.

The gap this guards: a state variable reached the UI, defaulted to a literal
nobody could override, and no test noticed -- the config surface grew by
accident rather than by decision. Adding a control now fails here until it is
either wired to a ``Scene`` field or declared session-local, with a reason.

This checks that the decision was *made*, not that the wiring works. That the
configured value actually reaches the running app is what the startup tests in
test_app_smoke.py assert, one field at a time.
"""

# System
import ast
import pathlib as pl
import re

# Third Party
import pydantic as pc
import pytest

# Internal
from cardio.object import Object
from cardio.scene import Scene
from cardio.segmentation import Segmentation
from cardio.view import DrawerSection
from cardio.volume import Volume

UI_DIR = pl.Path(__file__).parent.parent / "src" / "cardio" / "ui"

# Keys the UI binds by a literal name, and the Scene field each is seeded from.
CONFIGURED = {
    "active_volume_label": "active_volume_label",
    "angle_units": "mpr_rotation_sequence.metadata.angle_units",
    "bpm": "playback.bpm",
    "bpr": "playback.bpr",
    "camera_lock": "view.camera_lock",
    "drawer_sections": "view.drawer_sections",
    "frame": "current_frame",
    "help_overlay_visible": "view.help_visible",
    "incrementing": "playback.incrementing",
    "index_order": "mpr_rotation_sequence.metadata.index_order",
    "maximized_view": "view.layout",
    "mpr_segmentation_opacity": "mpr_segmentation_opacity",
    "playback_quality": "playback.quality",
    "playback_resolution": "playback.resolution",
    "rotating": "playback.rotating",
    "snap_locked": "snap.locked",
    "snap_mode": "snap.mode",
    "snap_orientation_locked": "snap.orientation_locked",
    "snap_seg_label": "snap.segmentation_label",
    "snap_traverse": "snap.traverse",
    "theme_mode": "view.theme",
}

# Keys that deliberately start fresh every session, and why. A reason here is a
# decision, not an excuse: anything a user would want to open the app in
# belongs above instead.
SESSION_LOCAL = {
    "clip_depth": "derived from the camera's clipping range at build time",
    "playing": "starting playback on launch is a behaviour, not view state",
    "rotations_saved_at": "written when a save happens",
    "rotations_stale": "derived from edits since the last save",
    "trame__title": "trame's own, set from the version",
}

# Bindings whose key is computed rather than named. Listed as source text so a
# new one shows up here rather than passing unnoticed; the value says which
# per-object field configures it, or why nothing does.
PER_OBJECT = {
    "ObjectState.of(seg).mpr_overlay": (Segmentation, "mpr_overlay"),
    "keys.clipping": (Object, "clipping_enabled"),
    "keys.preset": (Volume, "transfer_function_preset"),
    "keys.visibility": (Object, "visible"),
}

COMPUTED_KEYS = {
    "f'screenshot_viewport_{key}'": "Scene.screenshot_viewports, via the widget default",
    "key": "clip bounds, derived from each object's geometry",
    "keys.clip_panel": "whether a clip subpanel is expanded is browsing state",
    "keys.preset_panel": "whether a preset subpanel is expanded is browsing state",
    "variable": "the snap group and tile size loops, named in CONFIGURED",
    "f'mpr_rotation_data.angles_list[{i}].angle'": "a step within the sequence",
    "f'mpr_rotation_data.angles_list[{i}].axis'": "a step within the sequence",
    "f'mpr_rotation_data.angles_list[{i}].name'": "a step within the sequence",
    "f'mpr_rotation_data.angles_list[{i}].visible'": "a step within the sequence",
}


def _state_bindings() -> tuple[set[str], set[str]]:
    """Every state key the UI binds, as (literal names, computed expressions).

    Covers both halves of how a control reaches state: a ``v_model`` on a
    widget, and a direct write to ``server.state`` while building the page.
    """
    named, computed = set(), set()

    for path in sorted(UI_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "v_model":
                value = node.value
                target = value.elts[0] if isinstance(value, ast.Tuple) else value
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    named.add(target.value)
                else:
                    computed.add(ast.unparse(target))
            elif isinstance(node, ast.Assign):
                for assigned in node.targets:
                    if (
                        isinstance(assigned, ast.Attribute)
                        and isinstance(assigned.value, ast.Attribute)
                        and assigned.value.attr == "state"
                    ):
                        named.add(assigned.attr)

    return named, computed


def _resolve(model: type[pc.BaseModel], path: str):
    """The field ``path`` names, walking nested models by dotted name."""
    field = None
    for part in path.split("."):
        assert part in model.model_fields, f"{model.__name__} has no field '{part}'"
        field = model.model_fields[part]
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, pc.BaseModel):
            model = annotation
    return field


def test_every_ui_key_is_configured_or_declared_session_local():
    named, _ = _state_bindings()
    decided = set(CONFIGURED) | set(SESSION_LOCAL)

    assert named - decided == set(), (
        "UI controls with no config decision -- wire them to a Scene field in "
        f"CONFIGURED, or declare them in SESSION_LOCAL with a reason: {sorted(named - decided)}"
    )
    assert decided - named == set(), (
        f"declared but no longer bound by the UI: {sorted(decided - named)}"
    )


def test_every_computed_binding_is_accounted_for():
    _, computed = _state_bindings()
    declared = set(PER_OBJECT) | set(COMPUTED_KEYS)

    assert computed == declared, (
        "computed bindings differ from those declared; a new one needs a "
        f"config decision too. Missing: {sorted(computed - declared)}. "
        f"Stale: {sorted(declared - computed)}"
    )


@pytest.mark.parametrize("key,path", sorted(CONFIGURED.items()))
def test_configured_keys_name_a_real_scene_field(key, path):
    assert _resolve(Scene, path) is not None


@pytest.mark.parametrize("expression,target", sorted(PER_OBJECT.items()))
def test_per_object_keys_name_a_real_object_field(expression, target):
    model, field = target
    assert field in model.model_fields, f"{model.__name__} has no field '{field}'"


def test_session_local_keys_carry_a_reason():
    assert all(reason.strip() for reason in SESSION_LOCAL.values())


def test_the_scan_finds_the_ui():
    """Guards the guard: a broken walk would pass everything vacuously."""
    named, computed = _state_bindings()
    assert len(named) > 20 and len(computed) > 5


def test_the_drawer_sections_the_config_names_are_the_ones_the_ui_builds():
    """A renamed section would otherwise open nothing, silently.

    ``DrawerSection`` is what a config may name; the ``section(...)`` calls are
    what the accordion actually tracks. They have to be the same set.
    """
    source = (UI_DIR / "__init__.py").read_text()
    built = set(re.findall(r'section\(\s*"([a-z]+)"', source))

    assert built == {member.value for member in DrawerSection}
