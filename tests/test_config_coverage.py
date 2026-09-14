"""Every state variable the app declares has a decided config story.

The gap this guards: a state variable reached the UI, defaulted to a literal
nobody could override, and no test noticed -- the config surface grew by
accident rather than by decision. Adding a control now fails here until it is
either wired to a ``Scene`` field or declared session-local, with a reason.

The decisions themselves live in ``cardio.registry``, where the application can
read them. What is checked here is that the registry and the source still
describe the same app: every key the UI binds and every key the logic writes or
listens to is declared, and nothing is declared that has stopped being used.

That the configured value actually reaches the running app is what the startup
tests in test_app_smoke.py assert, one field at a time.
"""

# System
import ast
import pathlib as pl
import re

# Third Party
import pydantic as pc
import pytest

# Internal
import cardio.logic as logic
import cardio.registry as registry
import cardio.view as view
from cardio.action import declared_actions
from cardio.logic.base import Controller
from cardio.object import Object
from cardio.scene import Scene
from cardio.segmentation import Segmentation
from cardio.view import DrawerSection
from cardio.volume import Volume

SRC_DIR = pl.Path(__file__).parent.parent / "src" / "cardio"
UI_DIR = SRC_DIR / "ui"
LOGIC_DIR = SRC_DIR / "logic"

# Bindings whose key is computed rather than named. Listed as source text so a
# new one shows up here rather than passing unnoticed; the value says which
# per-object field configures it, by the ``ObjectState`` property that spells
# the key.
PER_OBJECT = {
    "ObjectState.of(seg).mpr_overlay": (Segmentation, "mpr_overlay"),
    "keys.clipping": (Object, "clipping"),
    "keys.preset": (Volume, "preset"),
    "keys.visibility": (Object, "visibility"),
}

# The keywords whose value is the key itself rather than a (key, default)
# pair, so that a computed one is seen here as a binding and not as markup.
BARE_BINDINGS = frozenset({"v_model", "v_show"})

COMPUTED_KEYS = {
    "visible_key": "the shared sheet dialog's v-model; both sheets are in the registry",
    "screenshot_viewport(key)": "Scene.screenshot_viewports, via the widget default",
    "capture_series_number(key)": "Scene.capture_series, one entry per viewport",
    "capture_series_description(key)": "Scene.capture_series, one entry per viewport",
    "key": "clip bounds, derived from each object's geometry",
    "keys.detail_panel": "whether an object's row is opened is browsing state",
    "variable": "the snap group and tile size loops, all named in the registry",
    "f'mpr_rotation_data.angles_list[{i}].angle'": "a step within the sequence",
    "f'mpr_rotation_data.angles_list[{i}].axis'": "a step within the sequence",
    "f'mpr_rotation_data.angles_list[{i}].name'": "a step within the sequence",
    "f'mpr_rotation_data.angles_list[{i}].visible'": "a step within the sequence",
    "f'volumetry_groups.{STRUCTURES}[{i}].chamber'": "a structure within the list",
    "f'volumetry_groups.{STRUCTURES}[{i}].density'": "a structure within the list",
    "f'volumetry_groups.{STRUCTURES}[{i}].labels'": "a structure within the list",
    "f'volumetry_groups.{STRUCTURES}[{i}].name'": "a structure within the list",
}


def _is_state(node) -> bool:
    """Whether ``node`` is the trame state, however it was reached."""
    return (isinstance(node, ast.Name) and node.id == "state") or (
        isinstance(node, ast.Attribute) and node.attr == "state"
    )


def _binding_target(node: ast.keyword):
    """The key a widget keyword binds, or None if it binds no state.

    A binding is either ``key=("name", default)`` or, for ``v_model`` alone, a
    bare expression, which ``v_model`` and ``v_show`` both take. Anything else
    -- a vue expression over state, a literal -- is not a declaration and is
    left to the other checks.
    """
    if isinstance(node.value, ast.Tuple) and node.value.elts:
        return node.value.elts[0]
    if node.arg in BARE_BINDINGS:
        return node.value
    return None


def _ui_bindings() -> tuple[set[str], set[str]]:
    """Every state key the UI binds, as (literal names, computed expressions).

    Covers both halves of how a control reaches state: a binding on a widget,
    and a direct write to ``server.state`` while building the page. A literal
    that is not an identifier is a vue expression rather than a key.
    """
    named, computed = set(), set()

    for path in sorted(UI_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword):
                target = _binding_target(node)
                if target is None:
                    continue
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    if target.value.isidentifier():
                        named.add(target.value)
                elif node.arg in BARE_BINDINGS:
                    computed.add(ast.unparse(target))
            elif isinstance(node, ast.Assign):
                for assigned in node.targets:
                    if isinstance(assigned, ast.Attribute) and _is_state(
                        assigned.value
                    ):
                        named.add(assigned.attr)

    return named, computed


def _source_keys() -> tuple[set[str], set[str]]:
    """Every literal key the source writes, and every one it listens to."""
    written, listened = set(), set()

    for path in sorted(SRC_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for assigned in node.targets:
                    if isinstance(assigned, ast.Attribute) and _is_state(
                        assigned.value
                    ):
                        written.add(assigned.attr)
                    elif (
                        isinstance(assigned, ast.Subscript)
                        and _is_state(assigned.value)
                        and isinstance(assigned.slice, ast.Constant)
                        and isinstance(assigned.slice.value, str)
                    ):
                        written.add(assigned.slice.value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "change"
                and _is_state(node.func.value)
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        listened.add(arg.value)

    return written, listened


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


def test_every_ui_key_is_declared():
    named, _ = _ui_bindings()
    undeclared = named - set(registry.VARIABLES)

    assert undeclared == set(), (
        "UI controls with no config decision -- give them a Scene field in "
        f"registry.DOCUMENT, or a reason in registry.SESSION: {sorted(undeclared)}"
    )


def test_every_key_the_source_writes_is_declared():
    written, _ = _source_keys()
    undeclared = written - set(registry.VARIABLES)

    assert undeclared == set(), (
        f"state written but not declared in cardio.registry: {sorted(undeclared)}"
    )


def test_every_key_the_source_listens_to_is_declared():
    """A listener on an undeclared key is a listener on nothing."""
    _, listened = _source_keys()
    undeclared = listened - set(registry.VARIABLES)

    assert undeclared == set(), (
        f"listened to but not declared in cardio.registry: {sorted(undeclared)}"
    )


def _seeded_keys() -> set[str]:
    """Every key a controller claims to write from the scene.

    The seeding pass writes through a variable rather than by name, so the
    scan above cannot see those writes. What names them is the ``seeds``
    tuple, which is where a controller says which keys are its to write.
    """
    return {
        key for controller in Controller.__subclasses__() for key in controller.seeds
    }


def test_nothing_is_declared_that_the_app_no_longer_uses():
    named, _ = _ui_bindings()
    written, _ = _source_keys()
    stale = set(registry.VARIABLES) - named - written - _seeded_keys()

    assert stale == set(), (
        f"declared but neither bound, written nor seeded: {sorted(stale)}"
    )


def test_every_computed_binding_is_accounted_for():
    _, computed = _ui_bindings()
    declared = set(PER_OBJECT) | set(COMPUTED_KEYS)

    assert computed == declared, (
        "computed bindings differ from those declared; a new one needs a "
        f"config decision too. Missing: {sorted(computed - declared)}. "
        f"Stale: {sorted(declared - computed)}"
    )


@pytest.mark.parametrize("key", registry.keys_in_scope(registry.Scope.DOCUMENT))
def test_document_keys_name_a_real_scene_field(key):
    assert _resolve(Scene, registry.source_of(key)) is not None


def test_no_two_document_keys_name_the_same_field():
    """One key to one field, so that seeding and saving cannot collide.

    Both directions read this table, so an entry pointing at another key's
    field is self-consistent and nothing downstream would notice -- except
    that the two keys would then write over each other on the way out. Two
    sources may nest, which is a mirror and is allowed; they may not be the
    same.
    """
    sources = {
        key: registry.source_of(key)
        for key in registry.keys_in_scope(registry.Scope.DOCUMENT)
    }
    shared = {
        source: sorted(k for k, s in sources.items() if s == source)
        for source in set(sources.values())
    }
    assert not [names for names in shared.values() if len(names) > 1], (
        f"more than one key names the same field: "
        f"{[names for names in shared.values() if len(names) > 1]}"
    )


@pytest.mark.parametrize("key", registry.keys_in_scope(registry.Scope.DOCUMENT))
def test_every_document_key_can_be_read_off_a_default_scene(key):
    """A dotted path walks models that always exist, so it cannot come up short.

    Not a value assertion -- an empty scene has nothing configured to check
    against. What it says is that the walk itself reaches a leaf.
    """
    registry.state_value(Scene(), key)


@pytest.mark.parametrize("layout", list(view.Layout))
def test_the_two_layout_conversions_are_inverses(layout):
    """The one key spelled differently in a config and in state.

    ``to_state`` takes the field and ``to_config`` takes the state value, so
    they compose to the identity without looking symmetric. This is what says
    they still do.
    """
    assert (
        registry.to_config(
            "maximized_view", registry.to_state("maximized_view", layout)
        )
        == layout.value
    )


def test_the_quad_layout_is_why_that_pair_exists():
    """Guarding the guard above, which every other member would pass anyway."""
    assert registry.to_state("maximized_view", view.Layout.QUAD) == ""
    assert registry.to_state("maximized_view", view.Layout.UL) == "ul"


@pytest.mark.parametrize(
    "key", sorted(k for k, v in registry.VARIABLES.items() if v.seeded_by)
)
def test_a_key_its_controller_writes_says_why(key):
    """The exceptions to the seeding pass are declared, not discovered.

    A hardcoded skip list inside the seeder would say which keys it passes
    over; this says why, next to the rule each one is an exception to.
    """
    variable = registry.VARIABLES[key]
    assert variable.scope is registry.Scope.DOCUMENT
    assert variable.source and variable.seeded_by.strip()


@pytest.mark.parametrize("expression,target", sorted(PER_OBJECT.items()))
def test_per_object_keys_name_a_real_object_field(expression, target):
    model, prop = target
    field = registry.OBJECT_SOURCES[prop]
    assert field in model.model_fields, f"{model.__name__} has no field '{field}'"


def test_object_sources_name_real_object_state_keys():
    """A source for a key ObjectState does not spell configures nothing."""
    from cardio.state import ObjectState

    for prop in registry.OBJECT_SOURCES:
        assert isinstance(getattr(ObjectState, prop, None), property), (
            f"ObjectState has no '{prop}' key"
        )


def test_session_and_items_keys_carry_a_reason():
    for scope in (registry.Scope.SESSION, registry.Scope.ITEMS):
        for key in registry.keys_in_scope(scope):
            assert registry.VARIABLES[key].reason.strip()


# Names the UI reaches for on trame's controller that are not actions: the
# render views' own methods, which ui/layout.py assigns as it builds them, and
# the two lifecycle hooks.
NOT_ACTIONS = {*view.VIEW_FUNCTIONS, "finalize_mpr_initialization", "on_server_ready"}


def _controller_references() -> set[str]:
    """Every ``…controller.X`` the UI names."""
    found = set()
    for path in sorted(UI_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "controller"
            ):
                found.add(node.attr)
    return found


def _declared_action_names() -> list[str]:
    """Every action name the controller classes declare, one entry per method."""
    classes = [
        value
        for value in vars(logic).values()
        if isinstance(value, type) and issubclass(value, logic.Controller)
    ]
    return [name for cls in classes for _, name in declared_actions(cls)]


def test_every_controller_call_the_ui_makes_is_an_action():
    """A renamed action would otherwise leave a button calling nothing.

    Trame's controller answers any attribute with a callable that does nothing
    when nothing is registered, so a stale name on a ``click=`` is silent both
    at build time and at click time.
    """
    unknown = _controller_references() - set(_declared_action_names()) - NOT_ACTIONS

    assert unknown == set(), (
        f"the UI calls controller functions that no action declares: {sorted(unknown)}"
    )


def test_every_declared_action_name_is_unique():
    """Two controllers claiming one name would have one silently win."""
    declared = _declared_action_names()
    duplicates = sorted({name for name in declared if declared.count(name) > 1})

    assert duplicates == [], f"actions declared more than once: {duplicates}"


def test_registering_writes_no_state():
    """The two passes stay two.

    ``register`` declares listeners and controller functions; ``seed`` writes
    the state. A write that creeps back into ``register`` is state the app
    would set before the pass that is supposed to set all of it, which is what
    ``Logic.apply_scene`` exists to rule out -- and it would not fail anywhere
    else, because at startup the two run one after the other regardless.
    """
    offenders = []

    for path in sorted(LOGIC_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "register"):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Assign):
                    continue
                for assigned in inner.targets:
                    written = isinstance(assigned, (ast.Attribute, ast.Subscript))
                    if written and _is_state(assigned.value):
                        offenders.append(
                            f"{path.name}:{inner.lineno} {ast.unparse(assigned)}"
                        )

    assert offenders == [], (
        f"state written from register(); it belongs in seed(): {offenders}"
    )


def test_the_scan_finds_the_ui():
    """Guards the guard: a broken walk would pass everything vacuously."""
    named, computed = _ui_bindings()
    written, listened = _source_keys()
    assert len(named) > 20 and len(computed) > 5
    assert len(written) > 20 and len(listened) > 20
    assert len(_seeded_keys()) > 20


def test_the_drawer_sections_the_config_names_are_the_ones_the_ui_builds():
    """A renamed section would otherwise open nothing, silently.

    ``DrawerSection`` is what a config may name; the ``section(...)`` calls are
    what the accordion actually tracks. They have to be the same set.
    """
    source = (UI_DIR / "__init__.py").read_text()
    built = set(re.findall(r'section\(\s*"([a-z]+)"', source))

    assert built == {member.value for member in DrawerSection}
