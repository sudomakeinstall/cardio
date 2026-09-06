"""What the seeding pass writes, against what the registry says it should.

The registry names the ``Scene`` field behind every document key, and until
now that name was read in one direction only -- to save a session. The way in
was hand-written, so the two could disagree about which field a key comes from
and nothing anywhere would notice.

What is checked here is the forward direction: run the pass, watch what each
controller writes, and compare the first write of every document key against
what the registry says the scene holds. A key whose source points at the wrong
field fails here.

A default scene would prove nothing -- a source moved from ``playback.bpm`` to
``playback.bpr`` reads the same default either way. So the scenes below move
every document field off its default, to values no two of which are alike.
"""

# System
import pathlib as pl

# Third Party
import pytest

# Internal
import cardio.registry as registry
from cardio.document import _place
from cardio.scene import Scene
from tests.test_app_smoke import build_app, build_scene, write_volume

# Document keys their own controller writes rather than the registry, and the
# checkpoint that will change that. Both locks are forced off unless a
# segmentation is actually selected, which is a decision about snapping rather
# than about the field.
WRITTEN_BY_HAND = {"snap_locked", "snap_orientation_locked"}

# Every document field, moved off its default. Keyed by the dotted source
# rather than the state key, because that is what a scene is built from.
MOVED = {
    "active_volume_label": "vol",
    "capture_format": "jpeg",
    "current_frame": 2,
    "mpr_level": 111.0,
    "mpr_origin": [1.0, 2.0, 3.0],
    "mpr_rotation_sequence": {
        "metadata": {"angle_units": "degrees", "index_order": "roma"},
        "angles_list": [{"name": "tilt", "axis": "X", "angle": 30.0}],
    },
    "mpr_segmentation_opacity": 0.25,
    "mpr_window": 222.0,
    "mpr_window_level_preset": None,
    "playback.bpm": 71,
    "playback.bpr": 5,
    "playback.quality": 44,
    "playback.resolution": 88,
    "snap.labels_a": [1],
    "snap.labels_b": [2],
    "snap.labels_c": [3],
    "snap.mode": "traverse",
    "snap.segmentation_label": "seg",
    "snap.traverse": 33,
    "tile_cols": 4,
    "tile_rows": 6,
    "view.camera_lock": "LL",
    "view.drawer_sections": ["orientation", "export"],
    "view.layout": "axial",
    "view.theme": "light",
}


def boolean_sources() -> list[str]:
    """The document sources holding a bool.

    They are the ones distinct values cannot separate: there are only two, and
    seven keys wanting them. ``moved_scene`` gives each a different pattern
    across the rounds instead.
    """
    default = Scene()
    return sorted(
        registry.source_of(key)
        for key in registry.keys_in_scope(registry.Scope.DOCUMENT)
        if isinstance(registry.read(default, registry.source_of(key)), bool)
    )


# Enough rounds to give every boolean source its own pattern: with seven of
# them, three rounds is eight patterns, and no two keys can share one.
ROUNDS = 3

# Enough frames that current_frame has somewhere to be other than the start.
FRAMES = 3


def moved_scene(directory: pl.Path, round: int) -> Scene:
    """A scene with every document field away from its default.

    ``_place`` is the same dotted-path writer the save direction uses, so the
    scene is built by the same walk the registry is read by.

    The volume is given three frames because ``current_frame`` is taken modulo
    the number there are: on a still scene it can only ever be zero, which is
    where it starts.
    """
    for frame in range(FRAMES):
        write_volume(directory / f"vol{frame}.nii.gz")

    data: dict = {
        "volumes": [
            {
                "label": "vol",
                "directory": directory,
                "file_paths": [f"vol{frame}.nii.gz" for frame in range(FRAMES)],
            }
        ]
    }
    for source, value in MOVED.items():
        _place(data, source, value)
    for index, source in enumerate(boolean_sources()):
        _place(data, source, bool((index + 1) >> round & 1))
    return build_scene(directory, **data)


class Recording:
    """A trame state that remembers what is written to it, and forwards it on.

    Watching rather than perturbing: the pass runs exactly as it always does,
    reading real values from siblings that have already seeded, and what is
    recorded is what actually reached the state.
    """

    def __init__(self, state):
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "writes", [])

    def __getattr__(self, name):
        return getattr(self._state, name)

    def __setattr__(self, name, value):
        self.writes.append((name, value))
        setattr(self._state, name, value)

    def __getitem__(self, key):
        return self._state[key]

    def __setitem__(self, key, value):
        self.writes.append((key, value))
        self._state[key] = value

    def __enter__(self):
        return self._state.__enter__()

    def __exit__(self, *exc_info):
        return self._state.__exit__(*exc_info)


class WatchedServer:
    """The real server, handing out the recording state instead of its own."""

    def __init__(self, server, state):
        self._server = server
        self.state = state

    def __getattr__(self, name):
        return getattr(self._server, name)


def observe(logic) -> dict[str, list]:
    """Every controller's writes during one seeding pass, in order.

    ``key -> [(controller name, value), ...]``, so that both what was written
    and who wrote it can be asked about.
    """
    recording = Recording(logic.server.state)
    watched = WatchedServer(logic.server, recording)

    controllers = logic.controllers
    originals = [controller.server for controller in controllers]
    for controller in controllers:
        controller.server = watched

    seen: dict[str, list] = {}
    try:
        for controller in controllers:
            recording.writes.clear()
            controller.seed()
            for key, value in recording.writes:
                seen.setdefault(key, []).append((type(controller).__name__, value))
    finally:
        for controller, original in zip(controllers, originals):
            controller.server = original

    return seen


@pytest.fixture(scope="module")
def rounds(tmp_path_factory):
    """One observed pass per boolean pattern, over a fully moved scene."""
    observed = []
    for round in range(ROUNDS):
        directory = tmp_path_factory.mktemp(f"seeding{round}")
        scene = moved_scene(directory, round)
        _, _, logic, _ = build_app(scene)
        observed.append((scene, observe(logic)))
    return observed


def test_every_document_field_is_accounted_for_by_the_fixture(rounds):
    """Guarding the guard: a field left at its default proves nothing.

    A document key added later would otherwise be checked against a scene
    that never moved it, which is a test that cannot fail. Every source is
    either moved by name, moved inside something moved by name, given a
    pattern because it is a bool, or the one the cameras mirror -- which holds
    no configured pose to move.
    """
    scene, _ = rounds[0]
    default = Scene()
    booleans = set(boolean_sources())

    for key in registry.keys_in_scope(registry.Scope.DOCUMENT):
        source = registry.source_of(key)
        if source in booleans or source == "view.cameras":
            continue

        moved = source in MOVED or any(
            source.startswith(f"{other}.") for other in MOVED
        )
        assert moved, f"{source} is not moved off its default anywhere"
        assert registry.read(scene, source) != registry.read(default, source), (
            f"{source} was asked to move and did not"
        )


def test_no_two_document_fields_hold_the_same_value(rounds):
    """A source pointing at the wrong field has to read something different.

    The booleans cannot manage that on their own -- there are seven of them
    and two values -- so each gets its own pattern across the rounds, and it
    is the patterns that have to differ.
    """
    scene, _ = rounds[0]
    booleans = set(boolean_sources())

    values = {
        source: repr(registry.read(scene, source))
        for source in (
            registry.source_of(key)
            for key in registry.keys_in_scope(registry.Scope.DOCUMENT)
        )
        if source not in booleans
    }
    assert len(set(values.values())) == len(values), (
        f"two document fields read alike: {sorted(values)}"
    )

    patterns = {
        source: tuple(registry.read(scene, source) for scene, _ in rounds)
        for source in booleans
    }
    assert len(set(patterns.values())) == len(patterns), (
        f"two boolean fields share a pattern: {patterns}"
    )


def test_every_document_key_is_written_by_the_pass(rounds):
    for _, seen in rounds:
        missing = [
            key
            for key in registry.keys_in_scope(registry.Scope.DOCUMENT)
            if key not in seen
        ]
        assert not missing, f"the seeding pass never writes {missing}"


def test_the_pass_writes_what_the_registry_says_it_should(rounds):
    """The forward direction, key by key, against the table.

    The first write is the seeded one: anything after it is a side effect
    acting on what was seeded, which is why a lock moves the origin.
    """
    for scene, seen in rounds:
        for key in registry.keys_in_scope(registry.Scope.DOCUMENT):
            if registry.VARIABLES[key].seeded_by or key in WRITTEN_BY_HAND:
                continue

            _, written = seen[key][0]
            expected = registry.state_value(scene, key)
            assert repr(written) == repr(expected), (
                f"{key} was seeded {written!r}, but its source "
                f"{registry.source_of(key)} says {expected!r}"
            )


def test_the_observation_sees_the_whole_pass(rounds):
    """Guarding the guard: an observer that saw nothing would pass everything."""
    for _, seen in rounds:
        document = set(registry.keys_in_scope(registry.Scope.DOCUMENT))
        assert len(document & set(seen)) >= 30
