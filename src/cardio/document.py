"""The document read back out of state, as a scene that can reopen it.

A config goes one way in: a ``Scene`` is seeded into state, key by key, and
from then on state is what the app is showing. Saving is that walk run
backwards over the same map, which is what keeps the two from drifting into
describing different apps -- and what makes the round trip a thing a test can
check rather than a thing to be careful about.

What is *not* in state, and never was, is which files to read. That is carried
over from the scene the session was built on rather than reconstructed from the
objects it loaded.
"""

# Internal
from . import registry, toml
from .scene import Scene
from .state import ObjectState


def _place(data: dict, path: str, value) -> None:
    """Write ``value`` at the dotted ``path``, making the branches it needs."""
    *branches, leaf = path.split(".")
    for branch in branches:
        data = data.setdefault(branch, {})
    data[leaf] = value


def _mirrored(sources: list[str]) -> set[str]:
    """The paths lying inside another one, which writing it already covers.

    ``angle_units`` is a field of the rotation sequence that ``mpr_rotation_data``
    writes whole. Both are document state -- an undo restores each -- but only
    the whole one belongs in a config, and which of two overlapping writes
    landed last is not something to leave to the order the keys sort in.
    """
    return {
        path
        for path in sources
        if any(path.startswith(f"{other}.") for other in sources)
    }


def _entries(scene: Scene, data: dict):
    """Each renderable beside its own entry in the dumped scene."""
    for group in ("meshes", "volumes", "segmentations"):
        yield from zip(getattr(scene, group), data.get(group, []))


def scene_from_state(state, scene: Scene) -> Scene:
    """The scene as it now stands: what was loaded, showing what is shown."""
    data = scene.model_dump(mode="json", exclude_none=True)

    keys = registry.keys_in_scope(registry.Scope.DOCUMENT)
    sources = [registry.source_of(key) for key in keys]
    mirrors = _mirrored(sources)

    for key, source in zip(keys, sources):
        if source not in mirrors:
            _place(data, source, registry.to_config(key, state[key]))

    data[registry.VIEWPORT_TICKS] = registry.ticked_viewports(state)

    for obj, entry in _entries(scene, data):
        keys = ObjectState.of(obj)
        for prop, field in registry.OBJECT_SOURCES.items():
            if field in type(obj).model_fields:
                entry[field] = state[getattr(keys, prop)]

        # The three range sliders are one field: a crop is a box, and the axes
        # are only split up because that is how it is dragged.
        entry["crop"] = [bound for key in keys.clip_bounds for bound in state[key]]

    return Scene(**data)


def to_toml(scene: Scene) -> str:
    """``scene`` as a config file -- the same one ``--config`` reads.

    A config says where the rotations come from either by naming a file or by
    spelling the sequence, and never by doing both: ``load_rotation_file``
    reads the file over whatever the config said, so a config carrying both
    would show a sequence that opening it would throw away.

    Which one it is, is whichever the scene was opened with. A named file goes
    on being where the rotations come from, so what a person changed in the app
    is not in the saved config -- it is saved by Save Rotations, as a rotation
    file, which is the format that holds rotations and the one to point at
    next.
    """
    data = scene.model_dump(mode="json", exclude_none=True)
    if scene.mpr_rotation_file is not None:
        data.pop("mpr_rotation_sequence", None)
    return toml.dumps(data)
