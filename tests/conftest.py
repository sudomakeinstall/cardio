"""Fixtures shared across the suite."""

# System
import gc
import pathlib as pl

# Third Party
import pytest
import tomlkit as tk
import trame.app.core as tm_core
import trame_vtk.modules.vtk as tm_vtk

ASSETS = pl.Path(__file__).parent / "assets"

# Both are keyed by server name, and neither is ever emptied: trame's registry
# of servers, and trame_vtk's of the helper it builds for each one.
REGISTRIES = (tm_core.AVAILABLE_SERVERS, tm_vtk.HELPERS_PER_SERVER)


@pytest.fixture(scope="session")
def asset():
    """Read one of the TOML fixtures in tests/assets, by file name."""

    def read(name: str):
        with (ASSETS / name).open("rt", encoding="utf-8") as fp:
            return tk.load(fp)

    return read


def held_servers() -> set[str]:
    """The names trame and trame_vtk are currently holding a server under."""
    return set().union(*(set(registry) for registry in REGISTRIES))


def release_servers(keep: set[str]) -> set[str]:
    """Drop every held server whose name is not in ``keep``, and say which.

    A held server holds the Logic, UI and Scene built on it, and through the
    scene every render window and image those own. Nothing in trame ever drops
    one, and the cycle runs through VTK, so the collector cannot break it
    either: a suite that names a server per test keeps every app it has ever
    built resident, which is what walked this one into the OOM killer.
    """
    released = held_servers() - keep
    for registry in REGISTRIES:
        for name in released:
            registry.pop(name, None)

    if released:
        gc.collect()

    return released


@pytest.fixture(autouse=True)
def release_test_servers():
    """Give back whatever servers a test built, as it ends."""
    keep = held_servers()
    yield
    release_servers(keep)
