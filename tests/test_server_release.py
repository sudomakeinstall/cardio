"""The suite gives back the app each test builds.

Nothing in trame drops a server, and a server holds the whole app built on it.
Without the release fixture in ``conftest`` every app the suite has ever built
stays resident, which is a scene, its render windows and its images per test.
"""

# System
import gc
import weakref

# Third Party
import trame.app

# Internal
from cardio.logic import Logic
from cardio.ui import UI
from tests.conftest import held_servers, release_servers
from tests.test_app_smoke import build_scene


def test_release_drops_only_the_servers_built_after_the_snapshot():
    keep = held_servers()
    trame.app.get_server("release-probe", client_type="vue3")

    assert release_servers(keep) == {"release-probe"}
    assert "release-probe" not in held_servers()


def test_release_keeps_the_servers_named_in_the_snapshot():
    trame.app.get_server("release-kept", client_type="vue3")
    keep = held_servers()

    assert release_servers(keep) == set()
    assert "release-kept" in held_servers()

    release_servers(keep - {"release-kept"})


def test_a_released_app_does_not_outlive_the_next_one(tmp_path):
    """The whole point: an app goes away with the server it was built on.

    One app stays resident -- the widgets keep a reference to the tree they
    built last -- so the property to hold is that the count does not grow, not
    that a build is gone the instant it is released.
    """
    refs = []
    for index in range(3):
        directory = tmp_path / f"app{index}"
        directory.mkdir()

        keep = held_servers()
        scene = build_scene(directory)
        server = trame.app.get_server(f"release-app-{index}", client_type="vue3")
        UI(server, scene, Logic(server, scene))
        refs.append(weakref.ref(scene))

        del scene, server
        release_servers(keep)

    gc.collect()

    assert [ref() is None for ref in refs] == [True, True, False]
