"""Test the encode-quality lever the playback slider drives.

The quality is set through trame's render helper, which only exists once a
client has connected, so the module has to be inert before that rather than
raising into the state flush that called it.
"""

# Third Party
import pytest

# Internal
import cardio.image_quality as image_quality
from cardio.image_quality import (
    DEFAULT_PLAYBACK_QUALITY,
    FULL_QUALITY,
    render_windows,
    set_image_quality,
)
from cardio.mpr_views import MPRViews
from cardio.reslice import VIEWS


class FakeScene:
    def __init__(self, mpr_views=None):
        self.renderWindow = "vr-window"
        self.mpr_views = mpr_views


class FakeServer:
    def __init__(self, protocol="connected"):
        self.name = "quality-test"
        self.protocol = protocol


class RecordingHelper:
    def __init__(self):
        self.calls = []

    def set_image_quality(self, window, quality, ratio):
        self.calls.append((window, quality, ratio))


@pytest.fixture
def helper(monkeypatch) -> RecordingHelper:
    recorder = RecordingHelper()
    monkeypatch.setattr(image_quality.tvtk, "get_helper", lambda server: recorder)
    return recorder


# --- which windows the quality applies to ------------------------------------


def test_only_the_volume_window_before_the_mpr_views_exist():
    assert render_windows(FakeScene()) == ["vr-window"]


def test_every_mpr_window_is_included():
    scene = FakeScene(MPRViews())

    windows = render_windows(scene)

    assert windows[0] == "vr-window"
    assert len(windows) == 1 + len(VIEWS)
    assert set(windows[1:]) == set(scene.mpr_views.windows.values())


# --- setting it --------------------------------------------------------------


def test_the_quality_reaches_every_window(helper):
    scene = FakeScene(MPRViews())

    assert set_image_quality(FakeServer(), scene, 40) is True
    assert [call[1] for call in helper.calls] == [40] * (1 + len(VIEWS))


def test_the_resolution_is_left_alone(helper):
    """Ratio 1 keeps set_view_quality from resizing the render window."""
    set_image_quality(FakeServer(), FakeScene(), 40)

    assert helper.calls[0][2] == 1


def test_nothing_happens_before_a_client_connects(helper):
    assert set_image_quality(FakeServer(protocol=None), FakeScene(), 40) is False
    assert helper.calls == []


def test_nothing_happens_without_a_render_helper(monkeypatch):
    monkeypatch.setattr(image_quality.tvtk, "get_helper", lambda server: None)

    assert set_image_quality(FakeServer(), FakeScene(), 40) is False


def test_the_playback_default_is_below_full_quality():
    assert 0 < DEFAULT_PLAYBACK_QUALITY < FULL_QUALITY
