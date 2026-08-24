"""Test the cine playback loop: pause responsiveness and render economy.

The loop is asynchronous and timing-dependent, so these run on the virtual
clock in ``playback_harness``: sleeps and renders advance a counter rather than
the wall, and the sleep shim caps how many times the loop may await, so a loop
that will not stop fails instead of hanging the suite.
"""

# System
import asyncio

# Third Party
import pytest

# Internal
from tests.playback_harness import (
    CHECK_INTERVAL,
    Clock,
    PlaybackApp,
    drain,
    install_shims,
    pump,
)


@pytest.fixture
def clock() -> Clock:
    return Clock()


def play(app):
    """The client ticking the play box: set, then flush as trame would."""
    app.state.playing = True
    app.state.flush()
    return app.playback._playback_task


def pause(app):
    app.state.playing = False
    app.state.flush()


def live_play_loops() -> list:
    return [
        task
        for task in asyncio.all_tasks()
        if "_play_loop" in str(task.get_coro()) and not task.done()
    ]


# --- the target frame is derived from the clock, not accumulated -------------


def test_target_frame_advances_through_the_cycle(clock):
    app = PlaybackApp(clock, nframes=10)
    calculate = app.playback._calculate_target_frame

    assert calculate(0.0, 60, 10) == 0
    assert calculate(0.35, 60, 10) == 3
    assert calculate(0.95, 60, 10) == 9


def test_target_frame_wraps_at_the_cycle_boundary(clock):
    app = PlaybackApp(clock, nframes=10)

    assert app.playback._calculate_target_frame(1.0, 60, 10) == 0
    assert app.playback._calculate_target_frame(1.25, 60, 10) == 2


def test_lag_does_not_accumulate_across_cycles(clock):
    """The frame comes from absolute elapsed time, so a loop that falls behind
    skips ahead rather than replaying the backlog."""
    app = PlaybackApp(clock, nframes=10)

    assert app.playback._calculate_target_frame(2.55, 60, 10) == 5
    assert app.playback._calculate_target_frame(60.55, 60, 10) == 5


def test_a_bpm_of_zero_divides_by_zero(clock):
    """Documents an unguarded division. The speed slider bottoms out at 20, so
    nothing in the UI can reach it -- but nothing in the loop rejects it."""
    app = PlaybackApp(clock, nframes=10)

    with pytest.raises(ZeroDivisionError):
        app.playback._calculate_target_frame(1.0, 0, 10)


# --- pausing stops the loop --------------------------------------------------


def test_pausing_stops_the_loop(clock, monkeypatch):
    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10, bpm=60)
        task = play(app)
        await pump(lambda: len(app.renders) >= 6)

        rendered_before_pause = len(app.renders)
        pause(app)
        await drain(task)

        return app, rendered_before_pause

    app, before = asyncio.run(scenario())
    assert before > 0
    assert len(app.renders) == before, "the loop rendered after being paused"


def test_pausing_stops_the_loop_promptly(clock, monkeypatch):
    """The whole point of sleeping in CHECK_INTERVAL chunks: no more than one
    chunk of virtual time may pass between the pause and the loop exiting."""

    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10, bpm=60)
        task = play(app)
        await pump(lambda: len(app.renders) >= 6)

        paused_at = clock.now
        pause(app)
        await drain(task)
        return clock.now - paused_at

    assert asyncio.run(scenario()) <= CHECK_INTERVAL


def test_a_fast_speed_still_pauses_promptly(clock, monkeypatch):
    """The reported runaway is worst at high speed, where the loop is behind
    schedule on every iteration and never gets to sleep its full interval."""

    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=25, bpm=120, render_seconds=0.05)
        task = play(app)
        await pump(lambda: len(app.renders) >= 10)

        rendered_before_pause = len(app.renders)
        pause(app)
        await drain(task)
        return app, rendered_before_pause

    app, before = asyncio.run(scenario())
    assert len(app.renders) == before


def test_cancelling_the_task_stops_the_loop(clock, monkeypatch):
    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10)
        task = play(app)
        await pump(lambda: len(app.renders) >= 3)

        task.cancel()
        await drain(task)
        return app, task

    app, task = asyncio.run(scenario())
    assert task.done()
    assert app.playback._is_rendering is False


def test_toggling_play_leaves_exactly_one_loop_running(clock, monkeypatch):
    """Each play must cancel the previous task rather than stacking onto it;
    two live loops would render twice per frame and outlive one pause."""

    async def scenario():
        install_shims(monkeypatch, clock, budget=20000)
        app = PlaybackApp(clock, nframes=10)

        for _ in range(4):
            play(app)
            await pump(lambda: False, budget=3)
            pause(app)
            await pump(lambda: False, budget=3)

        task = play(app)
        await pump(lambda: len(app.renders) >= 3)
        live = len(live_play_loops())

        pause(app)
        await drain(task)
        return live

    assert asyncio.run(scenario()) == 1


def test_the_loop_refuses_to_start_with_nothing_to_animate(clock, monkeypatch):
    async def scenario():
        install_shims(monkeypatch, clock, budget=100)
        app = PlaybackApp(clock, incrementing=False, rotating=False)
        task = play(app)
        await drain(task)
        return app

    app = asyncio.run(scenario())
    assert app.renders == []
    assert app.state.playing is False


def test_rotation_alone_keeps_the_loop_running(clock, monkeypatch):
    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10, incrementing=False, rotating=True)
        task = play(app)
        await pump(lambda: len(app.renders) >= 5)
        pause(app)
        await drain(task)
        return app

    app = asyncio.run(scenario())
    assert app.scene.renderer.camera.azimuths
    assert app.mpr.frames == [], "no frame stepping was requested"
    assert len(app.renders) == len(app.scene.renderer.camera.azimuths), (
        "a camera rotation has no state listener to render it, so the loop "
        "must render it directly"
    )


# --- render economy ----------------------------------------------------------


def test_frames_are_skipped_rather_than_queued_when_renders_are_slow(
    clock, monkeypatch
):
    """A render far slower than the frame interval must cost frames, not time:
    the frame shown has to track the clock, not lag further behind on every
    iteration."""

    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        # 20ms of frame budget against a 50ms render
        app = PlaybackApp(clock, nframes=25, bpm=120, render_seconds=0.05)
        task = play(app)
        await pump(lambda: len(app.mpr.frames) >= 5)
        pause(app)
        await drain(task)
        return app

    app = asyncio.run(scenario())
    shown = app.mpr.frames
    assert shown != list(range(1, len(shown) + 1)), "frames were not skipped"
    assert len(set(shown)) == len(shown), "a frame was rendered twice"


def test_stepping_is_inert_while_playing(clock):
    """increment_frame guards on `playing`, so the cine capture loop -- which
    calls it per frame -- silently does nothing if started during playback."""
    app = PlaybackApp(clock, nframes=10, playing=True)

    app.playback.increment_frame()
    app.playback.decrement_frame()

    assert app.state.frame == 0
    assert app.renders == []


def test_a_frame_step_renders_once(clock, monkeypatch):
    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10, bpm=60)
        task = play(app)
        await pump(lambda: len(app.mpr.frames) >= 4)
        pause(app)
        await drain(task)
        return app

    app = asyncio.run(scenario())
    assert len(app.renders) == len(app.mpr.frames)


def test_every_render_shows_the_frame_that_was_just_set(clock, monkeypatch):
    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10, bpm=60)
        task = play(app)
        await pump(lambda: len(app.mpr.frames) >= 4)
        pause(app)
        await drain(task)
        return app

    app = asyncio.run(scenario())
    assert all(render["shown"] == render["frame"] for render in app.renders)


def test_the_render_guard_is_held_for_the_whole_step(clock, monkeypatch):
    """``_is_rendering`` is set and cleared with no await in between, so no
    other coroutine can ever observe it set -- it guards nothing. Pinned here
    because the flush render now happens under it, and a future change that
    moved an await inside would start dropping frames."""

    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10, bpm=60)
        task = play(app)
        await pump(lambda: len(app.mpr.frames) >= 3)
        pause(app)
        await drain(task)
        return app

    app = asyncio.run(scenario())
    assert all(render["rendering"] for render in app.renders)
    assert len(app.renders) == len(app.mpr.frames)


def test_stepping_and_rotating_together_still_render_once(clock, monkeypatch):
    """Both modes at once is the branch where the two render paths could
    double up: the camera moves inside the same state block whose flush
    renders the new frame, so it rides along on that one render."""

    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10, bpm=60, incrementing=True, rotating=True)
        task = play(app)
        await pump(lambda: len(app.mpr.frames) >= 4)
        pause(app)
        await drain(task)
        return app

    app = asyncio.run(scenario())
    azimuths = app.scene.renderer.camera.azimuths

    assert len(app.renders) == len(azimuths), "one render per rotated step"
    assert all(render["shown"] == render["frame"] for render in app.renders)


def test_scrubbing_onto_the_next_frame_does_not_re_render_it(clock, monkeypatch):
    """The slider's own listener already rendered that frame, so the loop's
    write changes nothing and must not push a duplicate image."""

    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10, bpm=60, rotating=False)
        task = play(app)
        await pump(lambda: len(app.mpr.frames) >= 2)

        next_target = app.playback._last_target_frame + 1
        app.state.frame = next_target
        app.state.flush()
        after_scrub = len(app.renders)

        # let the loop run the iteration whose target is the scrubbed frame
        await pump(lambda: app.playback._last_target_frame == next_target)
        rendered_by_the_loop = len(app.renders) - after_scrub

        pause(app)
        await drain(task)
        return rendered_by_the_loop

    assert asyncio.run(scenario()) == 0


def test_scrubbing_does_not_swallow_the_camera_rotation(clock, monkeypatch):
    """Same no-op frame write, but with rotation on the camera has moved, and
    nothing else will render it."""

    async def scenario():
        install_shims(monkeypatch, clock, budget=5000)
        app = PlaybackApp(clock, nframes=10, bpm=60, rotating=True)
        task = play(app)
        await pump(lambda: len(app.mpr.frames) >= 2)

        next_target = app.playback._last_target_frame + 1
        app.state.frame = next_target
        app.state.flush()
        after_scrub = len(app.renders)

        await pump(lambda: app.playback._last_target_frame == next_target)
        rendered_by_the_loop = len(app.renders) - after_scrub

        pause(app)
        await drain(task)
        return rendered_by_the_loop

    assert asyncio.run(scenario()) >= 1
