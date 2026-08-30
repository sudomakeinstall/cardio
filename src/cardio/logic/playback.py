"""Frame stepping and the cine playback loop."""

# System
import asyncio
import time

# Third Party
from trame.app import asynchronous

# Internal
from ..image_quality import (
    DEFAULT_PLAYBACK_QUALITY,
    DEFAULT_PLAYBACK_RESOLUTION,
    FULL_QUALITY,
    FULL_RESOLUTION,
    ratio_from_percent,
    set_image_quality,
)
from ..state import ObjectState
from .base import Controller


class PlaybackController(Controller):
    """Advances the frame, on demand or on a timer."""

    def __init__(self, app):
        super().__init__(app)
        self._playback_start_time = None
        self._last_render_duration = 0.0
        self._last_target_frame = None
        self._is_rendering = False
        self._playback_task = None

    def register(self):
        state = self.server.state
        state.change("frame")(self.update_frame)
        state.change("playing")(self._handle_playing_change)
        state.change("playback_quality", "playback_resolution")(
            self.sync_playback_image
        )

        controller = self.server.controller
        controller.increment_frame = self.increment_frame
        controller.decrement_frame = self.decrement_frame
        controller.reset_all = self.reset_all
        controller.close_application = self.close_application

    def update_frame(self, frame, **kwargs):
        # Before update_mpr_frame reads mpr_origin, so the new frame is
        # positioned on its own centroid rather than the previous frame's.
        self.app.snap.apply_frame_lock(frame)

        self.scene.hide_all_frames()

        for obj in self.scene.renderables:
            actor = obj.frame_actor(frame)
            if actor is not None and self.server.state[ObjectState.of(obj).visibility]:
                actor.SetVisibility(True)

        # Update MPR views if MPR is enabled
        self.app.mpr.update_mpr_frame(frame)
        self.app.tiles.update_tiles(frame)

        self.server.controller.view_update()

    def _handle_playing_change(self, playing, **kwargs):
        """Handle playback state changes with task cancellation support."""

        # Cancel existing playback task if any
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            self._playback_task = None

        # Start new playback task if playing
        if playing:
            self._apply_playback_image()
            self._playback_task = asynchronous.create_task(
                self._play_loop(playing, **kwargs)
            )
        else:
            self._restore_full_image()

    def _apply_playback_image(self):
        """Hand the views the quality and size the sliders ask for."""
        state = self.server.state
        set_image_quality(
            self.server,
            self.scene,
            state.playback_quality,
            ratio_from_percent(state.playback_resolution),
        )

    def _restore_full_image(self):
        """A view being inspected is never the reduced one."""
        set_image_quality(
            self.server,
            self.scene,
            FULL_QUALITY,
            ratio_from_percent(FULL_RESOLUTION),
        )

    def sync_playback_image(self, playing=False, **kwargs):
        """Apply a slider change that lands mid-playback straight away."""
        if playing:
            self._apply_playback_image()

    def _calculate_target_frame(self, elapsed_seconds, bpm, nframes):
        """Calculate target frame based on elapsed time.

        Args:
            elapsed_seconds: Time since playback started
            bpm: Beats per minute (playback speed)
            nframes: Total number of frames

        Returns:
            Target frame index (0 to nframes-1)
        """
        cycle_duration = 60.0 / bpm
        cycles_elapsed = elapsed_seconds / cycle_duration
        fractional_frame = (cycles_elapsed * nframes) % nframes
        return int(fractional_frame)

    async def _play_loop(self, playing, **kwargs):
        """Robust cine playback loop with adaptive timing and frame skipping.

        Addresses:
        - Time-based frame calculation (not frame-increment based)
        - Render time accounting with adaptive sleep
        - Frequent cancellation checks for responsiveness
        - Frame skipping when behind schedule
        - Render debouncing to prevent concurrent renders
        - Task cancellation support for immediate pause
        """
        # Validate that at least one playback mode is active
        if not (self.server.state.incrementing or self.server.state.rotating):
            self.server.state.playing = False
            return

        self._playback_start_time = time.perf_counter()
        self._last_target_frame = self.server.state.frame
        self._last_render_duration = 0.0

        # Playback parameters
        CHECK_INTERVAL = 0.01  # Check pause flag every 10ms for responsiveness

        try:
            while self.server.state.playing:
                # Calculate elapsed time and target frame
                elapsed = time.perf_counter() - self._playback_start_time

                # Get current playback parameters from state
                with self.server.state as state:
                    bpm = state.bpm
                    nframes = self.scene.nframes
                    incrementing = state.incrementing
                    rotating = state.rotating
                    bpr = state.bpr

                # Calculate target frame from elapsed time (time-based, not frame-based)
                target_frame = self._calculate_target_frame(elapsed, bpm, nframes)

                # Determine if render is needed
                frame_changed = incrementing and (
                    target_frame != self._last_target_frame
                )
                needs_rotation = rotating
                needs_render = frame_changed or needs_rotation

                # Render only if needed and not already rendering (debouncing)
                if needs_render and not self._is_rendering:
                    self._is_rendering = True
                    render_start = time.perf_counter()

                    try:
                        with self.server.state as state:
                            # Writing the frame renders through the update_frame
                            # listener this block's flush fires, and only that
                            # render sees the new actors. A write that changes
                            # nothing fires nothing -- but then the views are
                            # already showing the frame. Rotation is the one
                            # thing with no listener to render it.
                            renders_on_flush = (
                                incrementing
                                and frame_changed
                                and state.frame != target_frame
                            )

                            if incrementing and frame_changed:
                                state.frame = target_frame
                                self._last_target_frame = target_frame

                            if rotating:
                                deg = 360 / (nframes * bpr)
                                self.scene.renderer.GetActiveCamera().Azimuth(deg)
                                if not renders_on_flush:
                                    self.server.controller.view_update()

                        # Track render duration for adaptive timing
                        self._last_render_duration = time.perf_counter() - render_start

                    finally:
                        self._is_rendering = False

                # Adaptive sleep interval calculation
                base_interval = 60.0 / bpm / nframes
                adjusted_interval = max(
                    CHECK_INTERVAL, base_interval - self._last_render_duration
                )

                # Sleep in small chunks to remain responsive to pause
                remaining_sleep = adjusted_interval
                while remaining_sleep > 0 and self.server.state.playing:
                    sleep_chunk = min(CHECK_INTERVAL, remaining_sleep)
                    await asyncio.sleep(sleep_chunk)
                    remaining_sleep -= sleep_chunk

                    # Early exit check after each sleep chunk
                    if not self.server.state.playing:
                        break

        except asyncio.CancelledError:
            # Task was cancelled (pause button pressed) - exit gracefully
            pass
        finally:
            # Clean up playback state
            self._playback_start_time = None
            self._last_target_frame = None
            self._is_rendering = False

    def increment_frame(self):
        if not self.server.state.playing:
            self.server.state.frame = (self.server.state.frame + 1) % self.scene.nframes
            self.server.controller.view_update()

    def decrement_frame(self):
        if not self.server.state.playing:
            self.server.state.frame = (self.server.state.frame - 1) % self.scene.nframes
            self.server.controller.view_update()

    def reset_all(self):
        self.server.state.frame = 0
        self.server.state.playing = False
        self.server.state.incrementing = True
        self.server.state.rotating = False
        self.server.state.bpm = 60
        self.server.state.bpr = 5
        self.server.state.playback_quality = DEFAULT_PLAYBACK_QUALITY
        self.server.state.playback_resolution = DEFAULT_PLAYBACK_RESOLUTION
        self.server.controller.view_update()

    @asynchronous.task
    async def close_application(self):
        """Close the application by stopping the server."""
        await self.server.stop()
