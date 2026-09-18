"""Regions traced on a cut, the area each encloses, and the pose each was at."""

# System
import datetime as dt
import logging

# Internal
from .. import camera, planimetry, registry
from ..action import action, background
from ..contour import SELECTED_COLOR, TRACED_COLOR, TRACING_COLOR, ContourActors
from ..measurement import Measurement, MeasurementSet, TileCut
from ..planimetry import ContourStyle
from ..reslice import VIEWS
from ..snap import Snap
from ..tile import Tile
from ..view import Layout
from .base import Controller

logger = logging.getLogger(__name__)

# The fewest points that enclose anything.
MINIMUM_POINTS = 3

# The viewport name the tile grid goes by, which is a grid of cuts rather than
# one of the three panes.
TILE_VIEW = "tile"

# The closure styles as the picker labels them. Keyed by the enum, so a new
# member without a label fails here rather than silently missing from it.
CONTOUR_TITLES = {
    ContourStyle.POLYGON: "Straight",
    ContourStyle.SPLINE: "Spline",
}


class MeasurementController(Controller):
    """Owns the traced regions, and what is on screen of them."""

    seeds = ("measurement_data", "measurement_contour")

    def __init__(self, app):
        super().__init__(app)
        # The trace in progress is held here rather than in state: it is not a
        # document, and pushing it through a flush would put a round trip on
        # every click.
        self._points: list[tuple[float, float]] = []
        self._cut: planimetry.Cut | None = None
        self._view: str = ""
        self._tile: int | None = None
        self._at_frame: int = 0
        self._actors: dict[tuple[str, object], ContourActors] = {}

    def register(self):
        state = self.server.state
        state.change("measurement_data")(self._on_edited)
        state.change("measuring")(self._on_mode_changed)
        state.change("measurement_selected")(self.refresh)
        # The MPR controller's own layout listener gives up before redrawing
        # when the new layout does not show the cuts, which is exactly when the
        # regions have to come off them.
        state.change("maximized_view")(self.refresh)

    def seed(self):
        """The regions a config opened with, and the state the tracing keeps."""
        super().seed()
        state = self.server.state

        state.measurement_contour_items = [
            {"text": CONTOUR_TITLES[style], "value": style.value}
            for style in ContourStyle
        ]

        # The mode and the trace in progress start empty every session: a
        # config can open the app showing regions, but not part way through
        # drawing one.
        self._abandon()
        state.measuring = False
        state.measurement_selected = None
        state.measurements_saved_at = None
        state.measurements_stale = False
        state.measurement_on_plane = [False for _ in self.regions.measurements]

    # --- the set ----------------------------------------------------------

    @property
    def regions(self) -> MeasurementSet:
        """The traced regions as their model, validated on the way in.

        From trame state, which is what the drawer edits, so a rename made
        there is read back here rather than only in the scene.
        """
        return MeasurementSet(**dict(self.server.state.measurement_data or {}))

    def publish(self, regions: MeasurementSet):
        """The only place the set is written back.

        Keeps the scene and the state in step, as ``RotationController.publish``
        does for the rotation sequence.
        """
        self.scene.measurements = regions
        self.server.state.measurement_data = regions.model_dump(mode="json")

    def _on_edited(self, **kwargs):
        """Every change to the set is one the saved file does not have yet.

        Hung off the state key rather than added to each action, so a rename
        typed straight into the drawer counts alongside the ones that come
        through ``publish``.
        """
        if self.server.state.measurements_saved_at:
            self.server.state.measurements_stale = True
        self.refresh()

    def _on_mode_changed(self, **kwargs):
        """Leaving the mode abandons whatever was half traced.

        A trace only means anything closed, and one left lying about would come
        back the next time the mode was entered, on a cut nobody was looking at.
        """
        if not self.server.state.measuring:
            self._abandon()
        self.refresh()

    def _abandon(self):
        self._points = []
        self._cut = None
        self._view = ""
        self._tile = None
        self.server.state.measurement_pending = 0
        self.server.state.measurement_view = ""

    # --- tracing ----------------------------------------------------------

    def _renderer(self, view_name: str, tile: int | None = None):
        """The renderer a cut draws into, or None before it is built."""
        if view_name == TILE_VIEW:
            views = self.scene.tile_views
            if views is None or tile is None or not 0 <= tile < len(views):
                return None
            return views.renderers[tile]

        views = self.scene.mpr_views
        if views is None or view_name not in VIEWS:
            return None
        return views.renderer(view_name)

    def cut_of(self, view_name: str, tile: int | None = None):
        """Where a cut sits in the patient, as it stands now.

        A tile is posed along a path rather than by the shared origin, so its
        cut is read from the pose the grid would give that tile -- the same one
        ``update_tiles`` hands the reslice -- rather than from the MPR pose.
        """
        if view_name == TILE_VIEW:
            poses = self._tile_poses()
            if poses is None or tile is None or not 0 <= tile < len(poses):
                return None
            return planimetry.cut_from(*poses[tile], self.app.tiles.cut_plane)

        if view_name not in VIEWS:
            return None
        return planimetry.cut_from(*self.app.mpr.current_pose(), view_name)

    def _tile_poses(self):
        """One pose per tile, or None when the grid has no path to sample."""
        try:
            return self.app.tiles.tile_poses(self._frame)
        except (RuntimeError, ValueError, IndexError):
            # A grid whose path has gone -- a label dropped, a segmentation
            # swapped -- draws nothing rather than taking the redraw down with
            # it, which is what the panel already reports on its own.
            return None

    def _sub_model(self, prefix: str, model):
        """The ``Scene`` sub-model at ``prefix``, as state now holds it.

        Walked from the registry rather than from a field list, so a tile or
        snap setting added later is carried by a region without this having to
        be told about it.
        """
        fields = {}
        for key in registry.keys_in_scope(registry.Scope.DOCUMENT):
            source = registry.source_of(key)
            if source.startswith(f"{prefix}."):
                fields[source.removeprefix(f"{prefix}.")] = registry.to_config(
                    key, self.server.state[key]
                )
        return model(**fields)

    def _restore_sub_model(self, prefix: str, model):
        """Write ``model`` back over the state keys ``prefix`` names."""
        for key in registry.keys_in_scope(registry.Scope.DOCUMENT):
            source = registry.source_of(key)
            if source.startswith(f"{prefix}."):
                field = source.removeprefix(f"{prefix}.")
                self.server.state[key] = registry.to_state(key, getattr(model, field))

    @action("toggle_measuring")
    def toggle_measuring(self):
        """Start or stop placing the points of a region."""
        state = self.server.state
        state.measuring = not state.measuring

    @action("place_measurement_point")
    def place_measurement_point(self, view_name: str, x: float, y: float):
        """Put a point of a region where a click landed on a cut.

        ``x`` and ``y`` are display coordinates, as the view events carry them
        and as ``rotate_view`` takes them: what the gesture layer reports is
        where the cursor was, and what that means is decided here.

        A script replaying this places its points where *its* window puts them,
        which is not necessarily where they were placed by hand. The durable
        record of a region is the measurement file, not the script.
        """
        if not self.server.state.measuring:
            return

        # The tile grid is one window of viewports, so which cut was clicked is
        # a question only the grid can answer.
        tile = None
        if view_name == TILE_VIEW:
            views = self.scene.tile_views
            tile = None if views is None else views.tile_at(x, y)
            if tile is None:
                return

        renderer = self._renderer(view_name, tile)
        cut = self.cut_of(view_name, tile)
        if renderer is None or cut is None:
            return

        if self._cut is None:
            self._cut = cut
            self._view = view_name
            self._tile = tile
            self._at_frame = self._frame
        elif not (
            planimetry.same_plane(self._cut, cut) and self._frame == self._at_frame
        ):
            # Refused rather than filed on the wrong plane: the points already
            # placed are measured in a cut this one is not.
            logger.warning(
                "Ignoring a point placed on a different cut from the one being "
                "traced in; close or cancel the region first."
            )
            return

        # In the traced cut's own millimetres, not this one's. They are the same
        # plane, but a pan between two clicks would otherwise move the origin
        # the later points are measured from.
        point = camera.cut_point(renderer, x, y)
        self._points.append(
            tuple(planimetry.from_lps(self._cut, planimetry.to_lps(cut, [point]))[0])
        )

        self.server.state.measurement_pending = len(self._points)
        self.server.state.measurement_view = self._view
        self.refresh()

    @action("undo_measurement_point")
    def undo_measurement_point(self):
        """Take back the last point placed."""
        if not self._points:
            return

        self._points.pop()
        if not self._points:
            self._abandon()
        else:
            self.server.state.measurement_pending = len(self._points)
        self.refresh()

    @action("cancel_measurement")
    def cancel_measurement(self):
        """Give up on the region being traced."""
        self._abandon()
        self.refresh()

    @action("close_measurement")
    def close_measurement(self, name: str = ""):
        """Close the region being traced, and measure what it encloses."""
        if self._cut is None:
            return

        if len(self._points) < MINIMUM_POINTS:
            logger.warning(
                "A region needs at least %d points to enclose anything; this "
                "one has %d.",
                MINIMUM_POINTS,
                len(self._points),
            )
            return

        regions = self.regions
        regions.measurements.append(
            Measurement(
                name=name or f"Region {len(regions.measurements) + 1}",
                view=self._view,
                frame=self._at_frame,
                contour=ContourStyle(self.server.state.measurement_contour),
                points=self._points,
                cut=self._cut,
                pose=self._pose(),
                tile=self._tile_cut(),
            )
        )

        self._abandon()
        self.publish(regions)
        self.refresh()

    def _tile_cut(self) -> TileCut | None:
        """The grid a tile region was traced on, or None for an MPR pane.

        A tile has no pose of its own to write down: it is computed from the
        panel's settings and, for the traverse source, from the path the snap
        selection names. So putting a grid back is putting those settings back,
        and both are carried whole -- a grid recalled from half of them would
        land the region on a different tile, which is the one outcome worth
        ruling out.
        """
        if self._view != TILE_VIEW or self._tile is None:
            return None

        return TileCut(
            index=self._tile,
            grid=self._sub_model("tile", Tile),
            snap=self._sub_model("snap", Snap),
        )

    def _pose(self):
        """The rotation sequence the cuts stand at, carrying the live origin.

        The sequence holds an ``mpr_origin`` of its own, which is where a
        rotation file's origin arrives and is not moved by a pan -- so what a
        recall has to put back is read from the origin the views are actually
        aimed at rather than from the copy inside the sequence.
        """
        pose = self.app.rotations.rotation_sequence()
        pose.mpr_origin = [float(value) for value in self.server.state.mpr_origin]
        pose.metadata.timestamp = dt.datetime.now().astimezone().isoformat()
        pose.metadata.volume_label = self.server.state.active_volume_label
        return pose

    # --- one region at a time ---------------------------------------------

    def _edit(self, index: int, change):
        """Apply ``change`` to the region at ``index`` and publish the result."""
        regions = self.regions
        if not 0 <= index < len(regions.measurements):
            logger.warning("There is no measurement %d to change.", index)
            return

        change(regions.measurements[index])
        self.publish(regions)
        self.refresh()

    @action("rename_measurement")
    def rename_measurement(self, index: int, name: str):
        """Give the region at ``index`` a name of its own.

        The drawer edits the field directly, as the rotations panel does; this
        is the same thing by name, for the console and for a script.
        """

        def renamed(region):
            region.name = name

        self._edit(index, renamed)

    @action("restyle_measurement")
    def restyle_measurement(self, index: int, contour: str):
        """Close the region at ``index`` the other way, and measure it again.

        The picker beside the mode says how the *next* region is closed, and
        deliberately leaves the ones already taken alone: a recorded area that
        changed because a picker moved would be a number nobody chose. This is
        how one is changed on purpose -- and tracing the same points both ways
        is the cheapest paired comparison there is, being the same clicks twice.
        """

        def restyled(region):
            region.contour = ContourStyle(contour)

        self._edit(index, restyled)

    @action("delete_measurement")
    def delete_measurement(self, index: int):
        """Drop the region at ``index``."""
        regions = self.regions
        if not 0 <= index < len(regions.measurements):
            logger.warning("There is no measurement %d to delete.", index)
            return

        regions.measurements.pop(index)
        self._forget(index)
        self.server.state.measurement_selected = None
        self.publish(regions)
        self.refresh()

    @action("clear_measurements")
    def clear_measurements(self):
        """Drop every region traced this session."""
        self._forget(0)
        self.server.state.measurement_selected = None
        self.publish(MeasurementSet(metadata=self.regions.metadata))
        self.refresh()

    def _forget(self, first: int, count: int = 0):
        """Take the props of regions ``first`` and after off every cut.

        The props are keyed by position, and removing a region shifts every
        later one down into a key that already has props pointed at the region
        that used to be there. Dropping them from ``first`` on is what stops a
        deleted region leaving its outline behind under its successor's name.

        Walks the props actually held rather than the views: a region may be
        drawn on any of the three panes and on any tile, and which of them it
        reached is what the keys say.
        """
        for key in list(self._actors):
            where, index = key
            if not isinstance(index, int) or index < first:
                continue

            actors = self._actors.pop(key)
            renderer = self._renderer_of(where)
            if renderer is not None:
                actors.remove_from(renderer)

    def _renderer_of(self, where: str):
        """The renderer a props key was drawn into, or None if it is gone."""
        if where.startswith(f"{TILE_VIEW}:"):
            return self._renderer(TILE_VIEW, int(where.split(":", 1)[1]))
        return self._renderer(where)

    @action("recall_measurement")
    def recall_measurement(self, index: int):
        """Put the cuts back where the region at ``index`` was measured.

        The order matters. The rotation sequence goes through
        ``RotationController.publish``, which is the one place rotation state is
        written and which carries the units and the index order with it -- so
        the origin written after it is already in the order the sequence just
        established.
        """
        regions = self.regions.measurements
        if not 0 <= index < len(regions):
            logger.warning("There is no measurement %d to recall.", index)
            return

        region = regions[index]
        state = self.server.state

        self._abandon()
        state.frame = region.frame
        self.app.rotations.publish(region.pose)
        state.mpr_origin = list(region.pose.mpr_origin)

        # A tile has no pose of its own: the grid computes one from these, so
        # they go back before the layout does or the tiles would be posed once
        # from the old settings on the way in.
        if region.tile is not None:
            self._restore_sub_model("tile", region.tile.grid)
            self._restore_sub_model("snap", region.tile.snap)

        # The quad view already shows all three cuts, so a recall from it is not
        # a reason to maximize one of them.
        if Layout(region.view) not in Layout.from_state(state.maximized_view).on_screen:
            state.maximized_view = region.view

        # A lock owns the pose and will snap back over this the next time
        # anything moves. Saying so beats unlocking on the user's behalf: a lock
        # is theirs, and a recall that silently dropped one would lose the
        # setting the rest of the session was working under.
        if state.snap_locked or state.snap_orientation_locked:
            logger.warning(
                "A snap lock is holding the pose; the recalled cut may be "
                "snapped away from again."
            )

        state.measurement_selected = index
        self.refresh()

    @action("save_measurements")
    @background
    async def save_measurements(self):
        """Write the regions out as a measurement file of their own.

        Under the active volume's name, as a rotation file is: the two travel
        together, and a config pointed at a new study by changing one directory
        keeps finding both.
        """
        timestamp = dt.datetime.now().astimezone()
        label = self.server.state.active_volume_label

        if not label:
            logger.warning("No active volume; there is nothing to save against.")
            return

        regions = self.regions
        if not regions.measurements:
            logger.warning("There are no measurements to save.")
            return

        regions.metadata.timestamp = timestamp.isoformat()
        regions.metadata.volume_label = label

        directory = self.scene.measurements_directory / label
        path = regions.to_file(
            directory / f"{timestamp.strftime(self.scene.timestamp_format)}.toml"
        )
        logger.info("Wrote %d measurements to %s", len(regions.measurements), path)

        with self.server.state as state:
            state.measurements_saved_at = timestamp.strftime("%H:%M:%S")
            state.measurements_stale = False

    # --- drawing ----------------------------------------------------------

    def refresh(self, **kwargs):
        """Redraw, and push the picture to whatever is on screen."""
        self.draw()
        self.server.controller.view_update()

    def draw(self, **kwargs):
        """Put every region that is on a cut onto that cut, and hide the rest.

        Called from the paths that *draw* the cuts as well as from the
        listeners above, and it has to be both. A rebuilt view has dropped the
        props with everything else, and the two rebuilds that matter -- a frame
        of a cine, and a segmentation overlay being switched on -- reach the
        renderers without a listener firing. The listeners cover the other
        half: what should be drawn changing while no cut is redrawn.

        A region is drawn in the *current* cut's millimetres rather than in the
        ones it was traced in. They differ by however much the view has been
        panned or rolled since, which is a rigid movement within the plane: the
        region stays on the anatomy and the area it encloses does not change.

        ``measurement_on_plane`` comes out of the same walk, so the dots in the
        drawer say what the renderers were actually told -- a region on a plane
        no layout is drawing reads as off, which is what the dot claims to mean.
        """
        regions = self.regions.measurements
        on_plane = [False] * len(regions)

        drawing = self.app.mpr.active
        views = self.scene.mpr_views

        for view in VIEWS if views is not None else ():
            self._draw_cut(view, None, views.renderer(view), regions, on_plane, drawing)

        self._draw_tiles(regions, on_plane)

        self.server.state.measurement_on_plane = on_plane

    def _draw_tiles(self, regions, on_plane):
        """The same again for every tile, each against its own pose.

        A tile region is not tied to the tile it was traced on: the grid is
        walked and the region drawn wherever a tile has come to rest on its
        plane. Which is the honest answer -- a tile is a place a cut is shown,
        not a thing a cut belongs to -- and it is what makes a region reappear
        after the grid is reshaped around it.
        """
        views = self.scene.tile_views
        if views is None:
            return

        drawing = self.app.tiles.active
        poses = self._tile_poses() if drawing else None

        for index, renderer in enumerate(views.renderers):
            cut = (
                planimetry.cut_from(*poses[index], self.app.tiles.cut_plane)
                if poses is not None and index < len(poses)
                else None
            )
            self._draw_cut(TILE_VIEW, index, renderer, regions, on_plane, drawing, cut)

    def _draw_cut(self, view, tile, renderer, regions, on_plane, drawing, cut=None):
        """Put every region that is on this one cut onto it, and hide the rest.

        ``cut`` is passed in where the caller has already worked it out, which
        the tile loop has: the grid's poses come as a list, and asking for each
        tile's again would recompute the whole path once per tile.
        """
        if renderer is None:
            return

        cut = self.cut_of(view, tile) if cut is None else cut
        drawing = drawing and cut is not None

        selected = self.server.state.measurement_selected
        key = view if tile is None else f"{TILE_VIEW}:{tile}"

        for index, region in enumerate(regions):
            shown = drawing and region.on(cut, self._frame)
            on_plane[index] = on_plane[index] or shown
            self._show(
                renderer,
                (key, index),
                handles=planimetry.from_lps(cut, region.lps()) if shown else [],
                style=region.contour,
                closed=True,
                visible=shown,
                color=SELECTED_COLOR if index == selected else TRACED_COLOR,
                label=f"{region.area:.1f} mm\u00b2",
            )

        tracing = (
            drawing
            and bool(self._points)
            and self._cut is not None
            and planimetry.same_plane(self._cut, cut)
        )
        self._show(
            renderer,
            (key, "tracing"),
            handles=self._points if tracing else [],
            style=ContourStyle.POLYGON,
            closed=False,
            visible=tracing,
            color=TRACING_COLOR,
            label="",
        )

    def _show(self, renderer, key, handles, style, closed, visible, color, label):
        """Point one region's props at what it looks like now.

        A region that is not being drawn has its props taken off the renderer
        rather than merely hidden: a layout that does not show the cuts should
        leave nothing behind on them, which is the same thing the reslicing
        itself is skipped for.
        """
        actors = self._actors.get(key)
        if actors is None:
            if not visible:
                return
            actors = self._actors[key] = ContourActors()

        if not visible:
            actors.set_visible(False)
            actors.remove_from(renderer)
            return

        curve = planimetry.contour_points(handles, style)
        actors.set_points(curve, handles, closed)
        actors.set_label(label, curve.mean(axis=0))
        actors.set_color(color)
        actors.add_to(renderer)
        actors.set_visible(True)
