"""The volumetry charts: what is measured off the labels, and when it is drawn.

A curve covers every frame, so almost nothing here happens per frame: the
measurement is taken once for a segmentation and held, and a frame change moves
a rule across a chart that is already drawn.  Nothing is measured at all while
the charts are off screen, which is what keeps a cine from paying for a view
nobody is looking at.

The report is a page per structure rather than a capture of what is on screen.
A viewport capture records a view; this records a measurement, and the two part
company as soon as the measurement is longer than one page.  It goes out through
the same writers all the same, so asking for it as ``dicom-rendered`` gives a
Secondary Capture series and as ``png`` gives numbered stills.
"""

# System
import csv
import datetime as dt
import logging

# Internal
from ..action import action
from ..capture import Context, WindowFrames, wants_alpha, writer_for
from ..volumetry import (
    body_size,
    body_surface_area,
    measure,
    metrics_table,
    page_table,
    timeseries,
    voxel_volume_ml,
)
from .base import Controller

logger = logging.getLogger(__name__)

# The value of ``maximized_view`` that puts the charts on screen.
VOLUMETRY_LAYOUT = "volumetry"

# What the report's images are filed under, inside the timestamped directory
# the rest of the export goes to.  Not "volumetry" again: they sit beneath a
# directory of that name already.
PAGES = "pages"

# What the report's series is called when the config names nothing.
REPORT = "Cardiac Volumetry"


class VolumetryController(Controller):
    """Chamber volumes over the cycle, and the chart they are drawn in."""

    # volumetry_seg_label is absent: an empty one means the first segmentation,
    # and a scene may have none, so it is written by hand below.
    seeds = ("volumetry_indexed",)

    def __init__(self, app):
        super().__init__(app)
        self._result = None
        self._drawn = None

    def register(self):
        state = self.server.state
        state.change("maximized_view")(self.refresh)
        state.change("volumetry_structure")(self.turn_page)
        state.change("volumetry_seg_label", "volumetry_indexed")(self.remeasure)
        state.change("frame")(self.mark_frame)

    def seed(self):
        """Pick the segmentation first, then write what the config says.

        The same order ``TileController`` seeds in, and for the same reason:
        an empty label means the first segmentation, which is a fact about the
        scene rather than about the config, so the picker has to be settled
        before anything reads it.
        """
        self.server.state.volumetry_seg_label = self.seeded_label()
        super().seed()

        state = self.server.state
        state.volumetry_rows = []
        state.volumetry_structures = self.names
        state.volumetry_structure = self.names[0] if self.names else ""
        state.volumetry_saved_at = None
        state.volumetry_summary = ""
        state.volumetry_ok = True

        self.forget()

    def seeded_label(self) -> str:
        """The segmentation to measure: the one configured, or the first one.

        A configured label is written through whatever the scene holds, as the
        tile picker's is: a label naming no segmentation is a config to correct
        rather than one to quietly reinterpret, and the charts simply stay
        empty until it is.
        """
        configured = self.scene.volumetry.segmentation_label
        if configured:
            return configured
        labels = [seg.label for seg in self.scene.segmentations]
        return labels[0] if labels else ""

    @property
    def names(self) -> list[str]:
        """The structures there are pages for, in the order they are drawn."""
        return [group.name for group in self.scene.volumetry.groups]

    @property
    def page(self) -> int:
        """Which structure's page is showing, as an index into ``names``."""
        chosen = getattr(self.server.state, "volumetry_structure", "")
        return self.names.index(chosen) if chosen in self.names else 0

    @property
    def active(self) -> bool:
        """Whether the charts are the view on screen.

        Everything below returns on this. Measuring a 4D segmentation is cheap
        but not free, and redrawing five charts thirty times a second for a
        window nobody is showing is neither.
        """
        return self.server.state.maximized_view == VOLUMETRY_LAYOUT

    @property
    def views(self):
        """The chart window, or None before the layout has built it."""
        return self.scene.volumetry_views

    def segmentation(self):
        """The segmentation the charts are measured off, or None."""
        label = getattr(self.server.state, "volumetry_seg_label", "")
        return next((s for s in self.scene.segmentations if s.label == label), None)

    def forget(self):
        """Drop the measurement, so the next refresh takes it again."""
        self._result = None
        self._drawn = None

    def body_surface_area(self) -> float | None:
        """The area to index by, from the config or from the images' header.

        The configured height and weight win over the tags, one at a time: a
        study that recorded only a weight should still be indexable by adding
        the height, rather than having to restate both.
        """
        config = self.scene.volumetry
        if not getattr(self.server.state, "volumetry_indexed", config.indexed):
            return None

        height, weight = config.patient_height_m, config.patient_weight_kg
        if height is None or weight is None:
            segmentation = self.segmentation()
            source = segmentation.source if segmentation is not None else None
            tagged = body_size(source.header if source is not None else {})
            height = height if height is not None else tagged[0]
            weight = weight if weight is not None else tagged[1]

        return body_surface_area(config.bsa_formula, height, weight)

    def result(self):
        """The measurement, taken once per segmentation and held.

        The counting behind it is memoised on the segmentation itself, so this
        holding is only of the arithmetic over it -- which is cheap, and is
        held anyway because the charts are rebuilt from the same object.
        """
        if self._result is None:
            segmentation = self.segmentation()
            if segmentation is None or not self.scene.volumetry.groups:
                return None

            frames = len(segmentation.actors) or 1
            self._result = measure(
                [segmentation.label_counts(frame) for frame in range(frames)],
                voxel_volume_ml(segmentation.mpr_image_data(0)),
                self.scene.volumetry.groups,
                bsa=self.body_surface_area(),
            )
        return self._result

    def refresh(self, **kwargs):
        """Rebuild the page, if it is on screen and not already right."""
        if not self.active or self.views is None:
            return

        signature = (self.server.state.volumetry_seg_label, self.page)
        if signature == self._drawn:
            return

        result = self.result()
        if result is None:
            return

        self.views.show(result)
        self.views.set_page(self.page)
        self.views.set_frame(self._frame)
        self._drawn = signature
        self.publish()
        self.server.controller.volumetry_update()

    def turn_page(self, **kwargs):
        """Draw another structure's page, off the measurement already taken."""
        self.refresh()

    def remeasure(self, **kwargs):
        """Take the measurement again, because what it was taken off changed."""
        self.forget()
        self.refresh()

    def publish(self):
        """Put the shown page's numbers where the drawer can list them.

        The page's rather than every structure's: the drawer sits beside one
        chart, and a table of all of them there would be the crowding this
        feature moved away from.
        """
        result = self.result()
        if result is None or not result.measurements:
            self.server.state.volumetry_rows = []
            return

        header, rows = page_table(result.measurements[self.page], result.bsa)
        self.server.state.volumetry_rows = [dict(zip(header, row)) for row in rows]

    def mark_frame(self, frame, **kwargs):
        """Move the rule saying which frame the rest of the app is showing.

        Drawn here rather than left to the frame path's own redraw: playback
        registers its listener first, so the render it asks for happens before
        this has moved anything, and the rule would sit a frame behind.
        """
        if not self.active or self.views is None:
            return

        self.views.set_frame(frame)
        self.server.controller.volumetry_update()

    @action("save_volumetry")
    def save_volumetry(self):
        """Write the curves and the metrics out as two CSV files.

        Not a background action, unlike a capture: this is two small file
        writes with no frames to step through and nothing to watch, so making
        it a coroutine would only mean a script had to wait for it by hand.

        Measured on demand rather than only when the charts are up, so that a
        headless session can ask for the numbers without first arranging to be
        looking at them.
        """
        result = self.result()
        if result is None:
            self.report("Nothing to export: no structures are configured", False)
            return

        stamp = dt.datetime.now().astimezone()
        directory = self.scene.volumetry_directory / stamp.strftime(
            self.scene.timestamp_format
        )
        directory.mkdir(parents=True, exist_ok=True)

        write_csv(directory / "timeseries.csv", *timeseries(result))
        write_csv(directory / "metrics.csv", *metrics_table(result))
        pages = self.write_pages(directory, result)

        self.report(
            f"Wrote 2 tables and {pages} pages to {directory.name}",
            True,
        )

    def write_pages(self, directory, result) -> int:
        """One image per structure, through the writer the format asks for.

        The report is a sequence, and a sequence is exactly what the capture
        writers take -- so ``dicom-rendered`` lands a Secondary Capture series
        in the study the images came from, and ``png`` lands numbered stills,
        without either being spelled out here.

        The window is left showing whatever page it was on: an export is not a
        thing that should move what the reader is looking at.
        """
        views = self.views
        if views is None:
            return 0

        # A viewport the page has never laid out is nothing by nothing, and a
        # window that size renders nothing and captures an empty buffer. The
        # report can be asked for without the charts ever having been looked
        # at, so it sizes the window the way a session with no browser does.
        if not all(views.window.GetSize()):
            views.window.SetSize(*self.scene.headless_size)

        fmt = self.app.capture.capture_format
        number, description = self.app.capture.series_for(VOLUMETRY_LAYOUT)
        context = Context(
            directory=directory,
            viewport=PAGES,
            frame_duration=self.app.capture.frame_duration(),
            window=self.server.state.mpr_window,
            level=self.server.state.mpr_level,
            identity=self.app.capture.identity(),
            series_number=number,
            series_description=description or REPORT,
            # A chart has no cut behind it, so a data capture of one records
            # the picture instead -- as it does for the volume rendering.
            has_plane=False,
            banner=self.app.capture.banner,
        )

        # Handed the measurement here rather than relying on the layout having
        # drawn it: a report is of what was measured, not of what is on screen,
        # and asking for one without having opened the charts is ordinary.
        showing = views.page
        views.show(result)
        views.set_frame(self._frame)

        frames = WindowFrames(
            views.window, alpha=wants_alpha(fmt), banner=context.banner
        )
        writer = writer_for(fmt, context)
        try:
            for page in range(views.pages):
                views.set_page(page)
                views.set_frame(self._frame)
                self.server.controller.volumetry_update()
                writer.add(page, frames.capture(None))
        finally:
            writer.close()
            views.set_page(showing)
            views.set_frame(self._frame)
            self.server.controller.volumetry_update()

        return views.pages

    def report(self, summary: str, ok: bool):
        """Say what an export did, in the shape the capture save reports."""
        with self.server.state as state:
            state.volumetry_saved_at = (
                dt.datetime.now().astimezone().strftime("%H:%M:%S")
            )
            state.volumetry_summary = summary
            state.volumetry_ok = ok


def write_csv(path, header: list[str], rows: list[list]):
    """One table as a CSV file, and where it went.

    ``None`` is written as an empty field rather than as the dash the on-screen
    table shows: a reader here is a spreadsheet or a script, and to one of
    those a dash in a column of numbers is a parse error rather than a mark
    meaning the number would not have meant anything.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(
            ["" if value is None else value for value in row] for row in rows
        )
    return path
