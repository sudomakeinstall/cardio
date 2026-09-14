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

# Third Party
import pydicom as pd

# Internal
from .. import dicom
from ..action import action
from ..capture import Context, WindowFrames, preflight, seg, sr, wants_alpha, writer_for
from ..volumetry import (
    STRUCTURES,
    Result,
    body_size,
    body_surface_area,
    measure,
    metrics_table,
    page_table,
    structure_rows,
    timeseries,
    usable_groups,
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


def blank_structure() -> dict:
    """A structure with nothing decided yet, as state spells one.

    Every field the model has, so that the drawer binds against the same shape
    whether the row came from a config or from the Add button -- a key that
    only appears once it is typed into is a key the widget cannot bind.
    """
    return {"name": "", "labels": [], "color": None, "density": None, "chamber": True}


class VolumetryController(Controller):
    """Chamber volumes over the cycle, and the chart they are drawn in."""

    # volumetry_seg_label is absent: an empty one means the first segmentation,
    # and a scene may have none, so it is written by hand below.
    seeds = ("volumetry_indexed", "volumetry_groups")

    def __init__(self, app):
        super().__init__(app)
        self._result = None
        self._drawn = None

    def register(self):
        state = self.server.state
        state.change("maximized_view")(self.refresh)
        state.change("volumetry_structure")(self.turn_page)
        state.change("volumetry_groups")(self.restructure)
        state.change("volumetry_seg_label")(self.resegment)
        state.change("volumetry_indexed")(self.remeasure)
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
        state.volumetry_available_labels = self.available_labels()
        state.volumetry_indexable = self.available_area() is not None
        self.publish_structures()
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

    def groups(self):
        """The structures state currently describes, as models.

        State is what the drawer edits, so it is what this reads -- the scene's
        own list is the seed and nothing more.  A row still being filled in is
        dropped rather than refused, which is the whole of ``usable_groups``.
        """
        return usable_groups(self.rows())

    def rows(self) -> list:
        """What the drawer currently holds, finished structures or not."""
        return structure_rows(getattr(self.server.state, "volumetry_groups", None))

    @property
    def names(self) -> list[str]:
        """The structures there are pages for, in the order they are drawn."""
        return [group.name for group in self.groups()]

    def available_labels(self) -> list[dict]:
        """The label picker's options, off the segmentation being measured."""
        segmentation = self.segmentation()
        if segmentation is None:
            return []
        return [
            {"title": str(value), "value": value}
            for value in segmentation.get_labels(self._frame)
        ]

    def publish_structures(self):
        """Fill the structure picker, keeping the page showing where it can.

        A structure can be renamed or deleted out from under the picker, so the
        selection is only moved when the one it names has stopped existing --
        editing the structure below the one being read should not turn the page.
        """
        names = self.names
        state = self.server.state
        state.volumetry_structures = names
        if getattr(state, "volumetry_structure", "") not in names:
            state.volumetry_structure = names[0] if names else ""

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
        """The area the measurement is indexed by, or None if it is not to be.

        Split from ``available_area`` because the tick and the tags answer
        different questions: one is whether to index, the other whether there
        is anything to index by.  The drawer needs the second on its own, to
        say why the tick is not offered.
        """
        config = self.scene.volumetry
        if not getattr(self.server.state, "volumetry_indexed", config.indexed):
            return None
        return self.available_area()

    def available_area(self) -> float | None:
        """The area there is to index by, from the config or the images.

        The configured height and weight win over the tags, one at a time: a
        study that recorded only a weight should still be indexable by adding
        the height, rather than having to restate both.
        """
        config = self.scene.volumetry
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
        if self._result is not None:
            return self._result

        segmentation = self.segmentation()
        if segmentation is None:
            return None

        groups = self.groups()
        frames = len(segmentation.actors) or 1
        bsa = self.body_surface_area()

        # Nothing to measure is not nothing to draw: an empty measurement is
        # what a blank page is made of, and counting voxels for no structure
        # would be work done to arrive at the same place.
        if not groups:
            self._result = Result(measurements=[], bsa=bsa, frames=frames)
        else:
            self._result = measure(
                [segmentation.label_counts(frame) for frame in range(frames)],
                voxel_volume_ml(segmentation.mpr_image_data(0)),
                groups,
                bsa=bsa,
            )
        return self._result

    def refresh(self, **kwargs):
        """Rebuild the page, if it is on screen and not already right."""
        if not self.active or self.views is None:
            return

        signature = (
            self.server.state.volumetry_seg_label,
            tuple(self.groups()),
            self.page,
        )
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

    def resegment(self, **kwargs):
        """Measure off another segmentation, and offer its labels instead.

        The labels index into the segmentation being left, so the picker
        cannot survive the change -- the same reason the tile grid rebuilds
        its own.  What the structures name is left alone: a chamber keeps its
        label numbers across two segmentations of the same study, and a
        structure naming one the new segmentation lacks measures zero and says
        so rather than quietly emptying itself.
        """
        state = self.server.state
        state.volumetry_available_labels = self.available_labels()
        # The body size is read off the segmentation's own header, so which
        # segmentation is chosen decides whether there is one at all.
        state.volumetry_indexable = self.available_area() is not None
        self.remeasure()

    def restructure(self, **kwargs):
        """The structures changed, so the pages and the picker follow."""
        self.forget()
        self.publish_structures()
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

    @action("add_structure")
    def add_structure(self):
        """Put a blank structure at the end of the list, to be filled in.

        Blank rather than named: a placeholder would have to be cleared before
        a real name could be typed, and an unnamed row already says what it is
        by having nothing in it.
        """
        rows = self.rows()
        if len(rows) >= self.scene.max_volumetry_groups:
            logger.warning(
                f"At most {self.scene.max_volumetry_groups} structures can be "
                "measured; raise max_volumetry_groups to add another."
            )
            return

        self.write_rows([*rows, blank_structure()])

    @action("remove_structure")
    def remove_structure(self, index: int):
        """Drop the structure at ``index``, finished or not.

        A new list rather than a pop: trame notices a variable being assigned,
        not a list being mutated under one.
        """
        rows = self.rows()
        if 0 <= index < len(rows):
            self.write_rows([*rows[:index], *rows[index + 1 :]])

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
        if result is None or not result.measurements:
            self.report("Nothing to export: no structures are named yet", False)
            return

        stamp = dt.datetime.now().astimezone()
        directory = self.scene.volumetry_directory / stamp.strftime(
            self.scene.timestamp_format
        )
        directory.mkdir(parents=True, exist_ok=True)

        write_csv(directory / "timeseries.csv", *timeseries(result))
        write_csv(directory / "metrics.csv", *metrics_table(result))
        pages = self.write_pages(directory, result)
        instances = self.write_dicom(directory, result)

        written = f"Wrote 2 tables and {pages} pages"
        if instances:
            written += f" and {instances} DICOM instance(s)"
        self.report(f"{written} to {directory.name}", True)

    def write_dicom(self, directory, result) -> int:
        """The segmentation and the measurements, as objects an archive files.

        Beside the tables rather than instead of them: a person opens a CSV and
        a system reads a Structured Report, and both are wanted.

        Only where the volume was read from DICOM.  A segmentation object names
        the images it segments and a report names the images it is evidence
        about, and a volume read from a file gives neither anything to name --
        so a research session gets its tables and is told why that is all.

        Refused on the same terms a capture is, and for the same reason: a
        report says whose measurements these are, and one that says it under a
        patient nobody checked is worse than no report.  The tables are written
        either way; they are nobody's to mistake for an archive object.
        """
        volume = self._active_volume()
        source = volume.source if volume is not None else None
        if source is None or not source.instances:
            logger.info(
                "The measured volume was not read from DICOM, so there is "
                "nothing for a segmentation or a report to name; tables only."
            )
            return 0

        if not self.scene.research:
            refused = preflight.refused(
                dicom.read_dataset(source.instances[0]), self.scene.uid_root
            )
            if refused:
                logger.warning(f"{refused}. The tables were written.")
                return 0

        segmentation = self.segmentation()
        groups = [measurement.group for measurement in result.measurements]
        equipment = self.scene.capture_equipment
        uid_root = self.scene.uid_root

        segmentations = []
        if segmentation is not None:
            paths = seg.write_segmentation(
                segmentation,
                source,
                directory / "segmentation",
                groups=groups,
                equipment=equipment,
                uid_root=uid_root,
            )
            segmentations = [pd.dcmread(path) for path in paths]

        sr.write_measurements(
            result,
            source,
            directory / "measurements.dcm",
            segmentations=segmentations,
            equipment=equipment,
            uid_root=uid_root,
        )
        return len(segmentations) + 1

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
            transfer_syntax=self.scene.capture_transfer_syntax,
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

    def write_rows(self, rows: list):
        """Put the structures back, in the shape the drawer's widget holds.

        A new object rather than a list appended to: trame notices a variable
        being assigned, and the widget mirroring it holds an object.
        """
        self.server.state.volumetry_groups = {STRUCTURES: rows}

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
