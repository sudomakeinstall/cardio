# cardio

`cardio` is a simple web-based viewer for 3D and 4D ('cine') medical imaging data,
built primarily on [trame](https://github.com/kitware/trame),
[vtk](https://github.com/kitware/vtk), and
[itk](https://github.com/insightsoftwareconsortium/itk).  `cardio` can render sequences
of mesh files (e.g., `*.obj` files), segmentation files (e.g., `*.nii.gz` files with
discrete labels) and volume renderings of grayscale images (e.g., `*.nii.gz` files with
continuous values).  Images may be NIfTI or DICOM.  `cardio` is launched from the
commandline and may be configured via commandline arguments, a static TOML
configuration file, or a combination of the two.

## Quickstart

### Installation

```bash
$ cd /path/to/your/project
$ uv init
$ uv add cardio
$ . ./.venv/bin/activate
```

### Reading DICOM

Point an object at a directory holding a DICOM series and it is read as one:

```toml
[[volumes]]
label = "Cine"
directory = "./data/dicom-cine"
```

There is no format to declare.  A directory is read as DICOM when the frame
pattern (`{frame}.nii.gz` by default) finds nothing in it, so the NIfTI layouts
below keep working untouched.

A time-resolved series becomes one frame per cardiac phase.  Slices are
identified by their position along the acquisition normal rather than by
`InstanceNumber`, so a series numbered slice-major, phase-major or not usefully
at all all read the same; within a slice, phases are ordered by `TriggerTime`,
then `TemporalPositionIdentifier`, `AcquisitionTime` and `InstanceNumber`.  A
series that does not hold the same number of images at every slice location is
reported rather than quietly reshaped.

If the directory holds more than one series, `cardio` says so and lists them;
name the one you want:

```toml
[[volumes]]
label = "Cine"
directory = "./data/study"
series_uid = "1.2.840.113619.2.55.3.12345"
```

This is what makes a media export -- a study burned to CD or exported from a
PACS -- readable as it arrives.  Such a directory holds the whole study in a
nested `DICOM/PA…/ST…/SE…` tree: every series, a `DICOMDIR`, and whatever else
the exporter filed beside them.  Files that are not DICOM images are skipped,
and so is an image of a shape `cardio` does not unpack, which is named in the
log rather than dropped silently.  Only when there is nothing else to open does
that become the error, because then it is the answer.

Obliquely acquired data needs nothing special -- the acquisition axes are
carried through to the views, and the MPR cuts are taken in patient (LPS)
coordinates whatever the slices were angled to.

### Snapping to a segmentation feature

The MPR views can lock onto a segmentation feature: the centroid of a group of
labels (`label`), the interface between two groups (`interface`), or a point along
the line joining two such interfaces (`traverse`).  That selection can be made in
the Snap & Align panel, or written down so the app opens with it already applied:

```toml
[snap]
segmentation_label = "BL_Labels"
mode = "traverse"
labels_a = [1]
labels_b = [2]
labels_c = [3]
traverse = 50            # percent of the way from the A|B interface to B|C
locked = true            # hold the origin on the feature, frame to frame
orientation_locked = true  # hold the views in the fitted plane
```

The same fields are available on the commandline, either individually or as JSON:

```bash
$ cardio --snap.mode traverse --snap.labels_a '[1]' --snap.labels_b '[2]' --snap.labels_c '[3]'
$ cardio --snap '{"mode": "interface", "labels_a": [1], "labels_b": [2], "locked": true}'
```

Reset returns to this configuration rather than to an empty panel.  A label that
the segmentation does not contain is reported and dropped rather than refusing to
start.

### Labels an island away from themselves

An interface plane is fitted to every cell of the surface between the two
groups, weighted by area.  That makes it sensitive to a patch of label in the
wrong place in a way that looking at the segmentation is not: a handful of
voxels carrying a label 80 mm from the structure it names are invisible on
screen, and yet they sit at the end of a long lever, so the plane fitted
through them tilts to reach.  A `traverse` snap shows this as a cut that is
square on at one interface and skewed by the time it arrives at the other.

`label_components` is how many connected components of a label to keep, largest
first, counted as the frames are read and applied to every frame alike:

```toml
[[segmentations]]
label = "CCTASeg"
directory = "./data/seg"
label_components = {3 = 1, 9 = 1}   # one component each: no islands
```

```bash
$ cardio --segmentations '[{"label": "CCTASeg", "directory": "./data/seg", "label_components": {"9": 1}}]'
```

A label named nowhere here keeps every component it has, which is what all of
them do by default -- so this is a thing to turn on for the labels that should
be one piece, rather than a cleanup to apply across the board.  Trabeculae,
papillary muscles and a vessel that leaves and re-enters the field of view are
all legitimately in several pieces, and thinning those to one would be throwing
away anatomy.  What is dropped is written to the log, label by label and frame
by frame, because a filter quietly removing half a structure is worth hearing
about.

The erasure happens on the way in, so everything downstream sees the same
labels: the mesh, the overlay on the cuts, the volumetry, the zoom fit, and the
planes a snap is aligned to.  It is the frames in memory that are thinned and
never the files on disk.

### Opening in a particular view

The three cut views are named for where they sit in the quad view -- `ul`, `ll`
and `lr` -- rather than for an anatomical plane.  They open on the axial,
coronal and sagittal frames of an unrotated volume, but a rotation makes those
names false while the panes stay where they are, and a capture written as
`ll.gif` does not claim to be a coronal cut.

The layout and theme the app opens in, and where the playback controls start:

```toml
[view]
layout = "tile"     # quad (default), volume, ul, ll, lr, tile
theme = "dark"      # selects between the two [background] colours
camera_lock = "ll"  # MPR view the volume rendering's camera follows, or "free"
drawer_sections = ["playback", "tiles"]   # sections open on load
help_visible = false                      # open showing the shortcut reference

[playback]
bpm = 75            # playback speed, in beats per minute
bpr = 3             # cardiac cycles per full rotation of the camera
rotating = true     # rotate the camera while playing

[tile]
rows = 3            # rows in the tile grid, 1 to 6
cols = 3            # columns in the tile grid, 1 to 6
source = "spacing"  # traverse, spacing, or labels
plane = "ul"        # pane the parallel sources cut in: ul, ll, lr
spacing = 10.0      # millimetres between adjacent cuts, in the spacing source
labels = [1, 2]     # labels the grid spans end to end, in the labels source
reverse = false     # walk the path from the far end, without turning the cut
```

```bash
$ cardio --view.layout tile --playback.bpm 75 --tile.rows 2
```

Tile view draws several cuts of one volume side by side.  Where those cuts come
from is the `source`:

* `traverse` walks the path between the two interface planes, so it wants a
  `[snap]` block in traverse mode to have anything to show.  Its plane is the
  interface, which is why it takes no `plane`.
* `spacing` steps the quad view's own cut a fixed number of millimetres along
  its normal, centred on the MPR origin.  It asks for nothing but a volume, so
  it is what a scene with no segmentation opens on.
* `labels` steps that same cut across a chosen set of labels instead: the first
  and last tile sit on the labels' outermost bounds, and a bigger grid samples
  the same span more finely rather than covering more of it.  The selection is
  the tile panel's own, independent of the `[snap]` groups and the `[zoom]`
  labels.

`reverse` takes the same tiles in the opposite order.  It is the control to
reach for when a stack runs base to apex and you wanted apex to base: a half
turn would flip the normal too, but it would flip one of the in-plane axes with
it and hand back every tile mirrored.

While the grid is on screen it is what `zoom_to_labels` frames, measured in the
plane the tiles are cut in.  Reset returns the playback controls to whatever is
written here.

Both the fit and the `labels` span are measured off the label cloud, which is
unioned over every frame -- so a handful of mislabelled voxels in one frame sets
the extent for all of them.  `label_percentile` is how much of that cloud a
measurement has to cover; below 100 the outermost points are ignored:

```toml
label_percentile = 99.95   # ignore the outermost 0.025% at each end
```

How much a stray voxel costs depends on where the normal points, so one that
barely shows in an axial fit can be most of an oblique stack.

The `volume` and `tile` layouts do not draw the three MPR views, so the slices
behind them are not resampled while either is on screen; they are brought up to
date on the way back.

### Capturing what is on screen

The Export panel writes the ticked viewports to
`<serialization_directory>/screenshots/<timestamp>/`.  With the playback
controls incrementing or rotating, a capture runs the whole cine; otherwise it
takes a single frame.  When it finishes it says which viewports it wrote and
into which folder, counted from the files that actually landed.

**A viewport is captured only while the layout is showing it.**  The quad view
shows the three cuts and the volume rendering; every maximized layout shows one
thing.  Anything else is a window holding whichever frame was last drawn into
it, and a capture of that is indistinguishable from one that worked -- so the
checkboxes for viewports that are not on screen grey out.  They stay ticked, so
a trip through another layout does not lose the selection.

```toml
serialization_directory = "./data"
capture_format = "dicom-data"                          # png (default), jpeg, gif, mp4,
screenshot_viewports = ["ul", "ll", "tile"]            # dicom-rendered, dicom-data,
                                                       # dicom-cine-rendered,
                                                       # dicom-cine-data
uid_root = "1.2.840.99999"                             # the deployment's own
research = false                                       # waive what is unsafe to send

[capture_equipment]
institution_name = "St Elsewhere"
station_name = "READING-3"
```

```bash
$ cardio --capture-format mp4 --screenshot-viewports '["vr"]'
```

| Format | Written as | Holds |
| --- | --- | --- |
| `png`, `jpeg` | `<viewport>/<i>.<ext>` | The picture, one file per frame |
| `gif`, `mp4` | `<viewport>.<ext>` | The picture, one animation at the configured BPM |
| `dicom-rendered` | `<viewport>/<i>.dcm` | The picture, as an RGB Secondary Capture series |
| `dicom-data` | `<viewport>/<i>.dcm` | The pixels behind it, in greyscale |
| `dicom-cine-rendered` | `<viewport>/0000.dcm` | The picture, as one multi-frame object |
| `dicom-cine-data` | `<viewport>/0000.dcm` | The pixels behind it, as one multi-frame object |

`dicom-data` is the one that keeps the measurements.  For the MPR views it
writes the resliced plane itself: the volume's own values, so a viewer reads
the numbers the study holds rather than a picture of them, with the window and
level as `WindowWidth` and `WindowCenter` tags rather than applied to the
pixels, and the true `ImageOrientationPatient`, `ImagePositionPatient` and
`PixelSpacing` of the cut.  Such a series reads straight back into `cardio`.

Those values are sampled onto the grid the view draws the cut on, not onto the
volume's own: the same rectangle cut into the same number of pixels the picture
beside it has, so the two are one frame twice over.  They are linear
interpolations of the volume, which the reslice behind the view already was --
an oblique cut lands between samples whatever the sampling -- so what changes
with the grid is how many places the cut is asked for, not what it is cut from.
`SliceThickness` says so: it stays the volume's own sampling however fine the
pixels get.

The values are stored on the step the volume was measured on rather than
stretched across the stored range.  A CT read from NIfTI arrives as floats
holding whole Hounsfield units, and reslicing it interpolates between them, but
the finest difference the study ever held is still one unit -- so `RescaleSlope`
is 1 and a viewer's measurement reads in Hounsfield units directly.  Spreading
the same cut over all 65535 levels would keep it to a thirty-second of a unit it
never had, and a lossless encoder is obliged to preserve every one of those
invented levels: on the source's own step a capture is a little over a third the
size and holds the same numbers.  A volume whose values do fall between whole
numbers has no step to name, and is mapped onto the stored range as before.

Matching the picture is what makes the banner legible.  It is stamped in
proportion to the image it is stamped on, so a cut written at the volume's
sampling carries a band a few pixels tall -- crisp in the file, and unreadable
once a viewer has magnified a few hundred pixels to fill a screen.  A view that
has never been sized has no grid to match and is written whole at the volume's
sampling, which is the only case left where the two differ.

What it does not carry is anything drawn on top -- crosshairs, segmentation
overlays, the transfer function -- because none of those are pixel values.
`dicom-rendered` is for that: it records the viewport as it looked.

The tile grid is a special case in `dicom-data`.  Its tiles are cuts taken at
different poses, so the composed image has a scale but no place in the patient:
it is written with `PixelSpacing`, so it can be windowed and measured within a
tile, and with no position or orientation at all, so nothing tries to localize
it.  `cardio` skips such a series when reading rather than misreading it.

The 3D view has a camera rather than an image plane, so it is always recorded as
it looked, whichever DICOM mode is chosen.

The two `dicom-cine-*` formats write the whole loop as a single multi-frame
instance carrying the Cine module, which is what a viewer plays; a series of
single-frame instances it merely sorts, and whether it plays them is up to the
viewer.  The trade is that one such instance carries one position, so a cine
whose plane moves through the cycle -- a snap lock following a valve -- is
written without one, and says so in the log.  Both read straight back into
`cardio` the same way the single-frame series does -- but not every workstation
stores them, so see *What a receiver reads* before sending one anywhere.

When the active volume was read from DICOM, the capture inherits its patient and
study whole, so a derived series lands in the study it came from, keeps its
frame of reference, and cites in `SourceImageSequence` the instances it was cut
through.  A volume read from a file has none of that, and the capture stands
alone under a study of its own rather than borrowing half an identity.

Every UID is minted under `uid_root`.  The default is pydicom's own registered
root, which is nobody else's to assert -- a session opened on it says so at
startup.

A DICOM capture is **refused** when the source series lacks `PatientID`,
`PatientName`, `StudyInstanceUID`, `AccessionNumber` or `StudyDate`, or when
`uid_root` is one the deployment has not registered.  The first three are what
the app would otherwise have to invent, and an instance filed under an invented
identity is not an incomplete instance but a wrong one -- an archive that
accepts it has no way to know.  The last two are honestly empty when they are
absent, and refused anyway: an instance nobody can reconcile with an order is a
stray in the archive.  Setting `research` waives all of it and writes the file
with warnings, which is the right setting for a session whose output goes
nowhere -- but it has to be set deliberately, so nothing reaches an archive
because somebody forgot.

Everything else is reported rather than refused.  The fields a receiving archive
may want -- study ID, frame of reference, patient size and weight, and the rest
-- are checked against what the source series actually carried, and whatever is
missing is named in the log and counted in what the drawer reports.  The capture
is written anyway: whether a given field is needed depends on where the file is
going, which the app does not know.  Which fields sit on which side of that line
is `REQUIRED` and `ADVISORY` in `cardio/capture/preflight.py`, and a site that
files differently moves an entry between them.

### Measuring an area on a cut

Press `m`, click round the thing you want the area of, and right-click to close
it.  The region is drawn on the cut with the area it encloses beside it, and
listed in the Measurements panel.

An area measured on a cut means nothing without the cut, so each region carries
the plane it was traced in and the pose that produced it.  It is drawn only
while the cuts are on that plane and at that frame; move off either and it is
hidden rather than drawn somewhere it does not belong.  *Recall* puts the cuts
back where it was measured.  Panning, rolling or zooming within the plane keep
it on screen, because none of those leaves the plane.

| Key | Does |
| --- | --- |
| `m` | Start or stop tracing |
| `u` | Take back the last point |
| `x` | Give up on the region being traced |
| `e` | Correct the region the drawer has highlighted |

The tile grid takes a region like any of the three panes, on whichever tile the
clicks land in.  A tile is a place a cut is shown rather than a thing a cut
belongs to, so the region is drawn on whichever tile has come to rest on its
plane -- reverse the stack and it walks to the other end of the grid with its
slice.  What a tile region carries, and what *Recall* puts back, is the whole
tile panel and the snap selection its path is taken from: a tile has no pose of
its own, and a grid restored from half its settings would land the region on a
different cut.

Tracing takes the click alone, so window/level, scrolling, panning, rotating
and zooming all go on working while a region is being traced.  Undo and cancel
have buttons in the panel as well; the keys are for a hand already on the mouse.

### Correcting a region already closed

A trace that came out nearly right does not have to be done again.  The pencil
on a region's row opens it for correction, and while it is open:

| Gesture | Does |
| --- | --- |
| Left drag, from a point | Moves that point |
| Left click, on the contour | Adds a point there, on the line |
| Right click, on a point | Takes that point away |

*Done* closes the correction; *Revert* puts the points back where it found
them.  The area follows every change, and the set is marked unsaved the way a
rename marks it.

One region is corrected at a time and only while its cut is showing, which is
why the pencil refuses a region that is off plane and points at *Recall*
instead: the points are moved by pressing on them, and there is nothing to press
on.  Tracing is suspended for the duration -- deliberately, because tracing the
same anatomy twice is how a repeat measurement is taken, and a scheme where a
click near an existing contour edited it would make the second trace of a pair
land inside the first.

A left drag belongs to the region only if it began on one of its points; every
other left drag is the window and level it always was.  A drag is one
correction however many frames it took to make, so it is one line in the console
and one step to undo rather than one of each per pixel travelled.

A region is closed either straight between the points or with a closed spline
through them:

```toml
measurement_contour = "spline"   # polygon (default) or spline
```

That setting is the default for the *next* region.  A region already closed
keeps the curve it was measured as -- a recorded area that changed because a
picker moved would be a number nobody chose -- and is changed deliberately, by
the Straight/Spline buttons on its own row.  Tracing the same points both ways
is the cheapest paired comparison there is.

*Save Measurements* writes the set out under
`<serialization_directory>/measurements/<volume>/`, and a config opens on one:

```toml
measurement_file = "./measurements/CCTA/2026-09-17-12-43-47.toml"
```

The file holds each region's points in the cut's own millimetres, the cut they
were measured on, the area they came to, and the rotation sequence that posed
it -- so a reader gets the number without recomputing it and `cardio` gets the
plane back.  It is written in LPS throughout, whatever index order the session
is spelled in.

As with rotations, a named file is read *over* whatever the config spelled, so
**measurements traced in the app are not saved by saving the session** when a
measurement file is named; *Save Measurements* is what writes them.  Unlike a
rotation file it moves nothing on the way in: a rotation file says where to
look, and a measurement file says where somebody looked, so opening a study
already turned to the last region anybody traced would be a surprise.  Recall
is how it moves the views, and it is asked for.

A capture records what a viewport shows, so `png`, `jpeg` and the
`dicom-rendered` formats carry the regions drawn on it and `dicom-data` does
not -- the same line the crosshairs sit on, and for the same reason: an outline
is not a pixel value.

### Measurements and segmentations as DICOM

*Save Volumetry* writes the curves and the metrics as two CSV files and the
charts as pages, which is what a person opens.  Where the measured volume was
read from DICOM it also writes what a system reads: the segmentation as a DICOM
SEG, one instance per cardiac phase, and the measurements as a TID 1500
Structured Report that points at it.

| Written | Holds |
| --- | --- |
| `timeseries.csv`, `metrics.csv` | The curves and what they came to |
| `segmentation/<i>.dcm` | One Segmentation Storage instance per phase |
| `measurements.dcm` | One Comprehensive 3D SR naming every structure |

A segment is a configured structure rather than a label value, so give each
group in `volumetry.groups` a `code` -- a SNOMED CT concept id -- to say what it
is; one without a code is typed only as tissue, and says so in the log.

The report quotes each structure's largest and smallest volume qualified as a
maximum and a minimum, not as end-diastolic and end-systolic.  Nothing here was
told which phase a frame was acquired at: the reader is the one who declared the
structure a pumping chamber, and the stroke volume and ejection fraction follow
from that declaration rather than from the images.

A volume read from a file gets the tables and nothing else -- a segmentation
object names the images it segments, and a report names the images it is
evidence about, and a NIfTI volume gives neither anything to name.  The same
refusal a capture gets applies here too, and for the same reason: a report says
whose measurements these are.  The tables are written either way.

### What a receiver reads

Everything above is about what a file holds.  What a workstation does with it is
a separate question, and the answer is not the same at every workstation, so it
is worth saying which formats travel.  The two receivers this was checked
against are **Sectra PACS / Workstation 28.1** and **TeraRecon iNtuition 4.9.0**,
read off their published conformance statements rather than guessed at.

| Written | SOP Class | Sectra IDS7 / UniView | TeraRecon iNtuition |
| --- | --- | --- | --- |
| `dicom-rendered` | `…1.1.7` | viewable | viewable |
| `dicom-data` | `…1.1.7` | viewable | viewable |
| `dicom-cine-rendered` | `…1.1.7.4` | viewable | not stored by default |
| `dicom-cine-data` | `…1.1.7.3` | viewable | not stored by default |
| `segmentation/<i>.dcm` | `…1.1.66.4` | stored, not viewable | viewable |
| `measurements.dcm` | `…1.1.88.34` | stored, not viewable | not listed |

| `capture_transfer_syntax` | Sectra | TeraRecon |
| --- | --- | --- |
| `jpeg-2000-lossless` | yes | yes |
| `jpeg-ls-lossless` | not listed | not listed |
| `uncompressed` | yes | yes |

Three things follow.

**Send the single-frame formats.**  `dicom-data` and `dicom-rendered` are the
only ones both receivers read: iNtuition's default storage list has plain
Secondary Capture and neither multi-frame class, and while its note says classes
can be added to the service list at run time, whether it then draws one is not
something the document says.  The `dicom-cine-*` formats are a better object --
one instance, the Cine module, a loop a viewer plays rather than a stack it
sorts -- and are the right choice for a Sectra-only workflow or for reading back
into `cardio`.  They are not the right choice for an export that has to arrive
at both.

**Leave the transfer syntax alone.**  `jpeg-2000-lossless` is the default
because it is the intersection: iNtuition lists it among the lossless syntaxes
it accepts, and Sectra both accepts it and writes its own media exports in it.
Neither document mentions JPEG-LS anywhere, so `jpeg-ls-lossless` is for a local
workflow -- it is a little smaller and several times faster -- and not for
anything being sent.  Both are lossless either way: the pixels that come back
are the pixels that went in, and `LossyImageCompression` says `00` truthfully.

**The overlay series is not redundant with the SEG.**  It looks like it ought to
be -- the SEG is the segmentation as data, and the burned-in overlay is a
picture of it -- but Sectra files a SEG without drawing it, so for a Sectra
reader the overlay is the only place the segmentation appears at all.  iNtuition
draws the SEG and is the reason to write one.  Between them, sending both is
what gets the segmentation in front of either reader.  By the same arithmetic
the Structured Report reaches neither: Sectra stores Comprehensive 3D SR without
displaying it and iNtuition does not list SR at all, so the CSV tables remain
how a person reads the numbers and the SR is there for whatever reads it next.

Two deviations are kept deliberately, so that a validator's complaints are not a
surprise.  A single-frame capture carries `TriggerTime`, which is a standard
attribute but not one the Secondary Capture IOD lists, because it is how
`cardio` puts the phases of a loop back in order on the way in; that makes the
instances a standard extended SOP class, which is a thing the standard provides
for.  And the `dicom-cine-*` objects carry their position and orientation at the
root of the dataset, where those IODs define nothing, rather than in the
functional groups that are the place the standard gives them -- `cardio` reads
them back, and a third-party viewer is entitled to treat such a cine as
unlocalizable.  Neither applies to the single-frame series,
where the geometry sits in the Image Plane module the standard defines for it.

### Running a session again without a browser

Everything `cardio` can be asked to do has a name, and the console records
every one of them as they happen -- including the drawer controls, which are
written down as the `set_state` call that would move them again.  Press the
export button in the console (`` ` `` opens it) and that log is written out as
a script:

```python
#!/usr/bin/env python
# Generated by cardio version 2026.9.3
"""Recorded 2026-09-06 22:10:40."""

# System
import pathlib as pl

# Third Party
import cardio

do = cardio.script(pl.Path(__file__).parent / "2026-09-06-22-10-40.toml")

do.add_rotation(axis="Z")
do.set_window_level_preset(preset=3)
do.set_state(key="tile_cols", value=4)
do.screenshot()
```

Every line below `do` is the line the console printed, character for
character, so a script is the log copied rather than rewritten -- and a line of
either is a line you can type at the prompt.  The prompt takes the `do.` off
your hands if you would rather not type it: `add_rotation(axis='Z')` asks for
the same thing.

A `set_state` key may name a place *inside* a document key rather than the
whole of it, which is how one rotation in a stack of them is a line short
enough to read and to type:

```python
do.set_state(key="mpr_rotation_data.angles_list.1.visible", value=False)
do.set_state(key="mpr_rotation_data.angles_list.0.angle", value=45.0)
```

The path is dotted, and a number in it indexes a list.  A value that is not a
structure is still written whole, so a range slider stays one line holding its
pair rather than two lines saying which end moved.

At the prompt, the up and down arrows walk back through the log and forward
again, narrowed to whatever is already typed: `do.t` and the up arrow steps
through the `do.toggle_` lines alone.  Everything the log holds is walked,
including the drawer controls nobody typed, and a command that was refused --
which is the one most likely to have something in it to fix.

The `.toml` written beside it is the scene as the app *opened*, not as it ended
up, so running the script does the session again rather than doing it twice.
The two are meant to travel together; the script finds the config beside
itself, so it runs from wherever it is run from:

```bash
$ python 2026-09-06-22-10-40.py
```

No browser is involved and none is needed.  Nothing sizes the render windows
without a page, so a session sizes its own; `headless_size` says how big, and
is what the resolution of anything captured this way comes from:

```toml
headless_size = [1920, 1080]
```

From there it is a python script like any other -- loop over a directory of
studies, take the same capture of each, or edit the calls by hand.  `do.session`
is the session itself, for the things that are not actions:

```python
for study in pl.Path("./studies").iterdir():
    do = cardio.script("template.toml", volumes=[{"label": "cine", "directory": study}])
    do.snap_to_centroid()
    do.screenshot()
    do.session.save(study / "as-captured.toml")
```

### Where the rotations come from

A config says where the MPR rotations come from in one of two ways, and never
in both:

```toml
mpr_rotation_file = "./rotations/template.toml"   # read this file
```

```toml
[[mpr_rotation_sequence.angles_list]]             # or spell them here
axis = "Z"
angle = 0.5
```

A named file is read *over* whatever the config spelled, so a config carrying
both would show a sequence that opening it would throw away.  A saved session
therefore writes whichever one the scene was opened with -- naming the file if
there was one, and spelling the sequence if there was not.

This means **rotations you change in the app are not saved by saving the
session** when a rotation file is named: the file goes on being where they come
from.  Press *Save Rotations* to write them out as a rotation file of their own,
under `<serialization_directory>/rotations/<volume>/`, and point the config at
that.  Rotations have their own format, and it is the one that holds them.

A rotation file may say which volume it is for:

```toml
[metadata]
volume_label = "CCTA"
```

which becomes the active volume, so a config can be pointed at a new study by
changing the volume's `directory` alone -- as long as its `label` keeps
matching.  A named file that is not there is refused rather than passed over,
so a mistyped path says so instead of opening on no rotations at all.

### Developing

Ensuring you have all required dependencies:

```bash
$ uv sync --all-extras
```

Pre-commit checklist:

```bash
$ ruff check --fix
$ ruff format
$ ./scripts/pytest-with-memory-cap.sh
```

Uploading:

```bash
$ uv version --bump major # Year
$ uv version --bump minor # Month
$ uv version --bump patch # Day
$ git commit -am "Bump version to $(uv version --short)"
$ git tag -a $(uv version --short) -m "Release $(uv version --short)"
$ git log --oneline -1 --decorate   # the tag should sit on the bump
$ rm -rf dist/
$ uv build --no-sources
$ git push origin main --follow-tags
$ uv publish --token <pypi_api_key>
```

`uv version` edits `pyproject.toml` and `uv.lock` but commits nothing, and a tag
names a commit rather than a working tree -- so the bump has to be committed
before the tag, or the tag lands on the previous commit and the release carries
the old version.  `--follow-tags` pushes the tag along with the branch, so a tag
made after the push does not reach the remote.
