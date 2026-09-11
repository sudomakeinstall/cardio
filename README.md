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

### Opening in a particular view

The layout and theme the app opens in, and where the playback controls start:

```toml
[view]
layout = "tile"     # quad (default), volume, axial, coronal, sagittal, tile
theme = "dark"      # selects between the two [background] colours
camera_lock = "LL"  # MPR view the volume rendering's camera follows, or "free"
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
plane = "axial"     # plane the parallel sources cut in: axial, coronal, sagittal
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
screenshot_viewports = ["axial", "coronal", "tile"]    # dicom-rendered, dicom-data
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

`dicom-data` is the one that keeps the measurements.  For the MPR views it
writes the resliced plane itself: the original values, so a viewer reads the
same numbers the volume holds, with the window and level as `WindowWidth` and
`WindowCenter` tags rather than applied to the pixels, and the true
`ImageOrientationPatient`, `ImagePositionPatient` and `PixelSpacing` of the cut.
Such a series reads straight back into `cardio`.

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

When the active volume was read from DICOM, the capture inherits its patient and
study, so a derived series lands in the study it came from.

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

do = cardio.script(pl.Path(__file__).parent / '2026-09-06-22-10-40.toml')

do.add_rotation(axis='Z')
do.set_window_level_preset(preset=3)
do.set_state(key='tile_cols', value=4)
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
do.set_state(key='mpr_rotation_data.angles_list.1.visible', value=False)
do.set_state(key='mpr_rotation_data.angles_list.0.angle', value=45.0)
```

The path is dotted, and a number in it indexes a list.  A value that is not a
structure is still written whole, so a range slider stays one line holding its
pair rather than two lines saying which end moved.

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
$ pytest -v
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
