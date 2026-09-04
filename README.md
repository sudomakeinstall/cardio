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
```

```bash
$ cardio --view.layout tile --playback.bpm 75
```

Tile view draws several cuts along the traverse path at once, so it wants a
`[snap]` block in traverse mode to have anything to show.  Reset returns the
playback controls to whatever is written here.

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
