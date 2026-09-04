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
