"""What a capture writes, and what survives the writing.

The phantom is a volume whose geometry is one sentence long, so every
expectation about where a cut landed can be computed rather than recorded.
"""

# System
import asyncio
import pathlib as pl

# Third Party
import itk
import numpy as np
import pydicom as pd
import pytest
import vtk
from vtk.util import numpy_support as vtknp

# Internal
from cardio import Scene, dicom
from cardio.capture import CaptureFormat, Context, Frame, Plane, image_to_array
from cardio.capture.dicom import SecondaryCaptureWriter, SliceWriter, encode
from cardio.capture.formats import WRITERS, writer_for
from cardio.capture.geometry import plane_from_reslice, reslice_axes
from cardio.capture.images import GifWriter, JpegWriter, Mp4Writer, PngWriter
from cardio.capture.mosaic import compose
from cardio.logic.capture import VIEWPORTS, summary_of, written_files
from cardio.orientation import create_vtk_reslice_matrix
from cardio.reslice import VIEW_TRANSFORMS
from tests.test_app_smoke import build_app, build_scene, connect

VOLUME_SIZE = (20, 24, 16)
VOLUME_SPACING = (1.0, 2.0, 3.0)
VOLUME_ORIGIN = (-5.0, 3.0, -7.0)

# Inside the volume, and away from every axis, so a test cannot pass with the
# row and column directions swapped.
POSE_ORIGIN = [5.0, 27.0, 17.0]
POSE_DEGREES = 26.0


def phantom() -> vtk.vtkImageData:
    """A small signed volume, with spacing distinct on all three axes."""
    image = vtk.vtkImageData()
    image.SetDimensions(*VOLUME_SIZE)
    image.SetSpacing(*VOLUME_SPACING)
    image.SetOrigin(*VOLUME_ORIGIN)
    image.AllocateScalars(vtk.VTK_SHORT, 1)

    columns, rows, slices = VOLUME_SIZE
    values = np.arange(slices * rows * columns, dtype=np.int16) % 2000 - 1000
    vtknp.vtk_to_numpy(image.GetPointData().GetScalars())[:] = values
    return image


def turned(degrees: float = POSE_DEGREES) -> np.ndarray:
    radians = np.radians(degrees)
    return np.array(
        [
            [np.cos(radians), -np.sin(radians), 0.0],
            [np.sin(radians), np.cos(radians), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def posed_reslice(image_data, rotation=None, origin=None) -> vtk.vtkImageReslice:
    """One axial cut of the phantom, posed the way the MPR views pose theirs."""
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(image_data)
    reslice.SetOutputDimensionality(2)
    reslice.SetInterpolationModeToLinear()
    reslice.SetBackgroundLevel(-1000.0)
    reslice.AutoCropOutputOn()
    reslice.SetOutputDirection(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    reslice.SetResliceAxes(
        create_vtk_reslice_matrix(
            (turned() if rotation is None else rotation) @ VIEW_TRANSFORMS["axial"],
            POSE_ORIGIN if origin is None else origin,
        )
    )
    reslice.Update()
    return reslice


def context(tmp_path, viewport="axial", **kwargs) -> Context:
    fields = {
        "directory": pl.Path(tmp_path),
        "viewport": viewport,
        "frame_duration": 0.05,
        "window": 800.0,
        "level": 200.0,
        "identity": {"PatientName": "Phantom^Capture", "PatientID": "PHANTOM-1"},
    }
    fields.update(kwargs)
    return Context(**fields)


def rgb_frame(rows=6, columns=8, value=140) -> Frame:
    """A window capture, as VTK hands one over: RGB, bottom row first."""
    image = vtk.vtkImageData()
    image.SetDimensions(columns, rows, 1)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

    array = vtknp.vtk_to_numpy(image.GetPointData().GetScalars())
    array[:] = value
    # A marked bottom row, so a flip is visible.
    array.reshape(rows, columns, 3)[0] = 255
    return Frame(image=image)


# --- the geometry a cut carries ----------------------------------------------


def test_the_direction_cosines_are_the_plane_axes():
    reslice = posed_reslice(phantom())
    axes = reslice_axes(reslice)

    plane = plane_from_reslice(reslice)
    orientation = np.array(plane.location.orientation)

    assert np.allclose(orientation[:3], axes[:3, 0])
    assert np.allclose(orientation[3:], axes[:3, 1])
    assert np.isclose(np.linalg.norm(orientation[:3]), 1.0)
    assert np.isclose(np.linalg.norm(orientation[3:]), 1.0)


def test_the_position_is_where_the_first_pixel_actually_sits():
    """Computed from the pose independently, not read back from the same call."""
    reslice = posed_reslice(phantom())
    axes = reslice_axes(reslice)
    output_origin = np.array(reslice.GetOutput().GetOrigin())

    expected = axes[:3, :3] @ output_origin + axes[:3, 3]

    plane = plane_from_reslice(reslice)
    assert np.allclose(plane.location.position, expected)


def test_the_pose_lands_inside_the_cut_it_posed():
    """Measured in the plane's own coordinates, which is where the grid is.

    Catches a position or a pair of cosines that describe some other rectangle:
    the pose has to fall within the pixels the cut actually holds.
    """
    reslice = posed_reslice(phantom())
    plane = plane_from_reslice(reslice)

    rows, columns = plane.scalars.shape
    orientation = np.array(plane.location.orientation)
    offset = np.array(POSE_ORIGIN) - np.array(plane.location.position)
    row_spacing, column_spacing = plane.pixel_spacing

    along_row = offset @ orientation[:3]
    down_column = offset @ orientation[3:]

    assert 0.0 <= along_row <= (columns - 1) * column_spacing
    assert 0.0 <= down_column <= (rows - 1) * row_spacing
    # The cut lies in the plane, so it has no component off it.
    assert np.isclose(offset @ np.cross(orientation[:3], orientation[3:]), 0.0)


def test_pixel_spacing_is_row_then_column():
    reslice = posed_reslice(phantom())
    spacing = reslice.GetOutput().GetSpacing()

    plane = plane_from_reslice(reslice)

    assert plane.pixel_spacing == (spacing[1], spacing[0])


def test_an_axis_aligned_cut_keeps_the_volume_spacing():
    reslice = posed_reslice(phantom(), rotation=np.eye(3))

    plane = plane_from_reslice(reslice)

    # The axial transform maps the volume's x and y onto the plane's.
    assert np.allclose(plane.pixel_spacing, (VOLUME_SPACING[1], VOLUME_SPACING[0]))


# --- what the values do on the way out ----------------------------------------


def test_integers_that_fit_are_written_through_untouched():
    scalars = np.array([[-1000, 0, 3000]], dtype=np.int16)

    stored, slope, intercept = encode(scalars)

    assert stored.dtype == np.int16
    assert (slope, intercept) == (1.0, 0.0)
    assert np.array_equal(stored, scalars)


def test_floats_are_mapped_so_the_rescale_inverts_them():
    scalars = np.linspace(-3.5, 11.25, 64, dtype=np.float32).reshape(8, 8)

    stored, slope, intercept = encode(scalars)

    assert stored.dtype == np.uint16
    assert np.allclose(stored * slope + intercept, scalars, atol=slope)


def test_wide_integers_fall_back_to_a_rescale():
    scalars = np.array([[0, 200000]], dtype=np.int32)

    stored, slope, intercept = encode(scalars)

    assert slope != 1.0
    assert np.allclose(stored * slope + intercept, scalars, atol=slope)


# --- a slice series -----------------------------------------------------------


def written(directory: pl.Path) -> list[pd.dataset.Dataset]:
    return [pd.dcmread(path) for path in sorted(directory.glob("*.dcm"))]


def write_slices(tmp_path, frames: int = 3, viewport="axial") -> list:
    reslice = posed_reslice(phantom())
    plane = plane_from_reslice(reslice)

    writer = SliceWriter(context(tmp_path, viewport))
    for index in range(frames):
        writer.add(index, Frame(image=rgb_frame().image, plane=plane))
    writer.close()

    return written(pl.Path(tmp_path) / viewport)


def test_a_slice_series_holds_one_instance_per_phase(tmp_path):
    datasets = write_slices(tmp_path, frames=4)

    assert len(datasets) == 4
    assert len({d.SeriesInstanceUID for d in datasets}) == 1
    assert [d.InstanceNumber for d in datasets] == [1, 2, 3, 4]


def test_the_slice_pixels_are_the_reslice_values(tmp_path):
    expected = plane_from_reslice(posed_reslice(phantom())).scalars

    dataset = write_slices(tmp_path, frames=1)[0]
    values = dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept

    assert dataset.PhotometricInterpretation == "MONOCHROME2"
    assert np.array_equal(values, expected)


def test_the_window_is_a_tag_rather_than_applied(tmp_path):
    dataset = write_slices(tmp_path, frames=1)[0]

    assert float(dataset.WindowWidth) == 800.0
    assert float(dataset.WindowCenter) == 200.0
    # Applying it would have clipped the values to the window.
    assert dataset.pixel_array.min() < 200.0 - 800.0 / 2


def test_a_slice_says_where_it_is(tmp_path):
    plane = plane_from_reslice(posed_reslice(phantom()))

    dataset = write_slices(tmp_path, frames=1)[0]

    assert np.allclose(dataset.ImageOrientationPatient, plane.location.orientation)
    assert np.allclose(dataset.ImagePositionPatient, plane.location.position)
    assert np.allclose(dataset.PixelSpacing, plane.pixel_spacing)


def test_the_study_the_capture_belongs_to_is_carried_through(tmp_path):
    dataset = write_slices(tmp_path, frames=1)[0]

    assert str(dataset.PatientName) == "Phantom^Capture"
    assert dataset.PatientID == "PHANTOM-1"


def test_a_viewport_showing_nothing_writes_nothing(tmp_path):
    writer = SliceWriter(context(tmp_path))
    writer.add(0, Frame(image=rgb_frame().image, plane=None))
    writer.close()

    assert written(pl.Path(tmp_path) / "axial") == []


def test_a_slice_series_reads_back_as_the_frames_it_was_written_from(tmp_path):
    write_slices(tmp_path, frames=3)

    frames = dicom.read_series(pl.Path(tmp_path) / "axial")

    assert len(frames) == 3


# --- a mosaic -----------------------------------------------------------------


def poses(count: int) -> list:
    """Cuts stepped along z and turned a little more at each step."""
    return [
        ([POSE_ORIGIN[0], POSE_ORIGIN[1], POSE_ORIGIN[2] + i], turned(10.0 * i))
        for i in range(count)
    ]


def mosaic_of(rows: int, columns: int) -> Plane:
    return compose(
        phantom(),
        poses(rows * columns),
        VIEW_TRANSFORMS["axial"],
        rows,
        columns,
    )


def test_a_mosaic_has_one_spacing_and_no_place():
    plane = mosaic_of(2, 3)

    assert plane.location is None
    assert plane.pixel_spacing[0] == plane.pixel_spacing[1]
    assert plane.pixel_spacing[0] == min(VOLUME_SPACING)


def test_a_mosaic_is_the_grid_it_was_asked_for():
    """The same six cuts, laid out two ways, tile to the same size."""
    wide = compose(phantom(), poses(6), VIEW_TRANSFORMS["axial"], 1, 6)
    grid = compose(phantom(), poses(6), VIEW_TRANSFORMS["axial"], 2, 3)

    tile_rows, tile_columns = wide.scalars.shape[0], wide.scalars.shape[1] // 6
    assert grid.scalars.shape == (2 * tile_rows, 3 * tile_columns)


def test_a_mosaic_keeps_the_volume_values():
    plane = mosaic_of(1, 1)

    assert plane.scalars.dtype == np.int16
    assert plane.scalars.max() > 0


def test_the_tiles_are_laid_out_row_major_from_the_top_left():
    """Four cuts down a column are the same four cuts along a row, in order."""
    tall = compose(phantom(), poses(4), VIEW_TRANSFORMS["axial"], 4, 1)
    wide = compose(phantom(), poses(4), VIEW_TRANSFORMS["axial"], 1, 4)

    tile_rows, tile_columns = tall.scalars.shape[0] // 4, tall.scalars.shape[1]

    for index in range(4):
        down = tall.scalars[index * tile_rows : (index + 1) * tile_rows, :]
        along = wide.scalars[:, index * tile_columns : (index + 1) * tile_columns]
        assert np.array_equal(down, along)

    # Distinct cuts, or the check above would hold for any layout at all.
    assert not np.array_equal(
        tall.scalars[:tile_rows], tall.scalars[tile_rows : 2 * tile_rows]
    )


def test_a_mosaic_is_written_without_a_position(tmp_path):
    writer = SliceWriter(context(tmp_path, "tile"))
    writer.add(0, Frame(image=rgb_frame().image, plane=mosaic_of(1, 2)))
    writer.close()

    dataset = written(pl.Path(tmp_path) / "tile")[0]

    assert "ImagePositionPatient" not in dataset
    assert "ImageOrientationPatient" not in dataset
    assert dataset.PixelSpacing == [min(VOLUME_SPACING)] * 2


def test_a_mosaic_series_is_skipped_rather_than_misread(tmp_path):
    """The omission is the point: nothing may place these pixels in a patient."""
    writer = SliceWriter(context(tmp_path, "tile"))
    writer.add(0, Frame(image=rgb_frame().image, plane=mosaic_of(1, 2)))
    writer.close()

    assert dicom.scan(pl.Path(tmp_path) / "tile") == []


# --- what the viewport looked like --------------------------------------------


def test_a_window_capture_is_turned_the_right_way_up():
    array = image_to_array(rgb_frame().image)

    assert array.shape == (6, 8, 3)
    # VTK hands over the bottom row first; it belongs last.
    assert np.all(array[-1] == 255)


def test_a_secondary_capture_holds_the_picture(tmp_path):
    writer = SecondaryCaptureWriter(context(tmp_path, "vr"))
    writer.add(0, rgb_frame())
    writer.close()

    dataset = written(pl.Path(tmp_path) / "vr")[0]

    assert dataset.PhotometricInterpretation == "RGB"
    assert (dataset.Rows, dataset.Columns) == (6, 8)
    assert np.array_equal(dataset.pixel_array, image_to_array(rgb_frame().image))
    assert "ImagePositionPatient" not in dataset


# --- stills and animations ----------------------------------------------------


@pytest.mark.parametrize(
    "writer_class,name",
    [(PngWriter, "0.png"), (JpegWriter, "0.jpg")],
)
def test_a_still_is_one_file_per_frame(tmp_path, writer_class, name):
    writer = writer_class(context(tmp_path, "vr"))
    writer.add(0, rgb_frame())
    writer.add(1, rgb_frame())
    writer.close()

    directory = pl.Path(tmp_path) / "vr"
    assert (directory / name).exists()
    assert len(list(directory.iterdir())) == 2


def test_a_gif_is_one_file_holding_every_frame(tmp_path):
    PIL = pytest.importorskip("PIL.Image")

    writer = GifWriter(context(tmp_path, "vr"))
    for index in range(4):
        writer.add(index, rgb_frame(value=40 * index))
    writer.close()

    with PIL.open(pl.Path(tmp_path) / "vr.gif") as gif:
        assert gif.n_frames == 4
        assert gif.size == (8, 6)


def test_an_mp4_is_one_file_holding_every_frame(tmp_path):
    iio = pytest.importorskip("imageio_ffmpeg")

    writer = Mp4Writer(context(tmp_path, "vr"))
    for index in range(4):
        writer.add(index, rgb_frame(value=40 * index))
    writer.close()

    path = pl.Path(tmp_path) / "vr.mp4"
    assert path.stat().st_size > 0
    assert iio.count_frames_and_secs(str(path))[0] == 4


def test_an_odd_sized_frame_still_encodes(tmp_path):
    """yuv420p needs even dimensions, so the odd row and column are dropped."""
    pytest.importorskip("imageio_ffmpeg")

    writer = Mp4Writer(context(tmp_path, "vr"))
    writer.add(0, rgb_frame(rows=7, columns=9))
    writer.close()

    assert (pl.Path(tmp_path) / "vr.mp4").stat().st_size > 0


# --- choosing between them ----------------------------------------------------


@pytest.mark.parametrize("fmt", list(CaptureFormat))
def test_every_format_resolves_to_a_writer(fmt):
    assert fmt in WRITERS


@pytest.mark.parametrize("fmt", list(CaptureFormat))
def test_every_format_can_write_the_volume_render(tmp_path, fmt):
    """The 3D view has no plane, so a data format has to fall back for it."""
    writer = writer_for(fmt, context(tmp_path, "vr", has_plane=False))
    writer.add(0, rgb_frame())
    writer.close()

    assert any(pl.Path(tmp_path).rglob("*"))


def test_a_data_capture_of_the_volume_render_records_the_picture(tmp_path):
    writer = writer_for(
        CaptureFormat.DICOM_DATA, context(tmp_path, "vr", has_plane=False)
    )

    assert isinstance(writer, SecondaryCaptureWriter)


def test_a_data_capture_of_a_cut_records_the_values(tmp_path):
    writer = writer_for(CaptureFormat.DICOM_DATA, context(tmp_path, "axial"))

    assert isinstance(writer, SliceWriter)


# --- which viewports a capture may write --------------------------------------


def running(scene, layout: str):
    """A built app showing ``layout``.

    ``connect`` is what arms the change listeners, so the layout has to be set
    after it for the availability to follow.
    """
    server, scene, logic, _ui = build_app(scene)
    connect(server)

    with server.state:
        server.state.maximized_view = layout

    return server, scene, logic


def built(tmp_path, layout: str = "", **overrides):
    """A whole app, the way the smoke tests build one, in a chosen layout."""
    return running(
        build_scene(
            tmp_path,
            serialization_directory=tmp_path / "out",
            active_volume_label="vol",
            **overrides,
        ),
        layout,
    )


def tick(server, *names):
    """Tick exactly the named viewports."""
    for name in VIEWPORTS:
        server.state[f"screenshot_viewport_{name}"] = name in names


@pytest.mark.parametrize(
    "layout,available",
    [
        ("", {"axial", "coronal", "sagittal", "vr"}),
        ("volume", {"vr"}),
        ("axial", {"axial"}),
        ("tile", {"tile"}),
    ],
)
def test_only_the_viewports_on_screen_are_offered(tmp_path, layout, available):
    server, _, logic = built(tmp_path, layout)

    assert set(server.state.capture_available) == available
    assert logic.capture.available == available


def test_the_offer_follows_the_layout(tmp_path):
    """The drawer greys checkboxes from this, so it has to track the switch."""
    server, _, _ = built(tmp_path, "")

    with server.state:
        server.state.maximized_view = "tile"

    assert set(server.state.capture_available) == {"tile"}


@pytest.mark.parametrize(
    "layout,ticked,expected",
    [
        ("tile", ["axial"], set()),
        ("tile", ["tile", "axial"], {"tile"}),
        # The two that used to be written from a window nobody was looking at.
        ("tile", ["vr"], set()),
        ("axial", ["vr", "coronal"], set()),
        ("", ["tile"], set()),
        ("", ["axial", "vr"], {"axial", "vr"}),
    ],
)
def test_an_off_screen_viewport_is_never_captured(tmp_path, layout, ticked, expected):
    server, _, logic = built(tmp_path, layout)
    tick(server, *ticked)

    assert set(logic.capture.selected_windows()) == expected


# --- a capture that cannot write anything -------------------------------------


def capture(logic):
    """Run the capture controller to completion."""

    async def drive():
        logic.capture.screenshot()
        await asyncio.gather(
            *[t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        )

    asyncio.run(drive())


def camera_position(scene):
    return scene.renderer.GetActiveCamera().GetPosition()


def test_a_capture_with_nothing_on_screen_does_not_run_the_cine(tmp_path):
    """The reported fault: it wrote nothing, but stepped the whole cine first.

    Watched through the camera rather than through ``frame``: the rotation is
    the one thing the loop does on every pass whatever the scene holds, whereas
    a phantom with a single frame would sit at frame zero either way.
    """
    server, scene, logic = built(tmp_path, "tile")
    tick(server, "axial")
    with server.state:
        server.state.rotating = True

    before = camera_position(scene)
    capture(logic)

    assert camera_position(scene) == before
    assert not (tmp_path / "out" / "screenshots").exists()


def test_a_capture_that_can_write_does_run_the_cine(tmp_path):
    """The other half: without it the check above would pass on a broken loop."""
    server, scene, logic = built(tmp_path, "volume")
    tick(server, "vr")
    with server.state:
        server.state.rotating = True

    before = camera_position(scene)
    capture(logic)

    assert camera_position(scene) != before


def test_a_capture_with_nothing_on_screen_says_so(tmp_path):
    server, _, logic = built(tmp_path, "tile")
    tick(server, "axial")

    capture(logic)

    assert server.state.capture_ok is False
    assert "on screen" in server.state.capture_summary
    assert server.state.capture_saved_at


def test_a_capture_that_writes_reports_where(tmp_path):
    server, _scene, logic = built(tmp_path, "volume")
    tick(server, "vr")

    capture(logic)

    folder = max((tmp_path / "out" / "screenshots").iterdir())
    assert server.state.capture_ok is True
    assert server.state.capture_summary == f"Captured vr to {folder.name}"
    assert server.state.capture_running is False


# --- counting what landed -----------------------------------------------------


def test_a_folder_of_stills_counts_as_written(tmp_path):
    folder = tmp_path / "vr"
    folder.mkdir()
    (folder / "0.png").write_bytes(b"x")

    assert len(written_files(tmp_path, "vr")) == 1


def test_an_animation_counts_as_written(tmp_path):
    """One file however many frames went into it."""
    (tmp_path / "tile.gif").write_bytes(b"x")

    assert len(written_files(tmp_path, "tile")) == 1


def test_a_viewport_that_wrote_nothing_is_not_reported_as_saved(tmp_path):
    """An empty folder is what a data capture of a planeless viewport leaves."""
    (tmp_path / "tile").mkdir()

    assert written_files(tmp_path, "tile") == []
    assert summary_of([], tmp_path) == "Capture wrote nothing"


# --- the frame a cine capture reads -------------------------------------------


def cine_app(tmp_path, frames: int = 3, **overrides):
    """An app whose volume has several frames, so the cine has somewhere to go."""
    for index in range(frames):
        array = np.zeros((8, 8, 8), dtype=np.float32)
        array[index : index + 2, 1:4, 1:4] = 1.0
        itk.imwrite(itk.image_from_array(array), str(tmp_path / f"v{index}.nii.gz"))

    scene = Scene(
        volumes=[
            {
                "label": "vol",
                "directory": tmp_path,
                "file_paths": [f"v{index}.nii.gz" for index in range(frames)],
            }
        ],
        serialization_directory=tmp_path / "out",
        active_volume_label="vol",
        **overrides,
    )
    return running(scene, "axial")


def test_a_cine_reads_only_cuts_that_have_been_posed(tmp_path, monkeypatch):
    """The data has to come from the frame on screen, not the one next up.

    Writing ``frame`` reposes the views when the state block flushes, which is
    after the pass that captures them.  A plane taken after the increment
    therefore belongs to a frame whose reslices may not exist yet, and asking
    for them builds a set centred on the image: the capture would record an
    axis-aligned cut through the middle instead of the one the user posed.
    """
    server, scene, logic = cine_app(tmp_path, capture_format="dicom-data")
    tick(server, "axial")
    with server.state:
        server.state.incrementing = True

    volume = scene.volumes[0]
    posed = []
    read_plane = logic.capture.plane_for

    def spy(viewport, frame):
        posed.append(frame in volume._mpr_actors)
        return read_plane(viewport, frame)

    monkeypatch.setattr(logic.capture, "plane_for", spy)
    capture(logic)

    assert posed
    assert all(posed)
