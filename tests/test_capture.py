"""What a capture writes, and what survives the writing.

The phantom is a volume whose geometry is one sentence long, so every
expectation about where a cut landed can be computed rather than recorded.
"""

# System
import logging
import pathlib as pl
from importlib.metadata import version

# Third Party
import itk
import numpy as np
import pydantic as pc
import pydicom as pd
import pytest
import vtk
from pydicom.pixels import apply_modality_lut
from vtk.util import numpy_support as vtknp

# Internal
from cardio import Scene, dicom
from cardio.capture import (
    CaptureFormat,
    Context,
    Equipment,
    Frame,
    Identity,
    Plane,
    image_to_array,
    preflight,
    uid,
)
from cardio.capture.banner import (
    band_height,
    coverage,
    stamp_image,
    stamp_rgb,
    stamp_scalars,
)
from cardio.capture.dicom import SecondaryCaptureWriter, SliceWriter, encode
from cardio.capture.formats import WRITERS, writer_for, writes_series
from cardio.capture.geometry import plane_from_reslice, reslice_axes
from cardio.capture.images import GifWriter, JpegWriter, Mp4Writer, PngWriter
from cardio.capture.mosaic import compose
from cardio.capture.series import (
    DESCRIPTION_LIMIT,
    Series,
    SeriesTags,
    describe,
    repeated_numbers,
)
from cardio.logic.capture import VIEWPORTS, summary_of, written_files
from cardio.orientation import create_vtk_reslice_matrix
from cardio.reslice import VIEW_TRANSFORMS
from cardio.session import until_settled
from tests.phantoms import write_cine_series
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


REFERENCE_FRAME_UID = "1.2.826.0.1.3680043.8.498.99999999999999999999999999999999"


def reference_dataset(modality: str = "MR") -> pd.dataset.Dataset:
    """One source instance's header, as the reader hands one over.

    The writers copy the patient and the study off this whole, so a test that
    wants to say what a capture inherited says it here.
    """
    dataset = pd.dataset.Dataset()
    dataset.SOPClassUID = pd.uid.MRImageStorage
    dataset.SOPInstanceUID = pd.uid.generate_uid()
    dataset.StudyInstanceUID = pd.uid.generate_uid()
    dataset.SeriesInstanceUID = pd.uid.generate_uid()
    dataset.FrameOfReferenceUID = REFERENCE_FRAME_UID
    dataset.PatientName = "Phantom^Capture"
    dataset.PatientID = "PHANTOM-1"
    dataset.StudyDate = "20260101"
    dataset.Modality = modality
    return dataset


def identity(modality: str = "MR") -> Identity:
    dataset = reference_dataset(modality)
    return Identity(
        source_images=(dataset,),
        frame_of_reference=REFERENCE_FRAME_UID,
        modality=modality,
    )


def context(tmp_path, viewport="axial", **kwargs) -> Context:
    fields = {
        "directory": pl.Path(tmp_path),
        "viewport": viewport,
        "frame_duration": 0.05,
        "window": 800.0,
        "level": 200.0,
        "identity": identity(),
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


def test_unsigned_integers_that_fit_are_written_through_untouched():
    scalars = np.array([[0, 40, 3000]], dtype=np.uint16)

    stored, slope, intercept = encode(scalars)

    assert stored.dtype == np.uint16
    assert (slope, intercept) == (1.0, 0.0)
    assert np.array_equal(stored, scalars)


def test_signed_integers_keep_their_values_under_an_intercept():
    """Secondary Capture carries no signed pixels, so the sign becomes an offset."""
    scalars = np.array([[-1000, 0, 3000]], dtype=np.int16)

    stored, slope, intercept = encode(scalars)

    assert stored.dtype == np.uint16
    assert slope == 1.0
    assert intercept == -1000.0
    assert np.array_equal(stored + intercept, scalars)


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


def write_slices(tmp_path, frames: int = 3, viewport="axial", **naming) -> list:
    reslice = posed_reslice(phantom())
    plane = plane_from_reslice(reslice)

    writer = SliceWriter(context(tmp_path, viewport, **naming))
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
    # The window is in the values' own units, which is what the stored pixels
    # mean once the modality LUT is applied.  Applying the window instead would
    # have clipped them to it.
    values = apply_modality_lut(dataset.pixel_array, dataset)
    assert values.min() < 200.0 - 800.0 / 2


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


# --- the banner on the lower margin -------------------------------------------

BANNER = "NOT FOR CLINICAL USE"

# Wide enough that the default font fits the banner without shrinking to
# nothing, and shallow enough that the band is the larger part of the result.
BLOCK_ROWS = 40
BLOCK_COLUMNS = 160


def rgb_block(components=3, value=90) -> np.ndarray:
    """A flat picture, so anything the band puts down is the only variation."""
    return np.full((BLOCK_ROWS, BLOCK_COLUMNS, components), value, np.uint8)


def test_a_banner_adds_a_band_below_the_picture():
    block = rgb_block()
    stamped = stamp_rgb(block, BANNER)

    assert stamped.shape == (BLOCK_ROWS + band_height(BLOCK_COLUMNS), BLOCK_COLUMNS, 3)
    assert np.array_equal(stamped[:BLOCK_ROWS], block)


def test_the_band_carries_lettering():
    band = stamp_rgb(rgb_block(), BANNER)[BLOCK_ROWS:]

    assert band.min() < band.max()


def test_alpha_is_opaque_across_the_band():
    band = stamp_rgb(rgb_block(components=4), BANNER)[BLOCK_ROWS:]

    assert np.all(band[..., 3] == 255)


def test_a_banner_too_long_for_the_width_is_still_a_band_of_it():
    """The fit gives up rather than widening the picture to suit the caption."""
    band = coverage(BANNER * 20, BLOCK_COLUMNS)

    assert band.shape == (band_height(BLOCK_COLUMNS), BLOCK_COLUMNS)


def test_the_band_stays_within_the_values_the_cut_holds():
    scalars = (np.arange(BLOCK_ROWS * BLOCK_COLUMNS) % 500 - 200).astype(np.int16)
    scalars = scalars.reshape(BLOCK_ROWS, BLOCK_COLUMNS)
    stamped = stamp_scalars(scalars, BANNER)

    assert stamped.dtype == scalars.dtype
    assert stamped.shape == (BLOCK_ROWS + band_height(BLOCK_COLUMNS), BLOCK_COLUMNS)
    assert np.array_equal(stamped[:BLOCK_ROWS], scalars)

    band = stamped[BLOCK_ROWS:]
    assert band.min() >= scalars.min()
    assert band.max() <= scalars.max()
    assert band.min() < band.max()


def test_a_stamped_capture_keeps_the_picture_the_right_way_up():
    stamped = stamp_image(rgb_frame(rows=6, columns=BLOCK_COLUMNS).image, BANNER)
    array = image_to_array(stamped)

    assert array.shape == (6 + band_height(BLOCK_COLUMNS), BLOCK_COLUMNS, 3)
    # The marked bottom row of the picture, with the band below it rather than
    # over it.
    assert np.all(array[5] == 255)


def test_an_empty_banner_adds_nothing():
    block = rgb_block()
    assert np.array_equal(stamp_rgb(block, ""), block)

    scalars = np.arange(120, dtype=np.int16).reshape(10, 12)
    assert np.array_equal(stamp_scalars(scalars, ""), scalars)

    image = rgb_frame().image
    assert stamp_image(image, "") is image


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


def test_the_charts_have_no_cut_behind_them_to_capture(tmp_path):
    """Volumetry draws a measurement of the labels, not a resampling of them.

    So a data capture of it has to fall back to recording what it looked like,
    which it does by not being named among the viewports a plane can be had
    from.  Writing one would mean inventing a pixel spacing and a position for
    an axis and a curve.
    """
    _, _, logic = built(tmp_path, "volumetry")

    assert "volumetry" not in logic.capture.plane_sources
    assert logic.capture.plane_for("volumetry", 0) is None


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
        ("volumetry", {"volumetry"}),
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
        ("volumetry", ["volumetry"], {"volumetry"}),
        ("volumetry", ["axial", "vr"], set()),
    ],
)
def test_an_off_screen_viewport_is_never_captured(tmp_path, layout, ticked, expected):
    server, _, logic = built(tmp_path, layout)
    tick(server, *ticked)

    assert set(logic.capture.selected_windows()) == expected


# --- a capture that cannot write anything -------------------------------------


def capture(logic):
    """Run the capture controller to completion.

    The same wait a headless session does, which is where it now lives.
    """
    until_settled(logic.capture.screenshot)


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


# --- naming a series ----------------------------------------------------------


def test_every_viewport_opens_on_a_number_of_its_own():
    """Unconfigured, a capture of several viewports is several series.

    Numbering them alike would be the one default a study cannot be sorted
    by, so the defaults are distinct rather than all 1.
    """
    numbers = [SeriesTags().of(name).number for name in VIEWPORTS]

    assert len(set(numbers)) == len(VIEWPORTS)


def test_a_series_carries_the_number_it_was_given(tmp_path):
    dataset = write_slices(tmp_path, frames=1, series_number=407)[0]

    assert dataset.SeriesNumber == 407


def test_a_named_series_is_called_what_it_was_named(tmp_path):
    dataset = write_slices(tmp_path, frames=1, series_description="Cine SAX")[0]

    assert dataset.SeriesDescription == "Cine SAX"


def test_an_unnamed_series_says_which_viewport_it_came_off(tmp_path):
    """The fallback still has to be tellable from the series written beside it."""
    dataset = write_slices(tmp_path, frames=1)[0]

    assert dataset.SeriesDescription == "cardio axial (reformat)"


def rendered(tmp_path, **naming) -> pd.dataset.Dataset:
    writer = SecondaryCaptureWriter(context(tmp_path, "vr", **naming))
    writer.add(0, rgb_frame())
    writer.close()
    return written(pl.Path(tmp_path) / "vr")[0]


def test_a_rendered_series_is_named_the_same_way(tmp_path):
    """Both writers, or a capture would be named by which one it went through."""
    assert rendered(tmp_path).SeriesDescription == "cardio vr (rendered)"
    assert rendered(tmp_path, series_description="As shown").SeriesDescription == (
        "As shown"
    )


@pytest.mark.parametrize("kind", ["rendered", "reformat", "mosaic"])
def test_an_unnamed_series_is_named_after_what_it_holds(kind):
    assert describe("tile", kind, "") == f"cardio tile ({kind})"


def test_a_name_that_was_given_is_the_whole_of_the_name():
    """Nothing appended: what was asked for is what a viewer lists."""
    assert describe("tile", "mosaic", "Cine SAX") == "Cine SAX"


def test_two_viewports_written_as_one_number_are_reported():
    assert repeated_numbers({"axial": 5, "coronal": 5, "vr": 6}) == {
        5: ["axial", "coronal"]
    }


def test_numbers_that_differ_are_not_reported():
    assert repeated_numbers({"axial": 5, "coronal": 6}) == {}


def test_a_description_longer_than_dicom_carries_is_refused():
    """Rejected by the receiver rather than shortened, so refuse it here."""
    Series(description="x" * DESCRIPTION_LIMIT)

    with pytest.raises(pc.ValidationError):
        Series(description="x" * (DESCRIPTION_LIMIT + 1))


def test_a_viewport_that_does_not_exist_is_refused():
    """A misspelled one would otherwise quietly name nothing."""
    with pytest.raises(pc.ValidationError):
        SeriesTags(oblique={"number": 3})


def test_only_the_dicom_formats_have_a_series_to_name():
    naming = {fmt for fmt in CaptureFormat if writes_series(fmt)}

    assert naming == {CaptureFormat.DICOM_RENDERED, CaptureFormat.DICOM_DATA}


def captured_series(tmp_path, viewport="axial") -> pd.dataset.Dataset:
    """The first instance of the one series a capture just wrote."""
    folder = max((tmp_path / "out" / "screenshots").iterdir())
    return written(folder / viewport)[0]


def exporting(tmp_path, **overrides):
    """An app set up to write one DICOM series, off the axial view."""
    server, _, logic = built(
        tmp_path, "axial", capture_format="dicom-data", **overrides
    )
    tick(server, "axial")
    return server, logic


def test_a_capture_is_written_as_the_series_the_scene_named(tmp_path):
    _, logic = exporting(
        tmp_path,
        capture_series={"axial": {"number": 407, "description": "Cine SAX"}},
    )

    capture(logic)

    dataset = captured_series(tmp_path)
    assert dataset.SeriesNumber == 407
    assert dataset.SeriesDescription == "Cine SAX"


def test_the_configured_naming_reaches_the_drawer(tmp_path):
    server, _, _ = built(
        tmp_path,
        capture_series={"axial": {"number": 400, "description": "Cine SAX"}},
    )

    assert server.state.capture_series_number_axial == 400
    assert server.state.capture_series_description_axial == "Cine SAX"


def test_what_the_drawer_holds_is_what_the_capture_is_written_as(tmp_path):
    """Retyped rather than reconfigured, and the number field hands back text.

    A ``type="number"`` field's v-model is a string, so a number typed into
    one has to reach ``SeriesNumber`` as the number it reads as.
    """
    server, logic = exporting(tmp_path)
    with server.state:
        server.state.capture_series_number_axial = "512"
        server.state.capture_series_description_axial = "Retyped"

    capture(logic)

    dataset = captured_series(tmp_path)
    assert dataset.SeriesNumber == 512
    assert dataset.SeriesDescription == "Retyped"


def test_a_number_field_left_empty_falls_back_on_what_was_configured(tmp_path):
    """Clearing the field is a state to pass through, not a capture to refuse."""
    server, _, logic = built(tmp_path, "volume", capture_series={"vr": {"number": 409}})
    with server.state:
        server.state.capture_series_number_vr = ""

    assert logic.capture.series_for("vr") == (409, "")


# --- the banner a capture is written with -------------------------------------


def banner_root(tmp_path, name: str) -> pl.Path:
    """A root of its own, so two captures can be compared side by side."""
    root = pl.Path(tmp_path) / name
    root.mkdir()
    return root


def test_a_capture_of_the_values_carries_the_band_and_says_so(tmp_path):
    """The cut keeps every row it was measured from, and gains the band below."""
    plain = banner_root(tmp_path, "plain")
    _, logic = exporting(plain)
    capture(logic)
    before = captured_series(plain)

    marked = banner_root(tmp_path, "marked")
    _, logic = exporting(marked, capture_banner=BANNER)
    capture(logic)
    after = captured_series(marked)

    assert before.BurnedInAnnotation == "NO"
    assert after.BurnedInAnnotation == "YES"
    assert after.Columns == before.Columns
    assert after.Rows == before.Rows + band_height(before.Columns)
    assert np.array_equal(after.pixel_array[: before.Rows], before.pixel_array)


def test_the_configured_banner_reaches_the_drawer(tmp_path):
    server, _, _ = built(tmp_path, capture_banner=BANNER)

    assert server.state.capture_banner == BANNER


def test_what_the_drawer_holds_is_the_banner_that_is_written(tmp_path):
    """Retyped rather than reconfigured, as the series naming is."""
    server, logic = exporting(tmp_path)
    with server.state:
        server.state.capture_banner = BANNER

    capture(logic)

    assert captured_series(tmp_path).BurnedInAnnotation == "YES"


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


# --- what a receiver is entitled to expect -------------------------------------


def elements(dataset):
    """Every element of a dataset, sequences included."""
    for element in dataset:
        yield element
        if element.VR == "SQ":
            for item in element.value:
                yield from elements(item)


def decimal_strings(dataset):
    """Every Decimal String value in a dataset, sequences included."""
    for element in elements(dataset):
        if element.VR != "DS":
            continue
        values = element.value
        if not isinstance(values, pd.multival.MultiValue):
            values = [values]
        for value in values:
            yield element.keyword, str(value)


def test_no_decimal_string_overruns_the_sixteen_characters_it_has(tmp_path):
    """An oblique cut is where a float printed in full overruns the VR."""
    dataset = write_slices(tmp_path, frames=1)[0]

    written_out = list(decimal_strings(dataset))

    assert written_out
    assert [pair for pair in written_out if len(pair[1]) > 16] == []


def encoded_length(element) -> int:
    """How many bytes one element takes up, written as the file writes it."""
    buffer = pd.filebase.DicomBytesIO()
    buffer.is_little_endian = True
    buffer.is_implicit_VR = False
    pd.filewriter.write_data_element(buffer, element)
    return len(buffer.getvalue())


def test_every_element_is_written_at_an_even_length(tmp_path):
    """An odd-length element is malformed however leniently it is read back."""
    datasets = write_slices(tmp_path, frames=1) + [
        # Sized so that rows * columns * 3 is odd, which is where an unpadded
        # colour capture goes wrong.
        _rendered_instance(tmp_path / "rendered", rows=7, columns=9)
    ]

    odd = [
        element.keyword
        for dataset in datasets
        for element in elements(dataset)
        if element.VR != "SQ" and encoded_length(element) % 2
    ]

    assert odd == []


def _rendered_instance(directory, rows=7, columns=9) -> pd.dataset.Dataset:
    """One instance off the writer that records what a viewport looked like."""
    writer = SecondaryCaptureWriter(context(directory, "vr"))
    writer.add(0, rgb_frame(rows=rows, columns=columns))
    writer.close()
    return written(pl.Path(directory) / "vr")[0]


# The elements a Secondary Capture instance must carry, empty or not.
REQUIRED = (
    "SpecificCharacterSet",
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "StudyID",
    "StudyDate",
    "StudyTime",
    "ReferringPhysicianName",
    "AccessionNumber",
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "Modality",
    "SeriesNumber",
    "InstanceNumber",
    "Manufacturer",
    "ConversionType",
    "ContentDate",
    "ContentTime",
    "PatientOrientation",
    "ImageType",
)


@pytest.mark.parametrize("viewport", ["axial", "vr"])
def test_every_required_element_is_present_even_where_it_is_empty(tmp_path, viewport):
    """Type 2 means present and possibly empty; absent is a different thing."""
    if viewport == "axial":
        dataset = write_slices(tmp_path, frames=1)[0]
    else:
        dataset = _rendered_instance(tmp_path / "rendered")

    assert [name for name in REQUIRED if name not in dataset] == []


def test_a_reformat_keeps_the_frame_of_reference_it_was_cut_from(tmp_path):
    """A new one would say the cut and its source cannot be compared."""
    dataset = write_slices(tmp_path, frames=1)[0]

    assert dataset.FrameOfReferenceUID == REFERENCE_FRAME_UID
    assert "PositionReferenceIndicator" in dataset


def slice_plane():
    """The cut ``write_slices`` writes, for a test that needs its own writer."""
    return plane_from_reslice(posed_reslice(phantom()))


def test_a_reformat_cites_the_images_it_was_made_from(tmp_path):
    source = reference_dataset()
    writer = SliceWriter(
        context(
            tmp_path,
            identity=Identity(
                source_images=(source,),
                frame_of_reference=REFERENCE_FRAME_UID,
                modality="MR",
            ),
        )
    )
    writer.add(0, Frame(image=rgb_frame().image, plane=slice_plane()))
    writer.close()

    dataset = written(pl.Path(tmp_path) / "axial")[0]

    assert [item.ReferencedSOPInstanceUID for item in dataset.SourceImageSequence] == [
        source.SOPInstanceUID
    ]
    assert dataset.DerivationCodeSequence[0].CodeValue == "113072"
    assert dataset.ImageType[0] == "DERIVED"


def test_a_cut_says_which_way_its_rows_and_columns_run(tmp_path):
    dataset = write_slices(tmp_path, frames=1)[0]

    assert len(dataset.PatientOrientation) == 2
    assert all(letter in "LRAPHF" for letter in dataset.PatientOrientation)


def test_a_render_says_nothing_about_which_way_its_rows_run(tmp_path):
    """A volume render has a camera rather than a row and a column."""
    dataset = _rendered_instance(tmp_path / "rendered")

    assert "PatientOrientation" in dataset
    assert dataset.PatientOrientation in ([], "")


def test_a_capture_without_a_dicom_source_stands_alone_whole(tmp_path):
    """Either the identity comes from a source or none of it does."""
    writer = SliceWriter(
        context(tmp_path, identity=Identity(study_instance_uid="1.2.3"))
    )
    writer.add(0, Frame(image=rgb_frame().image, plane=slice_plane()))
    writer.close()

    dataset = written(pl.Path(tmp_path) / "axial")[0]

    assert dataset.StudyInstanceUID == "1.2.3"
    assert str(dataset.PatientName) == "Anonymous^"
    assert dataset.PatientID == "CARDIO"
    assert "SourceImageSequence" not in dataset


# --- the study a capture off a real series lands in ----------------------------


def dicom_cine_app(tmp_path, **overrides):
    """An app whose volume was read from DICOM, so a capture has a study to join."""
    source = tmp_path / "source"
    write_cine_series(source, slices=3, phases=2)

    scene = Scene(
        volumes=[{"label": "vol", "directory": source}],
        serialization_directory=tmp_path / "out",
        active_volume_label="vol",
        capture_format="dicom-data",
        **overrides,
    )
    return running(scene, "axial")


def captured(logic) -> list[pd.dataset.Dataset]:
    """Every instance one capture left on disk, whichever viewport wrote it."""
    root = logic.scene.screenshot_directory
    return [pd.dcmread(path) for path in sorted(root.glob("*/*/*.dcm"))]


def test_a_capture_joins_the_study_its_volume_was_read_from(tmp_path):
    _server, _scene, logic = dicom_cine_app(tmp_path)
    source = pd.dcmread(next((tmp_path / "source").glob("*.dcm")))

    capture(logic)

    datasets = captured(logic)
    assert datasets
    for dataset in datasets:
        assert dataset.StudyInstanceUID == source.StudyInstanceUID
        assert str(dataset.PatientName) == "Phantom^Cine"
        assert dataset.PatientID == "PHANTOM-1"
        assert dataset.FrameOfReferenceUID == source.FrameOfReferenceUID
        # Its own series, not the one it was cut from.
        assert dataset.SeriesInstanceUID != source.SeriesInstanceUID


def test_a_capture_off_a_real_series_cites_every_instance_of_it(tmp_path):
    _server, _scene, logic = dicom_cine_app(tmp_path)
    written_uids = {
        str(pd.dcmread(path).SOPInstanceUID)
        for path in (tmp_path / "source").glob("*.dcm")
    }

    capture(logic)

    dataset = captured(logic)[0]

    assert {
        str(item.ReferencedSOPInstanceUID) for item in dataset.SourceImageSequence
    } == written_uids


def test_a_capture_off_a_real_series_reports_its_modality(tmp_path):
    """A reformat of an MR study arriving as OT is a reformat a PACS cannot route."""
    _server, _scene, logic = dicom_cine_app(tmp_path)

    capture(logic)

    assert captured(logic)[0].Modality == "MR"


def test_an_unset_equipment_name_is_left_out_rather_than_written_empty(tmp_path):
    """Type 3: absent says "not recorded", empty claims the value is blank."""
    dataset = write_slices(tmp_path, frames=1)[0]

    assert "StationName" not in dataset
    assert "InstitutionName" not in dataset


def test_the_configured_equipment_is_what_the_instance_names(tmp_path):
    dataset = write_slices(
        tmp_path,
        frames=1,
        equipment=Equipment(
            manufacturer="Acme",
            model_name="Cardio Station",
            institution_name="St Elsewhere",
            station_name="READING-3",
            device_serial_number="SN-7",
        ),
    )[0]

    assert dataset.Manufacturer == "Acme"
    assert dataset.ManufacturerModelName == "Cardio Station"
    assert dataset.InstitutionName == "St Elsewhere"
    assert dataset.StationName == "READING-3"
    assert dataset.DeviceSerialNumber == "SN-7"
    # Read off the package rather than configured, so it cannot disagree with
    # the code that wrote the file.
    assert dataset.SoftwareVersions == version("cardio")


def test_a_long_equipment_name_is_refused_rather_than_truncated(tmp_path):
    with pytest.raises(pc.ValidationError):
        Equipment(station_name="X" * 17)


# --- whose UIDs a capture writes under -----------------------------------------

REGISTERED_ROOT = "1.2.840.99999.1"


def test_every_uid_is_minted_under_the_configured_root(tmp_path):
    """Instances under a borrowed root claim a creator that did not create them."""
    datasets = write_slices(tmp_path, frames=2, uid_root=REGISTERED_ROOT)

    minted = {
        str(dataset[tag].value)
        for dataset in datasets
        for tag in ("SOPInstanceUID", "SeriesInstanceUID")
    }

    assert len(minted) == 3
    assert all(value.startswith(f"{REGISTERED_ROOT}.") for value in minted)


def test_the_file_says_which_implementation_wrote_it(tmp_path):
    """highdicom names itself here, which is true of how and not of what."""
    dataset = write_slices(tmp_path, frames=1, uid_root=REGISTERED_ROOT)[0]

    assert dataset.file_meta.ImplementationClassUID == f"{REGISTERED_ROOT}.1"
    assert dataset.file_meta.ImplementationVersionName.startswith("CARDIO ")
    assert len(dataset.file_meta.ImplementationVersionName) <= 16


def test_the_study_a_standalone_capture_opens_is_under_the_root_too(tmp_path):
    _server, _scene, logic = built(
        tmp_path, "axial", capture_format="dicom-data", uid_root=REGISTERED_ROOT
    )
    capture(logic)

    dataset = captured_series(tmp_path)

    assert dataset.StudyInstanceUID.startswith(f"{REGISTERED_ROOT}.")


def test_the_default_root_says_it_is_not_the_deployment_s(caplog):
    assert not uid.is_registered(uid.DEFAULT_ROOT)

    with caplog.at_level(logging.WARNING):
        assert uid.warn_if_unregistered(uid.DEFAULT_ROOT)

    assert "must not be sent to a production archive" in caplog.text


def test_a_registered_root_is_not_warned_about(caplog):
    with caplog.at_level(logging.WARNING):
        assert not uid.warn_if_unregistered(REGISTERED_ROOT)

    assert caplog.text == ""


# --- what a capture is going out without ---------------------------------------


def sparse_source(**fields) -> pd.dataset.Dataset:
    """A source instance carrying only the fields named."""
    dataset = pd.dataset.Dataset()
    for name, value in fields.items():
        setattr(dataset, name, value)
    return dataset


def test_a_complete_source_is_not_warned_about():
    complete = sparse_source(**{field.name: "1" for field in preflight.SOURCE_FIELDS})
    equipment = Equipment(institution_name="St Elsewhere")

    assert preflight.missing(complete, equipment) == []


def test_every_field_a_sparse_source_lacks_is_named():
    source = sparse_source(PatientID="X", PatientName="Y")

    absent = {field.name for field in preflight.missing(source, Equipment())}

    assert "PatientID" not in absent
    assert "PatientName" not in absent
    assert {"StudyInstanceUID", "FrameOfReferenceUID", "AccessionNumber"} <= absent
    assert "InstitutionName" in absent


def test_an_empty_field_counts_as_missing():
    """An empty Type 2 element looks exactly like one that was meant to be blank."""
    source = sparse_source(PatientID="", AccessionNumber="")

    absent = {field.name for field in preflight.missing(source, Equipment())}

    assert {"PatientID", "AccessionNumber"} <= absent


def test_a_volume_read_from_a_file_is_missing_all_of_it():
    absent = preflight.missing(None, Equipment())

    assert len(absent) == len(preflight.SOURCE_FIELDS) + 1


def test_the_warning_names_the_fields_and_says_why(caplog):
    with caplog.at_level(logging.INFO):
        preflight.report(sparse_source(PatientID="X"), Equipment())

    assert "a receiving archive may want" in caplog.text
    assert "StudyInstanceUID" in caplog.text
    assert "routing rules key off it" in caplog.text


def test_a_research_capture_is_written_anyway_and_says_what_is_missing(tmp_path):
    """Whether a field is needed depends on where it is going, which is not ours."""
    _server, _scene, logic = built(tmp_path, "axial", capture_format="dicom-data")

    capture(logic)

    assert logic.capture.server.state.capture_ok
    assert "field(s) a receiver may want are missing" in (
        logic.capture.server.state.capture_summary
    )
    assert captured_series(tmp_path)


def test_a_production_capture_under_a_borrowed_root_is_refused(tmp_path):
    _server, _scene, logic = built(
        tmp_path, "axial", capture_format="dicom-data", production=True
    )

    capture(logic)

    state = logic.capture.server.state
    assert not state.capture_ok
    assert "not this deployment's root" in state.capture_summary
    assert not (tmp_path / "out" / "screenshots").exists()


def test_a_production_capture_under_a_registered_root_is_written(tmp_path):
    _server, _scene, logic = built(
        tmp_path,
        "axial",
        capture_format="dicom-data",
        production=True,
        uid_root=REGISTERED_ROOT,
    )

    capture(logic)

    assert logic.capture.server.state.capture_ok
    assert captured_series(tmp_path)


def test_a_picture_capture_is_not_pre_flighted(tmp_path):
    """The checks are about what a receiver wants; a PNG has no receiver."""
    _server, _scene, logic = built(
        tmp_path, "axial", capture_format="png", production=True
    )

    capture(logic)

    assert logic.capture.server.state.capture_ok
