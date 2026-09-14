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
from cardio.camera import visible_rectangle
from cardio.capture import (
    CaptureFormat,
    Context,
    Equipment,
    Frame,
    Identity,
    Plane,
    TransferSyntax,
    encoding,
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
from cardio.capture.dicom import (
    MultiFrameRenderedWriter,
    SecondaryCaptureWriter,
    SliceWriter,
    encode,
)
from cardio.capture.formats import WRITERS, writer_for, writes_series
from cardio.capture.geometry import (
    plane_from_reslice,
    reslice_axes,
    scalars_2d,
    square_pixels,
)
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
    """One upper-left cut of the phantom, posed the way the MPR views pose theirs."""
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(image_data)
    reslice.SetOutputDimensionality(2)
    reslice.SetInterpolationModeToLinear()
    reslice.SetBackgroundLevel(-1000.0)
    reslice.AutoCropOutputOn()
    reslice.SetOutputDirection(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    reslice.SetResliceAxes(
        create_vtk_reslice_matrix(
            (turned() if rotation is None else rotation) @ VIEW_TRANSFORMS["ul"],
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


def context(tmp_path, viewport="ul", **kwargs) -> Context:
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
    """The column direction runs against the cut's y, the rows being turned
    over so that the top of the view is written first."""
    reslice = posed_reslice(phantom())
    axes = reslice_axes(reslice)

    plane = plane_from_reslice(reslice)
    orientation = np.array(plane.location.orientation)

    assert np.allclose(orientation[:3], axes[:3, 0])
    assert np.allclose(orientation[3:], -axes[:3, 1])
    assert np.isclose(np.linalg.norm(orientation[:3]), 1.0)
    assert np.isclose(np.linalg.norm(orientation[3:]), 1.0)


def test_the_cut_is_written_from_the_top_of_the_view_down():
    """Which is the row a viewer draws first, and the row the geometry names.

    A reslice hands its output back from the low end of its y, which is the
    bottom of the view: written in that order, the capture arrives upside down
    beside the rendered capture of the same view.
    """
    reslice = posed_reslice(phantom())

    plane = plane_from_reslice(reslice)

    assert np.array_equal(
        plane.scalars, np.flipud(scalars_2d(square_pixels(reslice).GetOutput()))
    )


def interpolated(image_data, point) -> float:
    """What the volume holds at ``point``, read the way a reslice reads it."""
    interpolator = vtk.vtkImageInterpolator()
    interpolator.SetInterpolationModeToLinear()
    interpolator.Initialize(image_data)
    return interpolator.Interpolate(point[0], point[1], point[2], 0)


def test_the_geometry_finds_the_pixels_the_capture_holds():
    """Walk from the declared corner along the declared cosines, and the pixel
    landed on holds what the volume holds there.

    The whole of what the geometry claims, checked against the volume rather
    than recomputed from the pose the way the writer computes it: a raster
    turned over without its position and cosines being turned with it lands on
    some other voxel and fails here.
    """
    image_data = phantom()
    plane = plane_from_reslice(posed_reslice(image_data))

    rows, columns = plane.scalars.shape
    row_spacing, column_spacing = plane.pixel_spacing
    orientation = np.array(plane.location.orientation)
    corner = np.array(plane.location.position)

    # Off both centres, so a flip about either cannot pass, and well inside the
    # cut, where an autocropped oblique corner would be outside the volume.
    for row, column in ((rows // 4, columns // 3), (rows // 3, columns // 2)):
        point = (
            corner
            + column * column_spacing * orientation[:3]
            + row * row_spacing * orientation[3:]
        )
        assert np.isclose(
            interpolated(image_data, point), plane.scalars[row, column], atol=1.0
        )


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


def test_a_cut_is_written_on_square_pixels():
    """However the cut fell, and whether or not it was turned.

    An autocropped cut comes out with a spacing of its own on each axis, which
    ``PixelSpacing`` describes and a viewer reading a Secondary Capture may
    never look at.  Square pixels are what such a viewer draws correctly.
    """
    finest = min(VOLUME_SPACING)

    for rotation in (None, np.eye(3)):
        plane = plane_from_reslice(posed_reslice(phantom(), rotation=rotation))

        assert plane.pixel_spacing == (finest, finest)
        assert plane.thickness == finest


def test_resampling_a_cut_changes_its_sampling_and_not_its_reach():
    """The capture is the view's own cut more finely sampled, not a wider one."""
    reslice = posed_reslice(phantom())
    view = reslice.GetOutput()
    columns, rows, _ = view.GetDimensions()
    spacing = view.GetSpacing()

    plane = plane_from_reslice(reslice)
    written_rows, written_columns = plane.scalars.shape
    row_spacing, column_spacing = plane.pixel_spacing

    assert np.isclose(
        (written_columns - 1) * column_spacing,
        (columns - 1) * spacing[0],
        atol=spacing[0],
    )
    assert np.isclose(
        (written_rows - 1) * row_spacing, (rows - 1) * spacing[1], atol=spacing[1]
    )


def test_a_cut_is_cropped_to_the_rectangle_it_is_given():
    """Which is what the view is showing, so a view zoomed onto the chambers
    does not export the whole reformat."""
    rectangle = ((-4.0, 6.0), (-3.0, 9.0))

    plane = plane_from_reslice(posed_reslice(phantom()), rectangle)

    rows, columns = plane.scalars.shape
    row_spacing, column_spacing = plane.pixel_spacing
    # Covering the rectangle, and not by more than the pixel it is covered in.
    assert 10.0 <= (columns - 1) * column_spacing < 10.0 + column_spacing
    assert 12.0 <= (rows - 1) * row_spacing < 12.0 + row_spacing


def test_a_cropped_cut_says_where_its_own_first_pixel_is():
    """The corner the geometry names moves with the crop, the first row written
    being the top of the rectangle rather than the top of the whole cut."""
    reslice = posed_reslice(phantom())
    rectangle = ((-4.0, 6.0), (-3.0, 9.0))
    axes = reslice_axes(reslice)

    plane = plane_from_reslice(reslice, rectangle)

    row_spacing, _ = plane.pixel_spacing
    corner = axes[:3, :3] @ np.array([-4.0, 9.0, 0.0]) + axes[:3, 3]
    assert np.allclose(plane.location.position, corner, atol=row_spacing)


def test_a_cut_asked_for_on_a_grid_lands_on_that_grid():
    """The rectangle cut into exactly the pixels it is drawn on, sampled at
    their centres, which is where the screen's pixels are and where DICOM says
    a pixel is."""
    rectangle = ((-4.0, 6.0), (-3.0, 9.0))

    square = square_pixels(posed_reslice(phantom()), rectangle, (50, 24))
    image = square.GetOutput()

    assert image.GetDimensions()[:2] == (50, 24)
    assert image.GetSpacing()[0] == pytest.approx(10.0 / 50)
    assert image.GetOrigin()[0] == pytest.approx(-4.0 + (10.0 / 50) / 2.0)
    assert image.GetOrigin()[1] == pytest.approx(-3.0 + (10.0 / 50) / 2.0)


def test_a_cut_on_a_grid_is_no_thinner_than_the_volume_it_is_cut_from():
    """A finer grid asks for the cut at more places, not from a thinner study."""
    rectangle = ((-4.0, 6.0), (-3.0, 9.0))

    plane = plane_from_reslice(posed_reslice(phantom()), rectangle, (50, 24))

    assert plane.pixel_spacing[0] < min(VOLUME_SPACING)
    assert plane.thickness == min(VOLUME_SPACING)


def test_a_capture_does_not_resample_the_view_it_was_taken_of():
    """The person posed the views; a capture is not a reason to redraw them."""
    reslice = posed_reslice(phantom())
    before = reslice.GetOutput().GetSpacing(), reslice.GetOutput().GetDimensions()

    plane_from_reslice(reslice)

    assert (
        reslice.GetOutput().GetSpacing(),
        reslice.GetOutput().GetDimensions(),
    ) == before


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


def write_slices(tmp_path, frames: int = 3, viewport="ul", **naming) -> list:
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

    assert written(pl.Path(tmp_path) / "ul") == []


def test_a_slice_series_reads_back_as_the_frames_it_was_written_from(tmp_path):
    write_slices(tmp_path, frames=3)

    frames = dicom.read_series(pl.Path(tmp_path) / "ul")

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
        VIEW_TRANSFORMS["ul"],
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
    wide = compose(phantom(), poses(6), VIEW_TRANSFORMS["ul"], 1, 6)
    grid = compose(phantom(), poses(6), VIEW_TRANSFORMS["ul"], 2, 3)

    tile_rows, tile_columns = wide.scalars.shape[0], wide.scalars.shape[1] // 6
    assert grid.scalars.shape == (2 * tile_rows, 3 * tile_columns)


def test_a_mosaic_tile_holds_what_a_tile_of_the_grid_was_showing():
    """Cut to the rectangle the grid shows of each cut, so the mosaic and the
    picture of the grid are framed alike."""
    rectangle = ((-8.0, 8.0), (-5.0, 5.0))

    plane = compose(phantom(), poses(2), VIEW_TRANSFORMS["ul"], 1, 2, rectangle)

    rows, columns = plane.scalars.shape
    spacing = plane.pixel_spacing[0]
    tile_columns = columns // 2
    assert 16.0 <= (tile_columns - 1) * spacing < 16.0 + spacing
    assert 10.0 <= (rows - 1) * spacing < 10.0 + spacing


def test_a_mosaic_tile_is_sampled_on_the_pixels_the_grid_gives_it():
    """Each tile filling the pixels the screen draws it on, so the mosaic is
    the grid as shown and the band stamped under it reads the same way."""
    rectangle = ((-8.0, 8.0), (-5.0, 5.0))

    plane = compose(
        phantom(), poses(2), VIEW_TRANSFORMS["ul"], 1, 2, rectangle, (64, 40)
    )

    assert plane.scalars.shape == (40, 2 * 64)
    assert plane.pixel_spacing == (16.0 / 64, 16.0 / 64)


def test_a_mosaic_says_how_thick_its_tiles_are_however_finely_it_samples_them():
    """The tiles are asked for at more places, not cut from a thinner study."""
    rectangle = ((-8.0, 8.0), (-5.0, 5.0))

    plane = compose(
        phantom(), poses(2), VIEW_TRANSFORMS["ul"], 1, 2, rectangle, (64, 40)
    )

    assert plane.pixel_spacing[0] < min(VOLUME_SPACING)
    assert plane.thickness == min(VOLUME_SPACING)


def test_a_grid_that_is_showing_nothing_yet_holds_its_cuts_whole():
    """A window that has never been sized frames nothing, and a mosaic of the
    cuts themselves is better than one cut to a rectangle nobody chose."""
    rectangle = ((-8.0, 8.0), (-5.0, 5.0))

    framed = compose(phantom(), poses(2), VIEW_TRANSFORMS["ul"], 1, 2, rectangle)
    whole = mosaic_of(1, 2)

    # The phantom's cuts run well past the rectangle above on both axes.
    assert whole.scalars.shape[0] > framed.scalars.shape[0]
    assert whole.scalars.shape[1] > framed.scalars.shape[1]


def test_a_mosaic_keeps_the_volume_values():
    plane = mosaic_of(1, 1)

    assert plane.scalars.dtype == np.int16
    assert plane.scalars.max() > 0


def test_the_tiles_are_laid_out_row_major_from_the_top_left():
    """Four cuts down a column are the same four cuts along a row, in order."""
    tall = compose(phantom(), poses(4), VIEW_TRANSFORMS["ul"], 4, 1)
    wide = compose(phantom(), poses(4), VIEW_TRANSFORMS["ul"], 1, 4)

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


def test_a_mosaic_says_nothing_about_a_plane_it_has_no_place_on(tmp_path):
    """The thickness belongs to the module the position does, so it goes with
    it: a slab thickness of a cut the instance never locates says nothing a
    receiver can use, and leaves the module half stated."""
    writer = SliceWriter(context(tmp_path, "tile"))
    writer.add(0, Frame(image=rgb_frame().image, plane=mosaic_of(1, 2)))
    writer.close()

    dataset = written(pl.Path(tmp_path) / "tile")[0]

    assert "SliceThickness" not in dataset


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
    writer = writer_for(CaptureFormat.DICOM_DATA, context(tmp_path, "ul"))

    assert isinstance(writer, SliceWriter)


def sized_views(scene, size: int = 256):
    """The MPR windows given a size, as a headless session gives them one.

    A renderer learns its viewport from the window, and until it has one it is
    showing nothing to crop a capture to.
    """
    for view in scene.mpr_views:
        scene.mpr_views[view].SetSize(size, size)
    return scene.mpr_views


def test_a_data_capture_is_framed_like_the_picture_beside_it(tmp_path):
    """Both captures of one view show the same part of the cut."""
    _server, scene, logic = built(tmp_path, "ul")
    views = sized_views(scene)

    plane = logic.capture.plane_for("ul", 0)

    (low_x, high_x), (low_y, high_y) = visible_rectangle(views.renderer("ul"))
    rows, columns = plane.scalars.shape
    row_spacing, column_spacing = plane.pixel_spacing
    assert columns * column_spacing == pytest.approx(high_x - low_x)
    assert rows * row_spacing == pytest.approx(high_y - low_y)


def test_a_data_capture_is_sampled_like_the_picture_beside_it(tmp_path):
    """Pixel for pixel the same frame, which is what makes the banner readable.

    A cut written at the volume's own sampling is a few hundred pixels across,
    and a band stamped in proportion to it is a few pixels tall -- crisp in the
    file, and unreadable once a viewer has magnified the image to fill a
    screen.  On the view's own grid the band is the size it is in the picture.
    """
    _server, scene, logic = built(tmp_path, "ul")
    views = sized_views(scene, size=192)

    plane = logic.capture.plane_for("ul", 0)

    assert plane.scalars.shape == (192, 192)


def test_zooming_a_view_samples_the_capture_more_finely(tmp_path):
    """The zoom is how a person says what they want to see, and a capture of
    the whole reformat is not what they asked for.  The view keeps its pixels
    and spends them on less of the cut, which is what zooming in is."""
    _server, scene, logic = built(tmp_path, "ul")
    views = sized_views(scene)
    before = logic.capture.plane_for("ul", 0)

    views.zoom(2.0)
    after = logic.capture.plane_for("ul", 0)

    assert after.scalars.shape == before.scalars.shape
    assert after.pixel_spacing[0] == pytest.approx(before.pixel_spacing[0] / 2.0)
    assert after.pixel_spacing[1] == pytest.approx(before.pixel_spacing[1] / 2.0)


def test_a_view_that_has_never_been_sized_is_captured_whole(tmp_path):
    """It is showing nothing to be framed like, and a capture of nothing at all
    would be worse than one framed differently."""
    _server, scene, logic = built(tmp_path, "ul")

    assert visible_rectangle(scene.mpr_views.renderer("ul")) is None
    assert logic.capture.plane_for("ul", 0).scalars.size > 0


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
    """A whole app, the way the smoke tests build one, in a chosen layout.

    A research session unless a test says otherwise: the phantom is written
    under pydicom's root, which is exactly what a deployment that has not said
    it is research refuses to write DICOM under.

    JPEG-LS unless a test says otherwise, too: the phantom's cuts are a few
    pixels across, which is under the 32 JPEG 2000 needs, and an app test
    should exercise the encoding rather than the fallback from it.
    """
    overrides.setdefault("research", True)
    overrides.setdefault("capture_transfer_syntax", TransferSyntax.JPEG_LS)
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
        ("", {"ul", "ll", "lr", "vr"}),
        ("volume", {"vr"}),
        ("ul", {"ul"}),
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
        ("tile", ["ul"], set()),
        ("tile", ["tile", "ul"], {"tile"}),
        # The two that used to be written from a window nobody was looking at.
        ("tile", ["vr"], set()),
        ("ul", ["vr", "ll"], set()),
        ("", ["tile"], set()),
        ("", ["ul", "vr"], {"ul", "vr"}),
        ("volumetry", ["volumetry"], {"volumetry"}),
        ("volumetry", ["ul", "vr"], set()),
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
    tick(server, "ul")
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
    tick(server, "ul")

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

    assert dataset.SeriesDescription == "cardio ul (reformat)"


def rendered(tmp_path, rows=6, columns=8, **naming) -> pd.dataset.Dataset:
    writer = SecondaryCaptureWriter(context(tmp_path, "vr", **naming))
    writer.add(0, rgb_frame(rows=rows, columns=columns))
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
    assert repeated_numbers({"ul": 5, "ll": 5, "vr": 6}) == {5: ["ll", "ul"]}


def test_numbers_that_differ_are_not_reported():
    assert repeated_numbers({"ul": 5, "ll": 6}) == {}


def test_a_description_longer_than_dicom_carries_is_refused():
    """Rejected by the receiver rather than shortened, so refuse it here."""
    Series(description="x" * DESCRIPTION_LIMIT)

    with pytest.raises(pc.ValidationError):
        Series(description="x" * (DESCRIPTION_LIMIT + 1))


def test_a_series_number_no_instance_can_carry_is_refused():
    """DICOM numbers a series from one, and the instance constructor refuses a
    zero -- which would otherwise stop a capture partway through it."""
    Series(number=1)

    with pytest.raises(pc.ValidationError):
        Series(number=0)


def test_a_viewport_that_does_not_exist_is_refused():
    """A misspelled one would otherwise quietly name nothing."""
    with pytest.raises(pc.ValidationError):
        SeriesTags(oblique={"number": 3})


def test_only_the_dicom_formats_have_a_series_to_name():
    naming = {fmt for fmt in CaptureFormat if writes_series(fmt)}

    assert naming == {
        CaptureFormat.DICOM_RENDERED,
        CaptureFormat.DICOM_DATA,
        CaptureFormat.DICOM_CINE_RENDERED,
        CaptureFormat.DICOM_CINE_DATA,
    }


def captured_series(tmp_path, viewport="ul") -> pd.dataset.Dataset:
    """The first instance of the one series a capture just wrote."""
    folder = max((tmp_path / "out" / "screenshots").iterdir())
    return written(folder / viewport)[0]


def exporting(tmp_path, **overrides):
    """An app set up to write one DICOM series, off the upper-left view."""
    server, _, logic = built(tmp_path, "ul", capture_format="dicom-data", **overrides)
    tick(server, "ul")
    return server, logic


def test_a_capture_is_written_as_the_series_the_scene_named(tmp_path):
    _, logic = exporting(
        tmp_path,
        capture_series={"ul": {"number": 407, "description": "Cine SAX"}},
    )

    capture(logic)

    dataset = captured_series(tmp_path)
    assert dataset.SeriesNumber == 407
    assert dataset.SeriesDescription == "Cine SAX"


def test_the_configured_naming_reaches_the_drawer(tmp_path):
    server, _, _ = built(
        tmp_path,
        capture_series={"ul": {"number": 400, "description": "Cine SAX"}},
    )

    assert server.state.capture_series_number_ul == 400
    assert server.state.capture_series_description_ul == "Cine SAX"


def test_what_the_drawer_holds_is_what_the_capture_is_written_as(tmp_path):
    """Retyped rather than reconfigured, and the number field hands back text.

    A ``type="number"`` field's v-model is a string, so a number typed into
    one has to reach ``SeriesNumber`` as the number it reads as.
    """
    server, logic = exporting(tmp_path)
    with server.state:
        server.state.capture_series_number_ul = "512"
        server.state.capture_series_description_ul = "Retyped"

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
    overrides.setdefault("research", True)
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
    return running(scene, "ul")


def test_a_cine_reads_only_cuts_that_have_been_posed(tmp_path, monkeypatch):
    """The data has to come from the frame on screen, not the one next up.

    Writing ``frame`` reposes the views when the state block flushes, which is
    after the pass that captures them.  A plane taken after the increment
    therefore belongs to a frame whose reslices may not exist yet, and asking
    for them builds a set centred on the image: the capture would record an
    axis-aligned cut through the middle instead of the one the user posed.
    """
    server, scene, logic = cine_app(tmp_path, capture_format="dicom-data")
    tick(server, "ul")
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


@pytest.mark.parametrize("viewport", ["ul", "vr"])
def test_every_required_element_is_present_even_where_it_is_empty(tmp_path, viewport):
    """Type 2 means present and possibly empty; absent is a different thing."""
    if viewport == "ul":
        dataset = write_slices(tmp_path, frames=1)[0]
    else:
        dataset = _rendered_instance(tmp_path / "rendered")

    assert [name for name in REQUIRED if name not in dataset] == []


def test_the_formats_a_capture_is_sent_in_are_plain_secondary_capture(tmp_path):
    """The one class both receivers this was checked against read.  Sectra
    stores the multi-frame classes too; TeraRecon iNtuition lists only this one
    among the classes it stores without being configured to, so an export that
    has to arrive at both is written single-frame."""
    cut = write_slices(tmp_path / "cut", frames=1)[0]
    picture = _rendered_instance(tmp_path / "picture")

    assert cut.SOPClassUID == pd.uid.SecondaryCaptureImageStorage
    assert picture.SOPClassUID == pd.uid.SecondaryCaptureImageStorage


def test_a_rendered_cine_says_how_its_values_are_to_be_presented(tmp_path):
    """Required of the multi-frame classes where it is merely allowed of the
    single-frame one, so the cine the one receiver that reads it gets is whole.
    """
    writer = writer_for("dicom-cine-rendered", context(tmp_path, "vr"))
    writer.add(0, rgb_frame(rows=7, columns=9))
    writer.close()

    dataset = written(pl.Path(tmp_path) / "vr")[0]

    assert dataset.PresentationLUTShape == "IDENTITY"


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

    dataset = written(pl.Path(tmp_path) / "ul")[0]

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

    dataset = written(pl.Path(tmp_path) / "ul")[0]

    assert dataset.StudyInstanceUID == "1.2.3"
    assert str(dataset.PatientName) == "Anonymous^"
    assert dataset.PatientID == "CARDIO"
    assert "SourceImageSequence" not in dataset


# --- the study a capture off a real series lands in ----------------------------


def dicom_cine_app(tmp_path, **overrides):
    """An app whose volume was read from DICOM, so a capture has a study to join."""
    overrides.setdefault("research", True)
    source = tmp_path / "source"
    write_cine_series(source, slices=3, phases=2)

    scene = Scene(
        volumes=[{"label": "vol", "directory": source}],
        serialization_directory=tmp_path / "out",
        active_volume_label="vol",
        capture_format="dicom-data",
        **overrides,
    )
    return running(scene, "ul")


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


# --- how the pixels are encoded ------------------------------------------------


# The two lossless encodings, which differ in what a receiver has heard of
# rather than in what comes back out of them.
COMPRESSED = (TransferSyntax.JPEG_LS, TransferSyntax.JPEG_2000)


def test_a_capture_is_compressed_unless_it_is_asked_not_to_be(tmp_path):
    """A cine of a rendered view is tens of megabytes of pixels nobody has to
    send, and the compression is lossless, so it is what a capture opens on.

    JPEG 2000 rather than the smaller JPEG-LS: the workstations these captures
    are read on are likelier to have heard of it.
    """
    compressed = write_slices(tmp_path / "default", frames=1)[0]
    plain = write_slices(
        tmp_path / "plain", frames=1, transfer_syntax=TransferSyntax.UNCOMPRESSED
    )[0]

    assert compressed.file_meta.TransferSyntaxUID == pd.uid.JPEG2000Lossless
    assert plain.file_meta.TransferSyntaxUID == pd.uid.ExplicitVRLittleEndian


@pytest.mark.parametrize("syntax", COMPRESSED)
def test_every_syntax_is_written_as_the_one_it_names(tmp_path, syntax):
    dataset = write_slices(tmp_path, frames=1, transfer_syntax=syntax)[0]

    assert dataset.file_meta.TransferSyntaxUID == encoding.uid_for(syntax)


@pytest.mark.parametrize("syntax", COMPRESSED)
def test_compression_keeps_every_value(tmp_path, syntax):
    """Lossless, and checked as such: a capture is what the measurements were
    taken off, and one that had quietly rounded them would read the same."""
    compressed = write_slices(tmp_path / str(syntax), frames=1, transfer_syntax=syntax)[
        0
    ]
    plain = write_slices(
        tmp_path / "plain", frames=1, transfer_syntax=TransferSyntax.UNCOMPRESSED
    )[0]

    assert np.array_equal(compressed.pixel_array, plain.pixel_array)
    assert compressed.RescaleSlope == plain.RescaleSlope
    assert compressed.RescaleIntercept == plain.RescaleIntercept


@pytest.mark.parametrize("syntax", COMPRESSED)
def test_a_colour_capture_says_what_its_components_really_are(tmp_path, syntax):
    """JPEG 2000 may transform the components on the way in, which DICOM has
    its own photometric interpretation for.  What the instance says has to be
    what the encoder did, or a viewer draws the colours it did not record.
    """
    # Above JPEG 2000's floor, so what is checked is the encoding and not the
    # fallback from it.
    frame = {"rows": 32, "columns": 40}
    dataset = rendered(tmp_path / str(syntax), transfer_syntax=syntax, **frame)

    assert dataset.file_meta.TransferSyntaxUID == encoding.uid_for(syntax)
    assert dataset.PhotometricInterpretation == "RGB"
    assert np.array_equal(dataset.pixel_array, image_to_array(rgb_frame(**frame).image))


@pytest.mark.parametrize("syntax", COMPRESSED)
def test_a_cine_is_compressed_as_one_instance(tmp_path, syntax):
    """Every frame of it, and it reads back as the frames it was written from."""
    dataset = write_cine(tmp_path / str(syntax), frames=3, transfer_syntax=syntax)[0]
    plain = write_cine(
        tmp_path / "plain", frames=3, transfer_syntax=TransferSyntax.UNCOMPRESSED
    )[0]

    assert dataset.file_meta.TransferSyntaxUID == encoding.uid_for(syntax)
    assert dataset.pixel_array.shape == (3, *plain.pixel_array.shape[1:])
    assert np.array_equal(dataset.pixel_array, plain.pixel_array)


def test_a_compressed_capture_is_the_instance_that_was_built(tmp_path):
    """pydicom renumbers what it compresses unless told not to, and a capture
    minted under the configured root must not be handed one under another."""
    plain = write_slices(
        tmp_path / "plain",
        frames=1,
        uid_root=REGISTERED_ROOT,
        transfer_syntax=TransferSyntax.UNCOMPRESSED,
    )[0]
    compressed = write_slices(tmp_path / "default", frames=1, uid_root=REGISTERED_ROOT)[
        0
    ]

    for dataset in (plain, compressed):
        assert dataset.SOPInstanceUID.startswith(f"{REGISTERED_ROOT}.")
        assert dataset.file_meta.MediaStorageSOPInstanceUID == dataset.SOPInstanceUID


@pytest.mark.parametrize(
    "syntax", [TransferSyntax.JPEG_LS, TransferSyntax.UNCOMPRESSED]
)
def test_the_configured_syntax_is_what_a_capture_is_written_as(tmp_path, syntax):
    """The scene field reaching the writers, which is where it does its work.

    JPEG 2000 is left out here rather than tested through the fallback: the
    phantom's cuts are under its floor, which ``built`` says more about.
    """
    _server, logic = exporting(tmp_path, capture_transfer_syntax=syntax)
    capture(logic)

    assert captured_series(tmp_path).file_meta.TransferSyntaxUID == encoding.uid_for(
        syntax
    )


def test_a_frame_under_the_encoder_floor_is_written_uncompressed(tmp_path, caplog):
    """JPEG 2000 is encoded at six resolution levels, which a frame under 32
    pixels either way cannot be halved into, and openjpeg refuses it.

    The real refusal rather than a patched one: a capture that cannot be
    compressed is still a capture, and a view zoomed in that far is the way to
    ask for one.
    """
    with caplog.at_level(logging.ERROR):
        dataset = rendered(
            tmp_path, transfer_syntax=TransferSyntax.JPEG_2000, rows=8, columns=8
        )

    assert dataset.file_meta.TransferSyntaxUID == pd.uid.ExplicitVRLittleEndian
    assert np.array_equal(
        dataset.pixel_array, image_to_array(rgb_frame(rows=8, columns=8).image)
    )
    assert "uncompressed" in caplog.text


def test_an_encoder_that_cannot_write_leaves_the_capture_uncompressed(
    tmp_path, monkeypatch, caplog
):
    """A capture nobody can compress is still a capture: the file is larger
    than it was asked to be, and everything in it is still true."""

    def refuse(*args, **kwargs):
        raise RuntimeError("no encoding plugins are available")

    monkeypatch.setattr(pd.dataset.Dataset, "compress", refuse)

    with caplog.at_level(logging.ERROR):
        dataset = write_slices(tmp_path, frames=1)[0]

    assert dataset.file_meta.TransferSyntaxUID == pd.uid.ExplicitVRLittleEndian
    assert "uncompressed" in caplog.text


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
        tmp_path, "ul", capture_format="dicom-data", uid_root=REGISTERED_ROOT
    )
    capture(logic)

    dataset = captured_series(tmp_path)

    assert dataset.StudyInstanceUID.startswith(f"{REGISTERED_ROOT}.")


def test_the_default_root_says_it_is_not_the_deployment_s(caplog):
    assert not uid.is_registered(uid.DEFAULT_ROOT)

    with caplog.at_level(logging.WARNING):
        assert uid.warn_if_unregistered(uid.DEFAULT_ROOT)

    assert "must not be sent to an archive" in caplog.text


def test_the_startup_warning_says_what_it_costs_a_capture(caplog):
    """A deployment learns its captures are refused before it makes one."""
    with caplog.at_level(logging.WARNING):
        uid.warn_if_unregistered(uid.DEFAULT_ROOT)
    assert "DICOM captures are refused" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        uid.warn_if_unregistered(uid.DEFAULT_ROOT, research=True)
    assert "DICOM captures are refused" not in caplog.text


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


# A placeholder only has to be present, but it does have to be writable, and
# "1" is not a date or a time.
PLACEHOLDERS = {"DA": "20240101", "TM": "120000"}


def placeholder(name: str) -> str:
    """Something the VR of ``name`` accepts."""
    vr = pd.datadict.dictionary_VR(pd.datadict.tag_for_keyword(name))
    return PLACEHOLDERS.get(vr, "1")


def test_a_complete_source_is_not_warned_about():
    complete = sparse_source(
        **{field.name: placeholder(field.name) for field in preflight.SOURCE_FIELDS}
    )
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


def test_a_field_is_required_or_advisory_and_never_both():
    """The two are read separately, so an entry in both would refuse and warn."""
    required = {field.name for field in preflight.REQUIRED}

    assert required == {
        "PatientID",
        "PatientName",
        "StudyInstanceUID",
        "AccessionNumber",
        "StudyDate",
    }
    assert not required & {field.name for field in preflight.ADVISORY}


def test_an_instance_nobody_can_reconcile_with_an_order_is_refused():
    """Honestly empty and still a stray, which is the site's call to make."""
    source = identified(AccessionNumber="")

    assert [field.name for field in preflight.blocking(source, REGISTERED_ROOT)] == [
        "AccessionNumber"
    ]


def identified(**fields) -> pd.dataset.Dataset:
    """A source carrying the identity a capture may not be written without."""
    return sparse_source(
        **{
            "PatientID": "X",
            "PatientName": "Y^Z",
            "StudyInstanceUID": "1.2.3",
            "AccessionNumber": "ACC-1",
            "StudyDate": "20260101",
            **fields,
        }
    )


def test_an_identified_source_under_a_registered_root_stops_nothing():
    assert preflight.blocking(identified(), REGISTERED_ROOT) == []
    assert preflight.refused(identified(), REGISTERED_ROOT) == ""


def test_an_advisory_field_alone_stops_nothing():
    """It is missing, and missing is the caller's to accept."""
    source = identified()

    assert preflight.missing(source, Equipment())
    assert preflight.blocking(source, REGISTERED_ROOT) == []


def test_a_borrowed_root_stops_a_capture_on_its_own():
    stopping = preflight.blocking(identified(), uid.DEFAULT_ROOT)

    assert stopping == [preflight.UNREGISTERED_ROOT]
    assert "not this deployment's registered root" in preflight.refused(
        identified(), uid.DEFAULT_ROOT
    )


def test_a_volume_read_from_a_file_stops_a_capture():
    stopping = {field.name for field in preflight.blocking(None, REGISTERED_ROOT)}

    assert stopping == {field.name for field in preflight.REQUIRED}


def test_the_refusal_names_the_fields_and_the_way_out():
    message = preflight.refused(sparse_source(PatientID="X"), REGISTERED_ROOT)

    assert "PatientName" in message
    assert "StudyInstanceUID" in message
    assert "PatientID" not in message
    assert "set research to write it anyway" in message


def test_a_refusal_says_both_of_its_causes_at_once():
    """Fixing one and being refused again for the other is a bad afternoon."""
    message = preflight.refused(None, uid.DEFAULT_ROOT)

    assert "the source carries no" in message
    assert "not this deployment's registered root" in message


def test_a_research_capture_is_written_anyway_and_says_what_is_missing(tmp_path):
    """Whether a field is needed depends on where it is going, which is not ours."""
    _server, _scene, logic = built(tmp_path, "ul", capture_format="dicom-data")

    capture(logic)

    assert logic.capture.server.state.capture_ok
    assert "field(s) a receiver may want are missing" in (
        logic.capture.server.state.capture_summary
    )
    assert captured_series(tmp_path)


def test_a_capture_under_a_borrowed_root_is_refused_by_default(tmp_path):
    """The point of the default: nobody had to remember to ask for this."""
    _server, _scene, logic = built(
        tmp_path, "ul", capture_format="dicom-data", research=False
    )

    capture(logic)

    state = logic.capture.server.state
    assert not state.capture_ok
    assert "not this deployment's registered root" in state.capture_summary
    assert not (tmp_path / "out" / "screenshots").exists()


def test_an_identified_capture_under_a_registered_root_is_written(tmp_path):
    """Nothing is being taken on trust here, so nothing has to be waived."""
    _server, _scene, logic = dicom_cine_app(
        tmp_path, research=False, uid_root=REGISTERED_ROOT
    )
    tick(_server, "ul")

    capture(logic)

    assert logic.capture.server.state.capture_ok
    assert captured(logic)


def test_a_volume_read_from_a_file_is_refused_by_default(tmp_path):
    """Its patient and study would be ones this app made up."""
    _server, _scene, logic = built(
        tmp_path,
        "ul",
        capture_format="dicom-data",
        research=False,
        uid_root=REGISTERED_ROOT,
    )

    capture(logic)

    state = logic.capture.server.state
    assert not state.capture_ok
    assert "the source carries no" in state.capture_summary
    assert not (tmp_path / "out" / "screenshots").exists()


def test_research_writes_what_would_otherwise_be_refused(tmp_path):
    _server, _scene, logic = built(
        tmp_path, "ul", capture_format="dicom-data", research=True
    )

    capture(logic)

    assert logic.capture.server.state.capture_ok
    assert captured_series(tmp_path)


def test_a_refusal_says_what_is_missing_before_it_refuses(tmp_path, caplog):
    """The message names the way out; the log names why it was in the way."""
    _server, _scene, logic = built(
        tmp_path, "ul", capture_format="dicom-data", research=False
    )

    with caplog.at_level(logging.INFO):
        capture(logic)

    assert "a receiving archive may want" in caplog.text
    assert "Refused" in logic.capture.server.state.capture_summary


def test_a_picture_capture_is_not_pre_flighted(tmp_path):
    """The checks are about what a receiver wants; a PNG has no receiver."""
    _server, _scene, logic = built(tmp_path, "ul", capture_format="png", research=False)

    capture(logic)

    assert logic.capture.server.state.capture_ok


# --- the cine as one object ----------------------------------------------------


def write_cine(tmp_path, frames: int = 3, fmt="dicom-cine-data", **naming) -> list:
    reslice = posed_reslice(phantom())
    plane = plane_from_reslice(reslice)

    writer = writer_for(fmt, context(tmp_path, "ul", **naming))
    for index in range(frames):
        writer.add(index, Frame(image=rgb_frame().image, plane=plane))
    writer.close()

    return written(pl.Path(tmp_path) / "ul")


def test_a_cine_is_written_as_one_instance_of_many_frames(tmp_path):
    datasets = write_cine(tmp_path, frames=5)

    assert len(datasets) == 1
    assert datasets[0].NumberOfFrames == 5
    assert (
        datasets[0].SOPClassUID
        == pd.uid.MultiFrameGrayscaleWordSecondaryCaptureImageStorage
    )


def test_a_cine_says_how_fast_it_runs(tmp_path):
    """A viewer plays a multi-frame object; a stack of stills it merely sorts."""
    dataset = write_cine(tmp_path, frames=4)[0]

    assert float(dataset.FrameTime) == 50.0
    assert dataset.CineRate == 20
    assert dataset.RecommendedDisplayFrameRate == 20
    assert dataset.FrameIncrementPointer == pd.tag.Tag(0x0018, 0x1063)
    # Its timing is said once, in the cine module, not once per frame.
    assert "TriggerTime" not in dataset


def test_a_cine_of_a_fixed_plane_says_where_it_is(tmp_path):
    plane = plane_from_reslice(posed_reslice(phantom()))

    dataset = write_cine(tmp_path, frames=3)[0]

    assert np.allclose(dataset.ImageOrientationPatient, plane.location.orientation)
    assert np.allclose(dataset.ImagePositionPatient, plane.location.position)
    assert dataset.PresentationLUTShape == "IDENTITY"


def test_a_cine_whose_plane_moves_is_written_without_a_position(tmp_path, caplog):
    """One multi-frame instance carries one pose, which a moving cut has not got."""
    writer = writer_for("dicom-cine-data", context(tmp_path, "ul"))
    for index in range(3):
        reslice = posed_reslice(phantom(), origin=[5.0 + index, 27.0, 17.0])
        writer.add(
            index, Frame(image=rgb_frame().image, plane=plane_from_reslice(reslice))
        )
    with caplog.at_level(logging.WARNING):
        writer.close()

    dataset = written(pl.Path(tmp_path) / "ul")[0]

    assert "ImagePositionPatient" not in dataset
    assert "moves over the cycle" in caplog.text


def test_a_rendered_cine_is_one_true_colour_object(tmp_path):
    writer = writer_for("dicom-cine-rendered", context(tmp_path, "vr"))
    for index in range(3):
        writer.add(index, rgb_frame(rows=7, columns=9))
    writer.close()

    dataset = written(pl.Path(tmp_path) / "vr")[0]

    assert dataset.SOPClassUID == pd.uid.MultiFrameTrueColorSecondaryCaptureImageStorage
    assert dataset.NumberOfFrames == 3
    assert dataset.SamplesPerPixel == 3
    assert dataset.pixel_array.shape == (3, 7, 9, 3)


def test_a_viewport_with_no_cut_falls_back_to_a_rendered_cine(tmp_path):
    writer = writer_for(
        CaptureFormat.DICOM_CINE_DATA, context(tmp_path, "vr", has_plane=False)
    )

    assert isinstance(writer, MultiFrameRenderedWriter)


def test_a_cine_reads_back_as_the_frames_it_was_written_from(tmp_path):
    scalars = plane_from_reslice(posed_reslice(phantom())).scalars
    write_cine(tmp_path, frames=3)

    frames = dicom.read_series(pl.Path(tmp_path) / "ul")

    assert len(frames) == 3
    for frame in frames:
        array = itk.array_from_image(frame)
        assert array.shape == (1, *scalars.shape)
        assert np.allclose(array[0], scalars)


def test_a_cine_reads_back_with_the_geometry_it_was_written_with(tmp_path):
    plane = plane_from_reslice(posed_reslice(phantom()))
    write_cine(tmp_path, frames=2)

    instances = dicom.select_instances(pl.Path(tmp_path) / "ul")

    assert [instance.frame for instance in instances] == [0, 1]
    assert np.allclose(instances[0].position, plane.location.position)
    assert np.allclose(instances[0].orientation, plane.location.orientation)


def test_an_enhanced_multiframe_is_still_refused(tmp_path):
    """Only the objects this app writes are unpacked; the rest are a different thing."""
    dataset = write_slices(tmp_path, frames=1)[0]
    dataset.NumberOfFrames = 4
    dataset.save_as(next((pl.Path(tmp_path) / "ul").glob("*.dcm")))

    with pytest.raises(ValueError, match="enhanced multi-frame"):
        dicom.read_series(pl.Path(tmp_path) / "ul")


def test_an_instance_is_timed_by_the_phase_it_stands_at(tmp_path):
    """A rotation runs through several cycles; a trigger time past one is nothing."""
    reslice = posed_reslice(phantom())
    plane = plane_from_reslice(reslice)

    writer = SliceWriter(context(tmp_path))
    # Four captures of two phases, which is what turning through two cycles
    # without advancing past the end of one looks like.
    for index, phase in enumerate([0, 1, 0, 1]):
        writer.add(index, Frame(image=rgb_frame().image, plane=plane, phase=phase))
    writer.close()

    times = [
        float(dataset.TriggerTime) for dataset in written(pl.Path(tmp_path) / "ul")
    ]

    assert times == [0.0, 50.0, 0.0, 50.0]


def test_an_untimed_frame_falls_back_to_its_place_in_the_capture(tmp_path):
    datasets = write_slices(tmp_path, frames=3)

    assert [float(d.TriggerTime) for d in datasets] == [0.0, 50.0, 100.0]
