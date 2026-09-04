"""The tile grid as one image: several cuts, one scale, no place.

Each displayed tile is autocropped, which gives it its own extent and its own
anisotropic spacing -- a 26-degree turn of an axis-aligned cut through a 1x2mm
volume comes out at 1.19 x 1.81mm -- so pasting the displayed tiles together
would leave the result with no single ``PixelSpacing`` to declare.  The export
therefore resamples every tile onto one shared isotropic grid instead.

What survives is what a mosaic can honestly carry: the values, so it can be
windowed, and the scale, so it can be measured within a tile.  It has no
position in the patient, because its tiles were cut at different poses, and it
says so by carrying none rather than by naming one of them.
"""

# System
import math

# Third Party
import numpy as np
import vtk

# Internal
from ..orientation import create_vtk_reslice_matrix
from ..reslice import configure_reslice
from .base import Plane
from .geometry import scalars_2d

# The value outside the volume, matching what the MPR views reslice with.
BACKGROUND_LEVEL = -1000.0


def _posed(image_data, pose, transform) -> vtk.vtkImageReslice:
    """One tile's reslice, autocropped, so its own extent can be measured."""
    origin, rotation = pose

    reslice = configure_reslice(image_data, "linear", BACKGROUND_LEVEL)
    reslice.SetResliceAxes(create_vtk_reslice_matrix(rotation @ transform, origin))
    return reslice


def reach(reslice) -> tuple[float, float]:
    """How far the autocropped cut runs from its pose, in plane millimetres.

    Read from the pipeline's information rather than from its output: the
    extent is settled before any pixel is resampled, and this pass only needs
    to know how big the tiles have to be.
    """
    reslice.UpdateInformation()
    info = reslice.GetOutputInformation(0)
    extent = info.Get(vtk.vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT())
    spacing = info.Get(vtk.vtkDataObject.SPACING())
    origin = info.Get(vtk.vtkDataObject.ORIGIN())

    spans = []
    for axis in (0, 1):
        low = origin[axis] + extent[2 * axis] * spacing[axis]
        high = origin[axis] + extent[2 * axis + 1] * spacing[axis]
        spans.append(max(abs(low), abs(high)))
    return spans[0], spans[1]


def tile_shape(reslices, spacing: float) -> tuple[int, int]:
    """Rows and columns wide enough for the largest cut, centred on its pose.

    Sized about the pose rather than about each cut's own crop, so every tile
    puts its pose at the same place and the grid stays comparable.
    """
    reaches = [reach(reslice) for reslice in reslices]
    half_columns = math.ceil(max(x for x, _ in reaches) / spacing)
    half_rows = math.ceil(max(y for _, y in reaches) / spacing)
    return 2 * half_rows + 1, 2 * half_columns + 1


def compose(image_data, poses, transform, rows: int, columns: int) -> Plane:
    """One frame's tiles, resampled onto a shared grid and laid out row-major.

    The layout matches what the grid draws -- tile 0 top left, as
    ``tile_views.tile_viewport`` places it -- so the mosaic is recognisable as
    the thing that was on screen.  Each tile is flipped to put its top row
    first, which is where a DICOM viewer draws row zero.
    """
    reslices = [_posed(image_data, pose, transform) for pose in poses]
    spacing = float(min(image_data.GetSpacing()))
    tile_rows, tile_columns = tile_shape(reslices, spacing)

    tiles = []
    for reslice in reslices:
        reslice.AutoCropOutputOff()
        reslice.SetOutputSpacing(spacing, spacing, spacing)
        reslice.SetOutputOrigin(
            -(tile_columns // 2) * spacing, -(tile_rows // 2) * spacing, 0.0
        )
        reslice.SetOutputExtent(0, tile_columns - 1, 0, tile_rows - 1, 0, 0)
        reslice.Update()
        tiles.append(np.flipud(scalars_2d(reslice.GetOutput())))

    # Any gap in the grid takes the darkest value the tiles hold, which is the
    # background they were resliced with, expressed in their own type.
    array = np.full(
        (rows * tile_rows, columns * tile_columns),
        min(tile.min() for tile in tiles),
        tiles[0].dtype,
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        array[
            row * tile_rows : (row + 1) * tile_rows,
            column * tile_columns : (column + 1) * tile_columns,
        ] = tile

    return Plane(
        scalars=array,
        pixel_spacing=(spacing, spacing),
        thickness=spacing,
        location=None,
    )
