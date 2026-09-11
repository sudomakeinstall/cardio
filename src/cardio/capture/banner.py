"""A line of text tagged onto the lower margin of what a capture writes.

The band is appended rather than drawn over the picture.  Nothing a viewport
rendered is covered, and a strip that was never part of the image is visibly
not part of it -- which is the whole point of a line that says the picture is
not for clinical use.

The text is drawn once into a greyscale mask, and each stamper tints that mask
in its own terms: white on black for a picture, the scalars' own extremes for
the data behind one.  Pillow's default font is scalable and ships with Pillow,
so a banner looks the same wherever the app runs without a font to bundle.
"""

# Third Party
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import vtk
from vtk.util import numpy_support as vtknp

# Internal
from .base import image_to_array

# The band is sized from the width, not fixed: a strip legible on a 256 pixel
# capture is a nameplate across a 1920 one.
BAND_SHARE = 0.045
MINIMUM_BAND = 14

# How much of the band the lettering stands in, and how much of the width is
# left clear at either end.
TEXT_SHARE = 0.6
MARGIN_SHARE = 0.03


def band_height(columns: int) -> int:
    """How many rows a banner adds to a picture ``columns`` wide."""
    return max(MINIMUM_BAND, round(columns * BAND_SHARE))


def _fitted(text: str, columns: int, band: int) -> PIL.ImageFont.FreeTypeFont:
    """The largest default-font size that keeps ``text`` inside the margins.

    A banner long enough to overrun even the smallest size is drawn at that
    size and clipped by the band it is drawn into, rather than widening a
    picture to suit its caption.
    """
    room = columns - 2 * round(columns * MARGIN_SHARE)
    size = max(1, round(band * TEXT_SHARE))
    while size > 1:
        font = PIL.ImageFont.load_default(size=size)
        left, _, right, _ = font.getbbox(text)
        if right - left <= room:
            return font
        size -= 1
    return PIL.ImageFont.load_default(size=1)


def coverage(text: str, columns: int) -> np.ndarray:
    """The band as ink, from none at 0 to full at 1, centred in its strip."""
    band = band_height(columns)
    image = PIL.Image.new("L", (columns, band), 0)
    if text:
        PIL.ImageDraw.Draw(image).text(
            (columns / 2, band / 2),
            text,
            fill=255,
            font=_fitted(text, columns, band),
            anchor="mm",
        )
    return np.asarray(image, dtype=np.float32) / 255.0


def stamp_rgb(rgb: np.ndarray, text: str) -> np.ndarray:
    """``rgb`` with the banner below it: white lettering on black.

    Alpha, where the capture carries it, is opaque across the band: the strip
    is part of the picture that gets written, not a hole in it.
    """
    if not text:
        return rgb

    components = rgb.shape[2]
    mask = coverage(text, rgb.shape[1])
    ink = np.rint(mask * 255).astype(rgb.dtype)

    band = np.zeros((mask.shape[0], rgb.shape[1], components), dtype=rgb.dtype)
    band[..., : min(3, components)] = ink[..., None]
    if components == 4:
        band[..., 3] = 255

    return np.ascontiguousarray(np.concatenate([rgb, band], axis=0))


def stamp_scalars(scalars: np.ndarray, text: str) -> np.ndarray:
    """``scalars`` with the banner below them, in the values they already hold.

    Lettering takes the brightest value present and the strip behind it the
    darkest, so the band reads at whatever window the pixels are shown at
    instead of blowing out or vanishing at one of them.  The rows the volume
    was measured from are untouched: the band is appended past the last of
    them, where a DICOM image's position and orientation still refer to the
    first.
    """
    if not text:
        return scalars

    mask = coverage(text, scalars.shape[1])
    low, high = float(scalars.min()), float(scalars.max())

    band = low + mask * (high - low)
    if np.issubdtype(scalars.dtype, np.integer):
        band = np.rint(band)

    return np.ascontiguousarray(
        np.concatenate([scalars, band.astype(scalars.dtype)], axis=0)
    )


def stamp_image(image: vtk.vtkImageData, text: str) -> vtk.vtkImageData:
    """A window capture with the banner below it, as VTK hands one over.

    Stamped through ``image_to_array`` so the band lands below the picture and
    not above it: VTK's buffer runs bottom-up, and the lower margin is the end
    a reader sees rather than the end the buffer starts at.

    A window nothing has sized yet captures as nothing by nothing, and a band
    across the bottom of that would be the whole of the picture.  It is handed
    back untouched instead: a margin needs something to be the margin of.
    """
    columns, rows, _ = image.GetDimensions()
    if not text or not (columns and rows):
        return image

    stamped = stamp_rgb(image_to_array(image), text)
    rows, columns, components = stamped.shape

    result = vtk.vtkImageData()
    result.SetDimensions(columns, rows, 1)
    result.SetSpacing(image.GetSpacing())
    result.SetOrigin(image.GetOrigin())
    result.GetPointData().SetScalars(
        vtknp.numpy_to_vtk(
            np.ascontiguousarray(np.flipud(stamped)).reshape(-1, components),
            deep=True,
            array_type=vtknp.get_vtk_array_type(stamped.dtype),
        )
    )
    return result
