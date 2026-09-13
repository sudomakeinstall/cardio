"""Writing a segmentation as a DICOM Segmentation object.

A label volume is the app's own business until somebody else has to read it.
DICOM SEG is how a segmentation travels: an archive files it against the study
it belongs to, a viewer overlays it on the series it was drawn on, and a report
can point at it.  A directory of NIfTI files does none of that.

highdicom builds the object.  What it wants is the labels as segment numbers,
a description of what each segment is, and the source images the segmentation
was drawn on -- and from those it derives the functional groups, the frame of
reference, the provenance and every module the IOD requires.  There is no
geometry assembled by hand here at all, which for an object whose whole content
is geometry is the point of using it.

The planes are stated explicitly rather than inferred from the source images.
A label volume is read and rendered on its own grid, and that grid is what the
app measured: taking the geometry from the source series instead would be
asserting a correspondence nothing here has checked.

One instance per cardiac phase.  highdicom's fourth pixel axis is segments
rather than time, so a moving segmentation is a series of objects, which is
also how a viewer steps through one.
"""

# System
import copy as copy_module
import logging

# Third Party
import highdicom as hd
import numpy as np
import vtk.util.numpy_support as vtk_np
from pydicom.sr.codedict import codes

# Internal
from . import uid
from .equipment import Equipment
from .series import DESCRIPTION_LIMIT

logger = logging.getLogger(__name__)

# What a segment is, when the configuration has not said what it is.  The
# category is what every anatomical segment is; the type is the specific thing,
# and "tissue" is the honest answer when nobody has named it.
SEGMENT_CATEGORY = codes.SCT.AnatomicalStructure
UNTYPED_SEGMENT = codes.SCT.Tissue

# SNOMED CT, which is what a bare concept id in the configuration means.
CODE_SCHEME = "SCT"

DEFAULT_DESCRIPTION = "cardio segmentation"

# BINARY rather than LABELMAP.  A label map is the tidier object and the newer
# SOP class, and that is the whole trouble with it: Segmentation Storage is
# what a deployed archive and a deployed viewer actually read today, and an
# object nothing can open is not an export.  The labels are still handed over
# as one map; highdicom expands them into a segment apiece.
SEGMENTATION_TYPE = hd.seg.SegmentationTypeValues.BINARY

# The Type 2 elements highdicom reads straight off a source image.  A
# conforming acquisition carries all of them, empty where it does not know
# them; a research export may not, and an absent one is an AttributeError
# rather than a message about what is missing.
EXPECTED_OF_SOURCE = (
    "PatientID",
    "PatientName",
    "PatientBirthDate",
    "PatientSex",
    "StudyID",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
)


def conforming(datasets: list) -> list:
    """Copies of ``datasets`` carrying the Type 2 elements they should have.

    Filled empty, never invented: an element the source did not carry is one
    nobody knows the value of, and empty is what the standard already says
    such a field looks like.  Which fields those were is the pre-flight
    check's business to report, not this function's to guess at.
    """
    filled = []
    for dataset in datasets:
        # Deep, because ``Dataset.copy`` shares the element dict: filling a
        # field on a shallow copy fills it on the caller's dataset too, and
        # what leaves here would then be what the reader believes it read.
        copy = copy_module.deepcopy(dataset)
        for name in EXPECTED_OF_SOURCE:
            if name not in copy:
                setattr(copy, name, None)
        filled.append(copy)
    return filled


def segment_type(group):
    """What one structure is, as a coded concept.

    An unconfigured group is typed as tissue rather than left out: the
    attribute is Type 1, and a segment that says only "tissue" is at least not
    claiming to be something it is not.
    """
    if not group.code:
        logger.warning(
            f"Structure {group.name!r} has no code, so its segment is typed "
            "only as tissue; set its code to say what it is."
        )
        return UNTYPED_SEGMENT
    return hd.sr.CodedConcept(
        value=str(group.code), scheme_designator=CODE_SCHEME, meaning=group.name
    )


def display_color(group) -> hd.color.CIELabColor | None:
    """The colour a viewer should draw the segment in, from the curve's own."""
    if group.color is None:
        return None
    return hd.color.CIELabColor(*_cielab(group.color))


def describe_segments(groups) -> list[hd.seg.SegmentDescription]:
    """One description per configured structure, numbered as the labels are."""
    return [
        hd.seg.SegmentDescription(
            segment_number=number,
            segment_label=group.name[:DESCRIPTION_LIMIT],
            segmented_property_category=SEGMENT_CATEGORY,
            segmented_property_type=segment_type(group),
            # The labels come off an editor rather than out of a classifier, and
            # saying which would be a claim about how they were made.
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
            display_color=display_color(group),
        )
        for number, group in enumerate(groups, start=1)
    ]


def label_map(image_data, groups) -> np.ndarray:
    """The label image as segment numbers, ``(slices, rows, columns)``.

    A segment number is a position in the configured structures rather than a
    label value: DICOM numbers segments from one without gaps, and the labels
    an editor wrote are whatever the editor used.

    Groups are applied in order, so a label named by two of them belongs to the
    later -- the same rule the volume curves measure by.
    """
    columns, rows, slices = image_data.GetDimensions()
    values = vtk_np.vtk_to_numpy(image_data.GetPointData().GetScalars()).reshape(
        slices, rows, columns
    )

    segments = np.zeros(values.shape, dtype=np.uint8)
    for number, group in enumerate(groups, start=1):
        segments[np.isin(values, group.labels)] = number
    return segments


def geometry_of(image_data):
    """Where each slice of a label image sits, and how finely it is sampled.

    Stated from the label image itself, which is the grid the app measured on.
    """
    direction = image_data.GetDirectionMatrix()
    axes = np.array(
        [[direction.GetElement(row, column) for column in range(3)] for row in range(3)]
    )
    spacing = np.asarray(image_data.GetSpacing())
    origin = np.asarray(image_data.GetOrigin())
    slices = image_data.GetDimensions()[2]

    positions = [
        hd.PlanePositionSequence(
            coordinate_system=hd.CoordinateSystemNames.PATIENT,
            image_position=[float(v) for v in origin + axes[:, 2] * spacing[2] * k],
        )
        for k in range(slices)
    ]
    orientation = hd.PlaneOrientationSequence(
        coordinate_system=hd.CoordinateSystemNames.PATIENT,
        image_orientation=[float(v) for v in np.concatenate([axes[:, 0], axes[:, 1]])],
    )
    measures = hd.PixelMeasuresSequence(
        # DICOM measures a pixel down a column first, then along a row.
        pixel_spacing=[float(spacing[1]), float(spacing[0])],
        slice_thickness=float(spacing[2]),
    )
    return positions, orientation, measures


def write_segmentation(
    segmentation,
    source,
    directory,
    groups,
    equipment: Equipment | None = None,
    uid_root: str = uid.DEFAULT_ROOT,
    series_number: int = 300,
    series_description: str = DEFAULT_DESCRIPTION,
) -> list:
    """Every frame of ``segmentation`` as a DICOM SEG series.

    ``source`` is what the segmentation was drawn on, and has to have been read
    from DICOM: a Segmentation object refers to the images it segments by UID,
    and there is nothing to refer to otherwise.
    """
    if source is None or not source.instances:
        raise ValueError(
            "A DICOM segmentation refers to the images it segments by UID, so "
            "the volume it was drawn on has to have been read from DICOM."
        )
    if not groups:
        raise ValueError(
            "A segment is one configured structure, so there is nothing to "
            "write until volumetry.groups names at least one."
        )

    equipment = equipment or Equipment()
    directory.mkdir(parents=True, exist_ok=True)
    series_uid = uid.generate(uid_root)
    references = conforming(source.datasets)
    descriptions = describe_segments(groups)

    written = []
    for number in range(len(segmentation._label_images)):
        image_data = segmentation.mpr_image_data(number)
        positions, orientation, measures = geometry_of(image_data)

        dataset = hd.seg.Segmentation(
            source_images=references,
            pixel_array=label_map(image_data, groups),
            segmentation_type=SEGMENTATION_TYPE,
            segment_descriptions=descriptions,
            series_instance_uid=series_uid,
            series_number=series_number,
            sop_instance_uid=uid.generate(uid_root),
            instance_number=number + 1,
            manufacturer=equipment.manufacturer,
            manufacturer_model_name=equipment.model_name,
            software_versions=equipment.software_versions,
            device_serial_number=equipment.device_serial_number,
            series_description=series_description[:DESCRIPTION_LIMIT],
            plane_positions=positions,
            plane_orientation=orientation,
            pixel_measures=measures,
            omit_empty_frames=False,
        )
        uid.stamp(dataset, uid_root)

        path = directory / f"{number:04d}.dcm"
        dataset.save_as(path, enforce_file_format=True)
        written.append(path)

    logger.info(
        f"{len(written)} segmentation instance(s) of {len(groups)} segment(s) "
        f"written to {directory}."
    )
    return written


def _cielab(rgb) -> tuple[float, float, float]:
    """One sRGB colour in [0, 1] as CIE L*a*b*, which is what DICOM stores."""
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in rgb
    ]
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    # D65, which is the white point sRGB is defined against.
    white = np.array([0.95047, 1.0, 1.08883])
    x, y, z = (matrix @ np.asarray(linear)) / white

    def f(value):
        return value ** (1 / 3) if value > 0.008856 else 7.787 * value + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)
