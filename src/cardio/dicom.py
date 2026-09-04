"""Reading a time-resolved DICOM series as a sequence of 3D frames.

A cine acquisition arrives as one file per slice per cardiac phase, all sharing
a SeriesInstanceUID.  Recovering the volumes means deciding, for each file,
which slice of which frame it is.

Slice identity comes from geometry rather than from numbering: the position of
a file along the acquisition normal is the one thing every conforming writer
gets right, whereas InstanceNumber may run slice-major, phase-major or neither.
Grouping by position and then ordering each group in time therefore recovers
the frames whatever order the numbering happens to be in, and a series that
does not divide evenly is reported rather than silently reshaped.

Headers are read with pydicom, which is cheap because the pixel data is skipped;
the pixels themselves are decoded by ITK, so compressed transfer syntaxes work
and the geometry is derived the same way it is for any other ITK series.
"""

# System
import dataclasses as dc
import logging
import pathlib as pl

# Third Party
import itk
import numpy as np
import pydicom as pd

# Distinct slice positions closer together than this are treated as one slice,
# absorbing the rounding in a writer's ImagePositionPatient.
POSITION_TOLERANCE = 1e-3

HEADER_TAGS = [
    "SeriesInstanceUID",
    "SeriesDescription",
    "SeriesNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "InstanceNumber",
    "TriggerTime",
    "TemporalPositionIdentifier",
    "AcquisitionTime",
    "NumberOfFrames",
    "PixelSpacing",
    "SliceThickness",
]


@dc.dataclass(frozen=True)
class Instance:
    """The header fields of one DICOM file needed to place it in the series."""

    path: pl.Path
    series_uid: str
    series_description: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float, float, float]
    instance_number: int | None
    trigger_time: float | None
    temporal_position: int | None
    acquisition_time: str | None
    pixel_spacing: tuple[float, float]
    slice_thickness: float


def _read_header(path: pl.Path) -> Instance | None:
    """One file as an ``Instance``, or None if it is not a usable DICOM image."""
    try:
        dataset = pd.dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS)
    except (pd.errors.InvalidDicomError, OSError):
        return None

    series_uid = getattr(dataset, "SeriesInstanceUID", None)
    position = getattr(dataset, "ImagePositionPatient", None)
    orientation = getattr(dataset, "ImageOrientationPatient", None)

    if series_uid is None or position is None or orientation is None:
        return None

    number_of_frames = getattr(dataset, "NumberOfFrames", 1) or 1
    if int(number_of_frames) > 1:
        raise ValueError(
            f"{path} is an enhanced multi-frame DICOM ({number_of_frames} frames "
            "in one file), which is not supported."
        )

    def optional(name, cast):
        value = getattr(dataset, name, None)
        return None if value is None else cast(value)

    return Instance(
        path=path,
        series_uid=str(series_uid),
        series_description=str(getattr(dataset, "SeriesDescription", "") or ""),
        position=tuple(float(v) for v in position),
        orientation=tuple(float(v) for v in orientation),
        instance_number=optional("InstanceNumber", int),
        trigger_time=optional("TriggerTime", float),
        temporal_position=optional("TemporalPositionIdentifier", int),
        acquisition_time=optional("AcquisitionTime", str),
        pixel_spacing=tuple(
            float(v) for v in getattr(dataset, "PixelSpacing", None) or (1.0, 1.0)
        ),
        slice_thickness=float(getattr(dataset, "SliceThickness", None) or 1.0),
    )


def scan(directory: pl.Path) -> list[Instance]:
    """Every readable DICOM image under ``directory``, recursively.

    Files pydicom cannot read are skipped rather than raised on, so a DICOMDIR
    or a stray note alongside the images is not fatal.
    """
    instances = []
    for path in sorted(directory.glob("**/*")):
        if not path.is_file():
            continue
        instance = _read_header(path)
        if instance is not None:
            instances.append(instance)
    return instances


def by_series(instances: list[Instance]) -> dict[str, list[Instance]]:
    series: dict[str, list[Instance]] = {}
    for instance in instances:
        series.setdefault(instance.series_uid, []).append(instance)
    return series


def select_series(
    instances: list[Instance], series_uid: str | None = None
) -> list[Instance]:
    """The one series to load, by UID when the directory holds more than one."""
    series = by_series(instances)

    if series_uid is not None:
        if series_uid not in series:
            available = ", ".join(sorted(series)) or "none"
            raise ValueError(
                f"No series {series_uid} in this directory. Available: {available}"
            )
        return series[series_uid]

    if len(series) > 1:
        listing = "\n".join(
            f"  {uid}  ({len(members)} images)"
            f"{'  ' + members[0].series_description if members[0].series_description else ''}"
            for uid, members in sorted(series.items())
        )
        raise ValueError(
            f"This directory holds {len(series)} DICOM series; set series_uid to "
            f"choose one:\n{listing}"
        )

    return next(iter(series.values()))


def slice_normal(instance: Instance) -> np.ndarray:
    """The acquisition plane normal, from the two in-plane direction cosines."""
    orientation = np.array(instance.orientation, dtype=np.float64)
    return np.cross(orientation[:3], orientation[3:])


def slice_position(instance: Instance, normal: np.ndarray) -> float:
    """How far along the normal this image sits; equal for slices of one location."""
    return float(np.array(instance.position, dtype=np.float64) @ normal)


def temporal_key(instance: Instance) -> tuple:
    """Orders the images at one slice location through the cardiac cycle.

    The keys are tried in order of how directly they describe cine timing, and
    each is paired with a flag so that images carrying a key sort ahead of any
    that do not, rather than the two interleaving.
    """
    return (
        (instance.trigger_time is None, instance.trigger_time or 0.0),
        (instance.temporal_position is None, instance.temporal_position or 0),
        (instance.acquisition_time is None, instance.acquisition_time or ""),
        (instance.instance_number is None, instance.instance_number or 0),
    )


def frame_instances(instances: list[Instance]) -> list[list[Instance]]:
    """The series' images as ``[frame][slice]``, ordered in time then in space."""
    if not instances:
        raise ValueError("No DICOM images to group.")

    normal = slice_normal(instances[0])

    locations: list[tuple[float, list[Instance]]] = []
    for instance in instances:
        position = slice_position(instance, normal)
        for known, members in locations:
            if abs(known - position) <= POSITION_TOLERANCE:
                members.append(instance)
                break
        else:
            locations.append((position, [instance]))

    locations.sort(key=lambda item: item[0])

    counts = {len(members) for _, members in locations}
    if len(counts) > 1:
        summary = ", ".join(
            f"{sum(1 for _, m in locations if len(m) == count)} slice(s) with {count}"
            for count in sorted(counts)
        )
        raise ValueError(
            "This series does not hold the same number of images at every slice "
            f"location, so its frames cannot be recovered: {summary}."
        )

    frame_count = counts.pop()
    for _, members in locations:
        members.sort(key=temporal_key)

    return [
        [members[frame] for _, members in locations] for frame in range(frame_count)
    ]


def _read_frame(frame: list[Instance]):
    """One frame's slices as a 3D ITK image.

    ITK's series reader derives the geometry from the slice positions, which it
    cannot do for a single-slice acquisition -- a cine of one plane, which is
    ordinary in cardiac MR -- so that case is assembled from its own header.
    """
    if len(frame) > 1:
        return itk.imread([str(instance.path) for instance in frame])

    instance = frame[0]
    array = itk.array_from_image(itk.imread(str(instance.path)))
    if array.ndim == 2:
        array = array[np.newaxis]

    row_spacing, column_spacing = instance.pixel_spacing
    orientation = np.array(instance.orientation, dtype=np.float64)
    direction = np.column_stack(
        [orientation[:3], orientation[3:], slice_normal(instance)]
    )

    volume = itk.image_from_array(np.ascontiguousarray(array))
    volume.SetOrigin([float(v) for v in instance.position])
    volume.SetSpacing([column_spacing, row_spacing, instance.slice_thickness])
    volume.SetDirection(itk.matrix_from_array(direction))
    return volume


def read_series(directory: pl.Path, series_uid: str | None = None) -> list:
    """A DICOM series directory as a list of 3D ITK images, one per frame."""
    instances = scan(directory)
    if not instances:
        raise ValueError(f"No DICOM images found in {directory}")

    selected = select_series(instances, series_uid)
    frames = frame_instances(selected)

    logging.info(
        f"{directory}: {len(frames)} frame(s) of {len(frames[0])} slice(s) "
        f"from series {selected[0].series_uid}."
    )

    return [_read_frame(frame) for frame in frames]
