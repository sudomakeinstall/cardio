"""What a loaded object is, as a table the metadata sheet can render.

The interesting facts about an object differ by where it came from -- a DICOM
series has a patient and a protocol, a NIfTI file has a qform code, a mesh has
neither -- so the description is built as a list of sections rather than a
fixed record.  Nothing here touches trame: the sheet is a dumb renderer over
these rows, and they can be checked without a server.
"""

# System
import dataclasses as dc

# Third Party
import numpy as np


@dc.dataclass(frozen=True)
class Row:
    """One field of one section."""

    name: str
    value: str


@dc.dataclass(frozen=True)
class Section:
    """A titled group of rows."""

    title: str
    rows: list[Row]


@dc.dataclass(frozen=True)
class ObjectMetadata:
    """Everything the sheet shows for one object."""

    key: str
    label: str
    kind: str
    sections: list[Section]

    @property
    def title(self) -> str:
        """How the object is named in the sheet's dropdown."""
        return f"{self.label} ({self.kind})"


def _numbers(values, places: int = 3) -> str:
    """A short vector, rounded to something a table column can hold."""
    return "  ".join(f"{float(v):.{places}g}" for v in values)


def _matrix(matrix: np.ndarray, places: int = 3) -> str:
    """A 3x3 matrix as three lines, one per row."""
    return "\n".join(_numbers(row, places) for row in np.asarray(matrix))


def source_section(obj) -> Section:
    """Where the object was read from, and how much of it there is."""
    rows = [
        Row("Label", obj.label),
        Row("Kind", obj.kind),
        Row("Format", obj.source.format if obj.source else "OBJ"),
        Row("Directory", str(obj.directory)),
        Row("Frames", str(len(obj.actors))),
    ]

    if obj.pattern is not None and obj.file_paths is None:
        rows.append(Row("Pattern", obj.pattern))
    if obj.file_paths is not None:
        rows.append(Row("Files", str(len(obj.file_paths))))
    if obj.series_uid is not None:
        rows.append(Row("Series UID", obj.series_uid))

    return Section("Source", rows)


def geometry_section(geometry) -> Section:
    """An image's sampling and placement, as it is rendered."""
    return Section(
        "Geometry",
        [
            Row("Size (voxels)", _numbers(geometry.size, places=6)),
            Row("Spacing (mm)", _numbers(geometry.spacing)),
            Row("Origin (mm, LPS)", _numbers(geometry.origin)),
            Row("Extent (mm)", _numbers(geometry.extent)),
            Row("Direction", _matrix(geometry.direction)),
            Row("Orientation", geometry.axcode),
            Row("Voxel type", geometry.voxel_type),
            Row("Intensity range", _numbers(geometry.intensity_range, places=6)),
        ],
    )


def header_section(source) -> Section | None:
    """The format's own header, or None when it carried none."""
    if not source.header:
        return None

    return Section(
        f"{source.format} header",
        [Row(name, value) for name, value in source.header.items()],
    )


def handedness_section(source) -> Section:
    """Whether the load flipped the image to make it renderable.

    A left-handed direction is legal but draws nothing, so it is silently
    corrected; saying so here is the only way to tell it happened.
    """
    applied = source.right_handed_correction
    return Section(
        "Handedness",
        [
            Row(
                "Right-handed correction",
                "applied: the slice axis was reversed" if applied else "not needed",
            )
        ],
    )


def surface_section(obj) -> Section | None:
    """The first frame's polydata, for the objects that draw one.

    A volume's mapper also has an input, but it is image data rather than a
    surface, so the type is checked rather than assumed.
    """
    if not obj.actors:
        return None

    polydata = obj.actors[0].GetMapper().GetInput()
    if polydata is None or not polydata.IsA("vtkPolyData"):
        return None

    rows = [
        Row("Points", str(polydata.GetNumberOfPoints())),
        Row("Cells", str(polydata.GetNumberOfCells())),
        Row("Bounds (mm)", _numbers(polydata.GetBounds())),
    ]

    for name, data in (
        ("Point arrays", polydata.GetPointData()),
        ("Cell arrays", polydata.GetCellData()),
    ):
        arrays = [data.GetArrayName(i) for i in range(data.GetNumberOfArrays())]
        if arrays:
            rows.append(Row(name, ", ".join(arrays)))

    return Section("Surface", rows)


def label_section(obj) -> Section | None:
    """The label values a segmentation's first frame holds."""
    if not obj.actors:
        return None

    scalars = obj.mpr_image_data(0).GetPointData().GetScalars()
    if scalars is None:
        return None

    low, high = scalars.GetRange()
    rows = [Row("Label range", f"{int(low)} to {int(high)}")]
    if obj.include_labels is not None:
        rows.append(
            Row("Included labels", ", ".join(str(v) for v in obj.include_labels))
        )

    return Section("Labels", rows)


def describe(obj) -> ObjectMetadata:
    """One object as the sheet shows it."""
    sections = [source_section(obj)]

    if obj.source is not None:
        sections.append(geometry_section(obj.source.geometry))
        sections.append(handedness_section(obj.source))
        sections.append(header_section(obj.source))

    if obj.kind == "segmentation":
        sections.append(label_section(obj))

    sections.append(surface_section(obj))

    return ObjectMetadata(
        # A mesh and a volume may carry the same label, so the kind is part of
        # the key the sheet switches on.
        key=f"{obj.kind}:{obj.label}",
        label=obj.label,
        kind=obj.kind,
        sections=[section for section in sections if section is not None],
    )


def describe_scene(scene) -> list[ObjectMetadata]:
    """Every renderable object in the scene, in the order the scene lists them."""
    return [describe(obj) for obj in scene.renderables]
