"""A region traced on a cut, and the pose it was traced at.

An area measured on a cut means nothing without the cut.  The same contour in
a plane a few degrees off is a different number, so a measurement here carries
the plane it was traced in and the settings that produced that plane, and the
app draws it only while that plane is the one on screen.

The geometry itself is ``planimetry``; this is the document written in terms of
it, the way ``rotation`` is the document written in terms of ``orientation``.
"""

# System
import datetime as dt
import pathlib as pl

# Third Party
import numpy as np
import pydantic as pc
import tomlkit as tk

# Internal
from . import planimetry, toml
from .planimetry import ContourStyle, Cut
from .rotation import RotationSequence
from .snap import Snap
from .tile import Tile


class TileCut(pc.BaseModel):
    """Which tile of the grid a region was traced on, and the grid it was.

    A tile's pose is not a thing the app holds: it is computed from the panel's
    settings and, for the traverse source, from the path the snap selection
    names.  So putting a grid back is putting those settings back, and both are
    carried whole rather than picked over -- a grid recalled from half of them
    would land the region on a different tile, which is the one outcome worth
    ruling out.
    """

    model_config = pc.ConfigDict(extra="ignore")

    index: int = pc.Field(
        default=0, ge=0, description="Which tile, row-major, as the grid numbers them"
    )
    grid: Tile = pc.Field(default_factory=Tile)
    snap: Snap = pc.Field(default_factory=Snap)


class Measurement(pc.BaseModel):
    """One closed region traced on one cut, and the pose it was traced at.

    ``points`` are the cut's own millimetres -- what was clicked -- and ``cut``
    is what turns them back into the patient.  Storing them that way rather
    than as patient coordinates is what makes the area a shoelace sum over the
    numbers a hand actually placed, and what makes drawing the region cost no
    pose to invert; the patient coordinates are one line away through
    ``planimetry.to_lps`` and are derived rather than kept, the way the
    rotation matrix is derived from the rotation sequence.

    The pose is stored twice over, deliberately.  ``cut`` is the geometry --
    what was measured.  ``pose``, ``view``, ``frame`` and ``tile`` are the
    settings -- how to get back.  They are written in the same instant from the
    same state, so they cannot disagree at birth, and if a recall ever fails to
    reproduce the cut -- a lock intervened, a label moved, the segmentation
    changed underneath -- the region stays hidden rather than drawing somewhere
    it does not belong.  The redundancy is a self-check.

    Extras are ignored here, where every other model in the app forbids them.
    ``area`` is written into the file on purpose, so that a reader outside this
    app gets the number without reimplementing the shoelace -- and a computed
    field cannot be assigned, so a model that refused what it did not declare
    would refuse its own output.  What comes back is measured again from the
    points, so a file edited to disagree with itself is read as the points.
    """

    model_config = pc.ConfigDict(extra="ignore")

    name: str = ""
    view: str = pc.Field(description="The viewport traced in: ul, ll, lr or tile")
    frame: int = pc.Field(default=0, ge=0)
    contour: ContourStyle = ContourStyle.POLYGON
    points: list[tuple[float, float]] = pc.Field(
        default_factory=list,
        description="The traced points, in the cut's own millimetres",
    )
    cut: Cut
    pose: RotationSequence = pc.Field(default_factory=RotationSequence)
    tile: TileCut | None = None
    started: str = pc.Field(
        default="",
        description="When the first point of the region was placed",
    )
    timestamp: str = pc.Field(
        default_factory=lambda: dt.datetime.now().astimezone().isoformat(),
        description="When the region was closed",
    )

    @pc.computed_field
    @property
    def area(self) -> float:
        """The area the region encloses, in square millimetres.

        Of the curve it is drawn as rather than of the points alone, so the
        number and the picture are the same region.
        """
        return planimetry.contour_area(self.points, self.contour)

    def lps(self) -> np.ndarray:
        """The traced points in the patient, as ``(n, 3)``."""
        return planimetry.to_lps(self.cut, self.points)

    def on(self, cut: Cut, frame: int) -> bool:
        """Whether this region belongs on ``cut`` as ``frame`` is showing it.

        The plane and the frame, and nothing else.  A cut moved within its own
        plane is still showing the anatomy the region was traced around, so a
        pan or a roll keeps it; the same plane at another phase of the cycle is
        showing different anatomy, so a cine does not.
        """
        return frame == self.frame and planimetry.same_plane(self.cut, cut)


class MeasurementMetadata(pc.BaseModel):
    """What the numbers in a measurement file are written in."""

    coordinate_system: str = "LPS"
    length_units: str = "mm"
    opened: str = pc.Field(
        default="",
        description="When the session the regions were traced in came up",
    )
    timestamp: str = pc.Field(
        default_factory=lambda: dt.datetime.now().astimezone().isoformat(),
        description="When the set was written",
    )
    volume_label: str = ""


class MeasurementSet(pc.BaseModel):
    """Every region traced in a session, as a file that can reopen them.

    Unlike a rotation sequence, this is written in LPS throughout whatever
    index order the rest of a session is spelled in.  Nobody types a cut: it is
    read off the composition VTK was handed.  The poses nested inside carry
    their own metadata and so say for themselves what order they are in.
    """

    model_config = pc.ConfigDict(extra="ignore")

    metadata: MeasurementMetadata = pc.Field(default_factory=MeasurementMetadata)
    measurements: list[Measurement] = pc.Field(default_factory=list)

    def to_toml(self) -> str:
        return toml.dumps(self.model_dump(mode="json", exclude_none=True))

    @classmethod
    def from_toml(cls, toml_content: str) -> "MeasurementSet":
        return cls(**dict(tk.loads(toml_content)))

    @classmethod
    def from_file(cls, path: pl.Path) -> "MeasurementSet":
        return cls.from_toml(pl.Path(path).read_text(encoding="utf-8"))

    def to_file(self, path: pl.Path) -> pl.Path:
        return toml.write(path, self.to_toml())
