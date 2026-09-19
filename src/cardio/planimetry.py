"""The geometry of a region traced on a cut: where it sits, and what it encloses.

Arrays in, arrays out.  The area of a traced region is a planar quantity, so
everything here works in the cut's own two dimensions -- the millimetres a
point is measured along the cut's right and up directions -- which is both what
a click gives and what a shoelace sum wants.

``Cut`` is the one piece that crosses back into the patient, and it lives here
rather than beside the document model for the same reason ``IndexOrder`` lives
in ``orientation`` rather than in ``rotation``: it is a thing the geometry is
defined in terms of, and the document is what is written in terms of it.
"""

# System
import enum
import math

# Third Party
import numpy as np
import pydantic as pc

# Internal
from .reslice import VIEW_TRANSFORMS


class ContourStyle(str, enum.Enum):
    """How the points of a traced region are joined into a closed curve."""

    POLYGON = "polygon"
    SPLINE = "spline"


# How many points each span between two control points is drawn as.  Sampling
# a curve is an approximation of it, so what settles this is where the area
# stops moving: for a region traced with eight points it is within a thousandth
# of the curve's own area by here, and finer sampling buys a smoother picture
# rather than a different number.
SPLINE_SAMPLES = 16

# How far out of a plane a cut may sit and still be showing it.  One wheel
# click travels a millimetre, so half of one is under anything a hand does on
# purpose and under the finest spacing anybody reformats a study at.
DEPTH_TOLERANCE = 0.5

# How nearly parallel two normals have to be to be the same direction, as
# ``1 - |cos|``, which is about eight tenths of a degree.  Wide enough that
# composing a pose twice by different routes still reads as one plane, and far
# below any turn somebody makes on purpose.
NORMAL_TOLERANCE = 1e-4

# How far a press may travel and still be a click rather than a drag.
CLICK_SLOP = 3.0

# How near a press has to land to be reaching for a point of a region, or for
# the contour between two of them.  Wider than the click slop, and for the
# opposite reason: that one is how still a hand has to be to mean one place,
# this one is how near it has to be to mean one thing.
GRAB_RADIUS = 8.0


class Cut(pc.BaseModel):
    """Where a cut sits in the patient: its centre, and the two directions a
    point on it is measured along.

    Always ITK/LPS millimetres, whatever index order the document holding it is
    written in.  Nobody types these: they are read off the composition VTK was
    handed, so there is no user convention for them to be in.
    """

    model_config = pc.ConfigDict(extra="forbid")

    origin: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]

    @property
    def normal(self) -> np.ndarray:
        """The out-of-plane axis, as a unit vector.

        ``right`` and ``up`` come from a rotation composed onto an axcode
        frame, so they are orthonormal and this needs no normalizing.

        Which way along that axis it points is not the reslice's: two of the
        three view frames are left-handed -- the same disagreement
        ``rotate_view`` settles with a determinant -- so this is the reslice's
        third axis for one of them and its opposite for the other two.  Nothing
        reads the sign.  ``same_plane`` is the only caller and it compares
        undirected axes, because a plane has no front.
        """
        return np.cross(self.right, self.up)

    @property
    def basis(self) -> np.ndarray:
        """The 3x2 whose columns are ``right`` and ``up``."""
        return np.column_stack([self.right, self.up])


def cut_from(origin, rotation, view: str) -> Cut:
    """The cut ``ResliceSet.set_pose`` would take, without taking it.

    The same composition ``set_pose`` makes -- ``rotation @
    VIEW_TRANSFORMS[view]`` -- whose first two columns are the directions the
    output's x and y advance along.  One definition of a view's frame, so a
    region cannot be traced in one basis and read back in another.
    """
    frame = np.asarray(rotation, dtype=float) @ VIEW_TRANSFORMS[view]
    return Cut(
        origin=tuple(float(value) for value in origin),
        right=tuple(float(value) for value in frame[:, 0]),
        up=tuple(float(value) for value in frame[:, 1]),
    )


def to_lps(cut: Cut, points) -> np.ndarray:
    """``(n, 2)`` cut millimetres as ``(n, 3)`` patient millimetres."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return points @ cut.basis.T + np.asarray(cut.origin, dtype=float)


def from_lps(cut: Cut, lps) -> np.ndarray:
    """``(n, 3)`` patient millimetres as ``(n, 2)`` cut millimetres.

    By projection, so a point off the plane lands where it would be seen from
    straight on.  Callers are expected to have asked ``same_plane`` first;
    this does not check, because what it should do about it is theirs to say.
    """
    lps = np.asarray(lps, dtype=float).reshape(-1, 3)
    return (lps - np.asarray(cut.origin, dtype=float)) @ cut.basis


def same_plane(
    a: Cut,
    b: Cut,
    depth: float = DEPTH_TOLERANCE,
    parallel: float = NORMAL_TOLERANCE,
) -> bool:
    """Whether two cuts are the same plane, whatever their in-plane difference.

    Parallel and coincident, and nothing else.  A pan or a roll moves a cut's
    origin and its in-plane axes without leaving the plane, and a region traced
    there is still exactly where the anatomy is -- so those are deliberately
    not compared, and the region stays drawn through both.

    The parallel test takes the absolute cosine, so a normal that has been
    turned right around counts as the same direction: reversing a tile stack
    and turning a cut through half a circle both produce one, and neither is a
    different plane.
    """
    normal = a.normal
    if abs(1.0 - abs(float(normal @ b.normal))) > parallel:
        return False

    offset = np.asarray(b.origin, dtype=float) - np.asarray(a.origin, dtype=float)
    return abs(float(offset @ normal)) <= depth


def polygon_area(points) -> float:
    """The area a closed polyline encloses, by the shoelace sum.

    Unsigned: which way round a region was traced is a fact about the hand that
    traced it, not about the region.  Fewer than three points enclose nothing.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return 0.0

    x, y = points[:, 0], points[:, 1]
    return float(abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2.0)


def closed_spline(points, per_segment: int = SPLINE_SAMPLES) -> np.ndarray:
    """A closed Catmull-Rom curve through every one of ``points``.

    Through rather than near: a vertex somebody placed is a vertex the curve
    passes through, or the picture disagrees with the click that made it.  The
    samples stop short of each segment's far end, which is the next segment's
    near end, so the curve closes without doubling a point.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return points

    before = np.roll(points, 1, axis=0)
    after = np.roll(points, -1, axis=0)
    beyond = np.roll(points, -2, axis=0)

    constant = 2.0 * points
    linear = after - before
    square = 2.0 * before - 5.0 * points + 4.0 * after - beyond
    cube = -before + 3.0 * points - 3.0 * after + beyond

    t = np.linspace(0.0, 1.0, per_segment, endpoint=False).reshape(1, -1, 1)
    curve = 0.5 * (
        constant[:, None, :]
        + linear[:, None, :] * t
        + square[:, None, :] * t**2
        + cube[:, None, :] * t**3
    )
    return curve.reshape(-1, 2)


def contour_points(points, style: ContourStyle) -> np.ndarray:
    """The closed polyline a region is drawn as.

    One function, so that the area and the picture are the same curve rather
    than two curves that ought to agree.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if ContourStyle(style) is ContourStyle.SPLINE:
        return closed_spline(points)
    return points


def contour_area(points, style: ContourStyle) -> float:
    """The area a region encloses, closed the way it is drawn."""
    return polygon_area(contour_points(points, style))


def nearest_vertex(points, probe) -> tuple[int, float]:
    """Which point of a region a probe is nearest, and how far away it is.

    The index is into ``points`` as given, so it is the one an edit removes or
    moves.  A region with no points has nothing to reach for, and says so with
    an infinite distance rather than an index nobody should use.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return -1, math.inf

    distances = np.linalg.norm(points - np.asarray(probe, dtype=float), axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def nearest_segment(
    points, style: ContourStyle, probe
) -> tuple[int, float, np.ndarray]:
    """Where on a region's contour a probe lands: after which point, how far, and where.

    Measured against the curve as it is *drawn* rather than against the polygon
    through the points, so that a press on the outside of a splined region's
    bulge is as near to it as it looks.  What comes back is the point it is
    nearest *on the contour*, which is where a vertex added there belongs -- put
    under the cursor instead, a new point would change the region's shape by
    however far the hand was from the line.

    The index is the control point the span begins at, so inserting after it is
    inserting into the span that was pressed.  For a splined region that means
    mapping a sample of the curve back to the span it was drawn for, which is a
    division because ``closed_spline`` lays its samples out span by span in
    order.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    probe = np.asarray(probe, dtype=float)
    if len(points) < 2:
        return -1, math.inf, probe

    curve = contour_points(points, style)
    starts = curve
    spans = np.roll(curve, -1, axis=0) - starts

    lengths = np.einsum("ij,ij->i", spans, spans)
    along = np.divide(
        np.einsum("ij,ij->i", probe - starts, spans),
        lengths,
        out=np.zeros(len(curve)),
        where=lengths > 0.0,
    )
    feet = starts + np.clip(along, 0.0, 1.0)[:, None] * spans

    distances = np.linalg.norm(feet - probe, axis=1)
    nearest = int(np.argmin(distances))
    per_span = len(curve) // len(points)
    return nearest // per_span, float(distances[nearest]), feet[nearest]


def is_click(start, end, slop: float = CLICK_SLOP) -> bool:
    """Whether a press and its release are near enough to be one click."""
    return math.dist(start, end) <= slop
