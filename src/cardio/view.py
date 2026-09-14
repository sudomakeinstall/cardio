"""How the page opens: which layout is on screen, and in which theme."""

# System
import enum

# Third Party
import pydantic as pc

# Internal
from .camera import Cameras

# The value ``maximized_view`` carries for the unmaximized quad view. Spelled
# once here rather than as a bare "" at every site that means it.
QUAD_LAYOUT = ""

# The controller functions the render views assign as ``ui/layout.py`` builds
# them. ``RENDER_VIEWS`` are the per-view updates the frame path pushes to; the
# other two act on whatever is on screen. Named here rather than at the
# assignment, so that a session with no page can say which of them are allowed
# to have no implementation without importing the page to find out.
RENDER_VIEWS = (
    "ul_update",
    "ll_update",
    "lr_update",
    "volume_update",
    "tile_update",
    "volumetry_update",
)

VIEW_FUNCTIONS = (*RENDER_VIEWS, "view_reset_camera", "view_update")


class Layout(str, enum.Enum):
    """The layouts a viewport can be maximized to, plus the quad view.

    The values are what a config names them; ``state_value`` is what
    ``maximized_view`` carries, which is empty for the quad view.
    """

    QUAD = "quad"
    VOLUME = "volume"
    UL = "ul"
    LL = "ll"
    LR = "lr"
    TILE = "tile"
    VOLUMETRY = "volumetry"

    @property
    def state_value(self) -> str:
        return QUAD_LAYOUT if self is Layout.QUAD else self.value

    @classmethod
    def from_state(cls, value: str | None) -> "Layout":
        """The layout ``maximized_view`` names, quad for its empty string.

        Unset state reads as None rather than raising, so an unbuilt layout
        counts as the quad view it will open in.
        """
        return cls(value) if value else cls.QUAD

    @property
    def shows_slices(self) -> bool:
        """Whether this layout draws the three MPR views.

        The volume rendering, the tile grid and the volumetry charts do not, so
        the reslicing behind those views is wasted work while any of them is on
        screen -- and a cine pays that cost once a frame.
        """
        return self not in (Layout.VOLUME, Layout.TILE, Layout.VOLUMETRY)

    @property
    def shows_reslice(self) -> bool:
        """Whether this layout draws a resampled cut of the volume at all.

        Wider than ``shows_slices`` at one end. A maximized upper-left view draws
        the same cut the quad view does, and the tile grid draws its own along
        the traverse path -- so those want the controls over a cut even though
        they are not the three MPR views.

        The volume rendering has a camera rather than a cut, and the volumetry
        charts have neither: what they draw is a measurement of the labels, and
        no window or level applies to it.
        """
        return self not in (Layout.VOLUME, Layout.VOLUMETRY)

    @property
    def on_screen(self) -> frozenset["Layout"]:
        """The layouts whose view this one actually draws.

        Every viewport's container is built at startup and hidden with ``v_if``,
        so the render windows all exist whatever is showing; this is the only
        thing that says which of them anybody can see.

        Deliberately not ``shows_slices``, which means "this layout resamples
        the cuts": that is true of a maximized upper-left view for all three cuts,
        only one of which is on screen.
        """
        if self is Layout.QUAD:
            return frozenset({Layout.UL, Layout.LL, Layout.LR, Layout.VOLUME})
        return frozenset({self})


class CameraLock(str, enum.Enum):
    """Which MPR view the volume rendering's camera is tied to, if any."""

    FREE = "free"
    UL = "ul"
    LL = "ll"
    LR = "lr"


class DrawerSection(str, enum.Enum):
    """The collapsible sections of the drawer, by the key the accordion tracks.

    ``ORIENTATION``, ``TILES``, ``ZOOM`` and ``VOLUMETRY`` are only built when
    the scene has the objects they control, so naming one of those in a scene
    without them opens nothing.
    """

    PLAYBACK = "playback"
    APPEARANCE = "appearance"
    ORIENTATION = "orientation"
    ZOOM = "zoom"
    TILES = "tiles"
    VOLUMETRY = "volumetry"
    EXPORT = "export"


class Theme(str, enum.Enum):
    """Light or dark, which selects between the two configured backgrounds."""

    LIGHT = "light"
    DARK = "dark"


class View(pc.BaseModel):
    """The layout and theme the app opens in.

    Extras are forbidden, as they are on ``Scene``: a misspelled key in a
    hand-written config should say so rather than quietly do nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    layout: Layout = pc.Field(
        default=Layout.QUAD,
        description="Layout to open in. CLI usage: --view.layout tile",
    )
    theme: Theme = pc.Field(
        default=Theme.DARK,
        description="Theme to open in, which selects one of the two backgrounds.",
    )
    camera_lock: CameraLock = pc.Field(
        default=CameraLock.FREE,
        description="MPR view the volume rendering's camera follows, or free.",
    )
    drawer_sections: list[DrawerSection] = pc.Field(
        default_factory=lambda: [DrawerSection.PLAYBACK],
        description="Drawer sections open on load. CLI usage: --view.drawer_sections \"['playback','tiles']\"",
    )
    help_visible: bool = pc.Field(
        default=False, description="Open with the keyboard shortcut dialog showing"
    )
    metadata_visible: bool = pc.Field(
        default=False,
        description="Open with the scene metadata dialog showing. CLI usage: --view.metadata_visible true",
    )
    console_visible: bool = pc.Field(
        default=False,
        description="Open with the action console showing. CLI usage: --view.console_visible true",
    )
    clip_depth: tuple[float, float] | None = pc.Field(
        default=None,
        description=(
            "Near and far clipping planes of the shared camera. Left unset, it "
            "opens on the camera's own range once the scene is built. "
            'CLI usage: --view.clip_depth "[23, 848]"'
        ),
    )
    cameras: Cameras = pc.Field(
        default_factory=Cameras,
        description=(
            "Where each camera is looking. Left unset, they open wherever "
            "fitting the scene puts them."
        ),
    )

    @property
    def open_sections(self) -> list[str]:
        """The drawer sections as the accordion's v-model spells them."""
        return [s.value for s in self.drawer_sections]
