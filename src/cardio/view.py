"""How the page opens: which layout is on screen, and in which theme."""

# System
import enum

# Third Party
import pydantic as pc

# The value ``maximized_view`` carries for the unmaximized quad view. Spelled
# once here rather than as a bare "" at every site that means it.
QUAD_LAYOUT = ""


class Layout(str, enum.Enum):
    """The layouts a viewport can be maximized to, plus the quad view.

    The values are what a config names them; ``state_value`` is what
    ``maximized_view`` carries, which is empty for the quad view.
    """

    QUAD = "quad"
    VOLUME = "volume"
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"
    TILE = "tile"

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

        The volume rendering and the tile grid do not, so the reslicing behind
        those views is wasted work while either is on screen.
        """
        return self not in (Layout.VOLUME, Layout.TILE)


class CameraLock(str, enum.Enum):
    """Which MPR view the volume rendering's camera is tied to, if any."""

    FREE = "free"
    UL = "UL"
    LL = "LL"
    LR = "LR"


class DrawerSection(str, enum.Enum):
    """The collapsible sections of the drawer, by the key the accordion tracks.

    ``ORIENTATION`` and ``TILES`` are only built when the scene has the objects
    they control, so naming one of those in a scene without them opens nothing.
    """

    PLAYBACK = "playback"
    APPEARANCE = "appearance"
    ORIENTATION = "orientation"
    TILES = "tiles"
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

    @property
    def open_sections(self) -> list[str]:
        """The drawer sections as the accordion's v-model spells them."""
        return [s.value for s in self.drawer_sections]
