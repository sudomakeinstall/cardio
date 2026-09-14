import logging
import pathlib as pl
import typing as ty

import numpy as np
import pydantic as pc
import pydantic_settings as ps
import vtk

from .capture import CaptureFormat, Equipment, SeriesTags
from .capture.uid import DEFAULT_ROOT as DEFAULT_UID_ROOT
from .mesh import Mesh
from .mpr_views import MPRViews
from .playback import Playback
from .rotation import RotationSequence
from .segmentation import Segmentation
from .snap import Snap
from .tile import Tile
from .tile_views import TileViews
from .types import RGBColor
from .view import View
from .volume import Volume
from .volumetry import Volumetry
from .volumetry_views import VolumetryViews
from .window_level import presets
from .zoom import Zoom

logger = logging.getLogger(__name__)


MeshListAdapter = pc.TypeAdapter(list[Mesh])
VolumeListAdapter = pc.TypeAdapter(list[Volume])
SegmentationListAdapter = pc.TypeAdapter(list[Segmentation])

# Create annotated types for better CLI integration
MeshList = ty.Annotated[
    list[Mesh],
    pc.Field(
        description='List of mesh objects. CLI usage: --meshes \'[{"label":"mesh1","directory":"./data/mesh1"}]\''
    ),
]
VolumeList = ty.Annotated[
    list[Volume],
    pc.Field(
        description='Volume objects. CLI: --volumes \'[{"label":"vol1","directory":"./data/vol1"}]\''
    ),
]
SegmentationList = ty.Annotated[
    list[Segmentation],
    pc.Field(
        description='Segmentation objects. CLI: --segmentations \'[{"label":"seg1","directory":"./data/seg1"}]\''
    ),
]


class Background(pc.BaseModel):
    model_config = pc.ConfigDict(extra="forbid")

    light: RGBColor = pc.Field(
        default=(1.0, 1.0, 1.0),
        description="Background color in light mode.  CLI usage: --background.light '[0.8,0.9,1.0]'",
    )
    dark: RGBColor = pc.Field(
        default=(0.0, 0.0, 0.0),
        description="Background color in dark mode.  CLI usage: --background.dark '[0.1,0.1,0.2]'",
    )


class Scene(ps.BaseSettings):
    model_config = ps.SettingsConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        cli_parse_args=False,
        cli_use_class_docstring=True,
    )

    @classmethod
    def load(cls, config_file=None, cli_source=None, **overrides) -> "Scene":
        """Build a scene from a config file, the command line, or neither.

        Which sources are read is decided in ``settings_customise_sources``,
        which pydantic calls with nothing passed in, so they are handed over as
        class attributes and taken away again.
        """
        cls._cli_source = cli_source
        cls._config_file = config_file
        try:
            return cls(**overrides)
        finally:
            del cls._cli_source
            del cls._config_file

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Get the CLI settings source and config file from the class if available
        cli_source = getattr(cls, "_cli_source", None)
        config_file = getattr(cls, "_config_file", None)

        sources = [init_settings]

        # Add CLI settings source if available
        if cli_source is not None:
            sources.append(cli_source)

        # Add TOML config file source if config file is specified
        if config_file is not None:
            sources.append(
                ps.TomlConfigSettingsSource(settings_cls, toml_file=config_file)
            )

        sources.extend([env_settings, file_secret_settings])
        return tuple(sources)

    current_frame: int = 0
    serialization_directory: pl.Path = pc.Field(
        default=pl.Path("./data"),
        description="Base directory for all serialized data (screenshots, exports, etc.)",
    )
    timestamp_format: str = pc.Field(
        default="%Y-%m-%d-%H-%M-%S",
        description="Timestamp format for serialized data subdirectories",
    )
    headless_size: tuple[int, int] = pc.Field(
        default=(1024, 1024),
        description=(
            "Size every render window opens at when no browser is sizing it. "
            "CLI usage: --headless-size '[1920,1080]'"
        ),
    )
    background: Background = pc.Field(
        default_factory=Background,
        description='Background colors. CLI usage: \'{"light": [0.8, 0.9, 1.0], "dark": [0.1, 0.1, 0.2]}\'',
    )
    meshes: MeshList = pc.Field(default_factory=list)
    volumes: VolumeList = pc.Field(default_factory=list)
    segmentations: SegmentationList = pc.Field(default_factory=list)
    active_volume_label: str = pc.Field(
        default="",
        description="Label of the volume to use for multi-planar reconstruction",
    )
    mpr_origin: list = pc.Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="MPR origin position [x, y, z] in LPS coordinates",
    )
    mpr_window: float = pc.Field(
        default=800.0, description="Window width for MPR image display"
    )
    mpr_level: float = pc.Field(
        default=200.0, description="Window level for MPR image display"
    )
    mpr_window_level_preset: int | None = pc.Field(
        default=7,
        description=(
            "Window/level preset key for MPR views, or none for a window and "
            "level of their own"
        ),
    )
    mpr_rotation_sequence: RotationSequence = pc.Field(
        default_factory=RotationSequence,
        description="Dynamic rotation sequence for MPR views",
    )
    mpr_rotation_file: pl.Path | None = pc.Field(
        default=None,
        description="Path to TOML file containing rotation configuration to load",
    )
    max_mpr_rotations: int = pc.Field(
        default=20,
        description="Maximum number of MPR rotations supported",
    )
    mpr_crosshairs_enabled: bool = pc.Field(
        default=True, description="Show crosshair lines indicating slice intersections"
    )
    mpr_crosshair_colors: dict = pc.Field(
        default_factory=lambda: {
            "ul": (0.0, 0.5, 1.0),
            "lr": (1.0, 0.3, 0.3),
            "ll": (0.3, 1.0, 0.3),
        },
        description="RGB colors for crosshair lines (keyed by view name)",
    )
    mpr_crosshair_width: float = pc.Field(
        default=1.5, description="Line width for crosshair lines"
    )
    mpr_segmentation_opacity: float = pc.Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Opacity of the segmentation overlays on the MPR and tile views",
    )
    label_percentile: float = pc.Field(
        default=100.0,
        ge=50.0,
        le=100.0,
        description=(
            "How much of a label cloud a measurement taken off it has to cover, "
            "as a percentage. Below 100 the outermost points are ignored, so a "
            "few mislabelled voxels cannot set the whole extent of a zoom fit or "
            "a tile stack. CLI usage: --label-percentile 99.95"
        ),
    )
    screenshot_viewports: list[str] = pc.Field(
        default=["vr", "ul", "ll", "lr", "tile", "volumetry"],
        description=(
            "Viewports to capture in screenshots. Options: vr, ul, ll, lr, "
            "tile, volumetry"
        ),
    )
    capture_format: CaptureFormat = pc.Field(
        default=CaptureFormat.PNG,
        description=(
            "Format a capture is written in. Options: "
            + ", ".join(CaptureFormat)
            + ". CLI usage: --capture-format jpeg"
        ),
    )
    capture_banner: str = pc.Field(
        default="",
        description=(
            "A line tagged onto the lower margin of everything a capture "
            "writes, in a band below the picture rather than over it. Empty "
            'adds nothing. CLI usage: --capture-banner "NOT FOR CLINICAL USE"'
        ),
    )
    capture_series: SeriesTags = pc.Field(
        default_factory=SeriesTags,
        description=(
            "How each viewport's exported DICOM series is named. "
            "CLI usage: --capture_series.ul.number 400 "
            '--capture_series.ul.description "Cine SAX"'
        ),
    )
    research: bool = pc.Field(
        default=False,
        description=(
            "Whether this deployment is sending nothing anywhere. Set, a DICOM "
            "capture is written with warnings even where something about it "
            "would be unsafe to send -- an invented patient and study, or a "
            "UID root the deployment has not registered. Unset, which is the "
            "default, such a capture is refused. CLI usage: --research"
        ),
    )
    uid_root: str = pc.Field(
        default=DEFAULT_UID_ROOT,
        description=(
            "The registered root every UID a capture writes is generated "
            "under. The default is pydicom's own, which is nobody else's to "
            "assert: a DICOM capture under it is refused unless research is "
            "set. CLI usage: --uid-root 1.2.840.99999"
        ),
    )
    capture_equipment: Equipment = pc.Field(
        default_factory=Equipment,
        description=(
            "Who a DICOM capture says it was made by: the General Equipment "
            "module every written instance carries. "
            'CLI usage: --capture_equipment.institution_name "St Elsewhere"'
        ),
    )
    playback: Playback = pc.Field(
        default_factory=Playback,
        description="Playback controls' starting positions. CLI usage: --playback.bpm 75",
    )
    view: View = pc.Field(
        default_factory=View,
        description="Layout and theme to open in. CLI usage: --view.layout tile",
    )
    snap: Snap = pc.Field(
        default_factory=Snap,
        description='Snap selection at load time. CLI usage: --snap.mode traverse --snap.labels_a "[1]"',
    )
    zoom: Zoom = pc.Field(
        default_factory=Zoom,
        description='Labels the views are fitted to. CLI usage: --zoom.labels "[1]" --zoom.plane ll',
    )
    tile: Tile = pc.Field(
        default_factory=Tile,
        description="Tile grid settings. CLI usage: --tile.rows 2 --tile.cols 4",
    )
    volumetry: Volumetry = pc.Field(
        default_factory=Volumetry,
        description=(
            "What the volumetry chart measures, one entry per curve. "
            'CLI usage: --volumetry.groups \'[{"name":"LV","labels":[6,3,8]}]\''
        ),
    )
    max_volumetry_groups: int = pc.Field(
        default=12,
        description="Maximum number of volumetry structures the drawer can edit",
    )

    # Field validators for JSON string inputs
    @pc.field_validator("meshes", mode="before")
    @classmethod
    def validate_meshes(cls, v):
        if isinstance(v, str):
            return MeshListAdapter.validate_json(v)
        return v

    @pc.field_validator("volumes", mode="before")
    @classmethod
    def validate_volumes(cls, v):
        if isinstance(v, str):
            return VolumeListAdapter.validate_json(v)
        return v

    @pc.field_validator("segmentations", mode="before")
    @classmethod
    def validate_segmentations(cls, v):
        if isinstance(v, str):
            return SegmentationListAdapter.validate_json(v)
        return v

    @pc.model_validator(mode="after")
    def load_rotation_file(self):
        """Read the rotation sequence from the file the config names.

        Read *over* whatever the config spelled: a file is the answer once it
        is named, which is why saving a scene that names one does not write a
        sequence beside it.

        A named file that is not there is refused rather than passed over. It
        used to leave the app open on no rotations at all, which is a strange
        way to be told that a path was mistyped -- and mistyping one is what
        happens when a config is pointed at a new study.
        """
        if self.mpr_rotation_file is not None:
            if not self.mpr_rotation_file.exists():
                raise ValueError(
                    f"mpr_rotation_file: {str(self.mpr_rotation_file)!r} does not exist"
                )

            self.mpr_rotation_sequence = RotationSequence.from_file(
                self.mpr_rotation_file
            )
            if (
                not self.active_volume_label
                and self.mpr_rotation_sequence.metadata.volume_label
            ):
                self.active_volume_label = (
                    self.mpr_rotation_sequence.metadata.volume_label
                )

            if self.mpr_rotation_sequence.mpr_origin:
                self.mpr_origin = list(self.mpr_rotation_sequence.mpr_origin)

        return self

    @pc.model_validator(mode="after")
    def resolve_window_level(self):
        """Reconcile the window and level with the preset that also names one.

        A preset is a window and a level, so a config giving both is saying the
        same thing twice. Given only the preset, it supplies the pair; given
        only the pair, the selection is dropped, the values no longer answering
        to a preset; given both, in disagreement, the config is wrong and says
        so rather than having one quietly win -- which is what used to happen,
        and the pair was what lost.
        """
        preset = presets.get(self.mpr_window_level_preset)
        if preset is None:
            return self

        chosen = {"mpr_window", "mpr_level"} & self.model_fields_set
        if not chosen or preset.matches(self.mpr_window, self.mpr_level):
            self.mpr_window = preset.window
            self.mpr_level = preset.level
            return self

        if "mpr_window_level_preset" in self.model_fields_set:
            raise ValueError(
                f"mpr_window_level_preset {self.mpr_window_level_preset} is "
                f"{preset.name}, window {preset.window} level {preset.level}, "
                f"but mpr_window/mpr_level say {self.mpr_window}/{self.mpr_level}. "
                "Configure one or the other."
            )

        self.mpr_window_level_preset = None
        return self

    # VTK objects as private attributes
    _renderer: vtk.vtkRenderer = pc.PrivateAttr(default_factory=vtk.vtkRenderer)
    _renderWindow: vtk.vtkRenderWindow = pc.PrivateAttr(
        default_factory=vtk.vtkRenderWindow
    )
    _renderWindowInteractor: vtk.vtkRenderWindowInteractor = pc.PrivateAttr(
        default_factory=vtk.vtkRenderWindowInteractor
    )

    # Built lazily: these windows only exist once their layout branch is built
    _mpr_views: MPRViews = pc.PrivateAttr(default=None)
    _tile_views: TileViews = pc.PrivateAttr(default=None)
    _volumetry_views: VolumetryViews = pc.PrivateAttr(default=None)

    @property
    def renderer(self) -> vtk.vtkRenderer:
        return self._renderer

    @property
    def renderWindow(self) -> vtk.vtkRenderWindow:
        return self._renderWindow

    @property
    def renderWindowInteractor(self) -> vtk.vtkRenderWindowInteractor:
        return self._renderWindowInteractor

    @property
    def mpr_views(self) -> MPRViews | None:
        """The three MPR render windows, or None before the quad view is built."""
        return self._mpr_views

    @property
    def tile_views(self) -> TileViews | None:
        """The tile grid's render window, or None before tile mode is built."""
        return self._tile_views

    @property
    def volumetry_views(self) -> VolumetryViews | None:
        """The chart's render window, or None before the charts are built."""
        return self._volumetry_views

    @pc.model_validator(mode="after")
    def setup_scene(self):
        # Validate unique labels
        self._validate_unique_labels()

        # Validate active volume label
        self._validate_active_volume_label()

        # Validate the segmentation the snap selection names
        self._validate_snap_segmentation_label()

        # Configure VTK objects
        self._renderer.SetBackground(
            *self.background.light,
        )
        self._renderWindow.AddRenderer(self._renderer)
        self._renderWindow.SetOffScreenRendering(True)
        self._renderWindowInteractor.SetRenderWindow(self._renderWindow)
        self._renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        for obj in self.renderables:
            obj.configure_actors()

        # Set current frame using nframes property
        self.current_frame = self.current_frame % self.nframes

        # Setup rendering pipeline
        self.setup_pipeline()

        return self

    def _validate_unique_labels(self):
        mesh_labels = [mesh.label for mesh in self.meshes]
        volume_labels = [volume.label for volume in self.volumes]
        segmentation_labels = [seg.label for seg in self.segmentations]

        if len(mesh_labels) != len(set(mesh_labels)):
            duplicates = [
                label for label in set(mesh_labels) if mesh_labels.count(label) > 1
            ]
            raise ValueError(f"Duplicate mesh labels found: {duplicates}")

        if len(volume_labels) != len(set(volume_labels)):
            duplicates = [
                label for label in set(volume_labels) if volume_labels.count(label) > 1
            ]
            raise ValueError(f"Duplicate volume labels found: {duplicates}")

        if len(segmentation_labels) != len(set(segmentation_labels)):
            duplicates = [
                label
                for label in set(segmentation_labels)
                if segmentation_labels.count(label) > 1
            ]
            raise ValueError(f"Duplicate segmentation labels found: {duplicates}")

    def _validate_active_volume_label(self):
        """Validate that active_volume_label refers to an existing volume."""
        if self.active_volume_label and self.volumes:
            volume_labels = [volume.label for volume in self.volumes]
            if self.active_volume_label not in volume_labels:
                raise ValueError(
                    f"Active volume label '{self.active_volume_label}' not found in available volumes: {volume_labels}"
                )
        elif self.active_volume_label and not self.volumes:
            raise ValueError(
                "Active volume label specified but no volumes are available"
            )

    def _validate_snap_segmentation_label(self):
        """Validate that snap.segmentation_label refers to an existing segmentation."""
        if not self.snap.segmentation_label:
            return
        labels = [seg.label for seg in self.segmentations]
        if self.snap.segmentation_label not in labels:
            raise ValueError(
                f"Snap segmentation label '{self.snap.segmentation_label}' not found in available segmentations: {labels}"
            )

    @property
    def renderables(self) -> list:
        """Every drawable object, whatever its type.

        Most per-object handling differs only in which list it loops over, so
        it belongs here rather than being written once per type at each site.
        """
        return [*self.meshes, *self.volumes, *self.segmentations]

    @property
    def screenshot_directory(self) -> pl.Path:
        """Computed property that returns the screenshots subdirectory."""
        return self.serialization_directory / "screenshots"

    @property
    def rotations_directory(self) -> pl.Path:
        """Computed property that returns the rotations subdirectory."""
        return self.serialization_directory / "rotations"

    @property
    def scripts_directory(self) -> pl.Path:
        """Computed property that returns the scripts subdirectory."""
        return self.serialization_directory / "scripts"

    @property
    def volumetry_directory(self) -> pl.Path:
        """Computed property that returns the volumetry subdirectory."""
        return self.serialization_directory / "volumetry"

    @property
    def nframes(self) -> int:
        ns = [len(obj.actors) for obj in self.renderables]
        if not len(ns) > 0:
            logger.warning("No objects were found to display.")
            return 1
        result = int(max(ns))
        ns = np.array(ns)
        if not np.all(ns == ns[0]):
            logger.warning(f"Unequal number of frames: {ns}.")
        return result

    def setup_pipeline(self):
        """Add all actors to the renderer and configure initial visibility."""
        for obj in self.renderables:
            obj.add_to_renderer(self.renderer)

        # Show current frame
        self.show_frame(self.current_frame)
        self.renderer.ResetCamera()

        # Set default camera elevation to -90 degrees
        camera = self.renderer.GetActiveCamera()
        camera.Elevation(-90)

    def setup_mpr_render_windows(self):
        """Initialize MPR render windows when MPR mode is enabled."""
        if self._mpr_views is None:
            self._mpr_views = MPRViews()

    def setup_tile_render_window(self):
        """Initialize the tile grid's render window on first use."""
        if self._tile_views is None:
            self._tile_views = TileViews()
            self._tile_views.set_grid(self.tile.rows, self.tile.cols)

    def setup_volumetry_render_window(self):
        """Initialize the volumetry chart's render window on first use."""
        if self._volumetry_views is None:
            self._volumetry_views = VolumetryViews()

    def hide_all_frames(self):
        for a in self.renderer.GetActors():
            a.SetVisibility(False)
        for a in self.renderer.GetVolumes():
            a.SetVisibility(False)

    def show_frame(self, frame: int):
        for obj in self.renderables:
            actor = obj.frame_actor(frame) if obj.visible else None
            if actor is not None:
                actor.SetVisibility(True)
