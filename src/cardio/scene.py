# System
import logging
import pathlib as pl

# Type adapters for list types to improve CLI integration
from typing import Annotated

# Third Party
import numpy as np
import pydantic as pc
import pydantic_settings as ps
import vtk
from pydantic import Field, PrivateAttr, TypeAdapter, field_validator, model_validator

# Internal
from .mesh import Mesh
from .segmentation import Segmentation
from .types import RGBColor
from .volume import Volume

MeshListAdapter = TypeAdapter(list[Mesh])
VolumeListAdapter = TypeAdapter(list[Volume])
SegmentationListAdapter = TypeAdapter(list[Segmentation])

# Create annotated types for better CLI integration
MeshList = Annotated[
    list[Mesh],
    Field(
        description='List of mesh objects. CLI usage: --meshes \'[{"label":"mesh1","directory":"./data/mesh1"}]\''
    ),
]
VolumeList = Annotated[
    list[Volume],
    Field(
        description='Volume objects. CLI: --volumes \'[{"label":"vol1","directory":"./data/vol1"}]\''
    ),
]
SegmentationList = Annotated[
    list[Segmentation],
    Field(
        description='Segmentation objects. CLI: --segmentations \'[{"label":"seg1","directory":"./data/seg1"}]\''
    ),
]


class Background(pc.BaseModel):
    light: RGBColor = Field(
        default=(1.0, 1.0, 1.0),
        description="Background color in light mode.  CLI usage: --background.light '[0.8,0.9,1.0]'",
    )
    dark: RGBColor = Field(
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

    project_name: str = "Cardio"
    current_frame: int = 0
    screenshot_directory: pl.Path = pl.Path("./data/screenshots")
    screenshot_subdirectory_format: str = "%Y-%m-%d-%H-%M-%S"
    rotation_factor: float = 3.0
    background: Background = Field(
        default_factory=Background,
        description='Background colors. CLI usage: \'{"light": [0.8, 0.9, 1.0], "dark": [0.1, 0.1, 0.2]}\'',
    )
    meshes: MeshList = Field(default_factory=list)
    volumes: VolumeList = Field(default_factory=list)
    segmentations: SegmentationList = Field(default_factory=list)
    mpr_enabled: bool = Field(
        default=False,
        description="Enable multi-planar reconstruction (MPR) mode with quad-view layout",
    )
    active_volume_label: str = Field(
        default="",
        description="Label of the volume to use for multi-planar reconstruction",
    )
    axial_slice: float = Field(
        default=0.5, description="Axial slice position as fraction (0.0 to 1.0)"
    )
    sagittal_slice: float = Field(
        default=0.5, description="Sagittal slice position as fraction (0.0 to 1.0)"
    )
    coronal_slice: float = Field(
        default=0.5, description="Coronal slice position as fraction (0.0 to 1.0)"
    )
    mpr_window: float = Field(
        default=400.0, description="Window width for MPR image display"
    )
    mpr_level: float = Field(
        default=40.0, description="Window level for MPR image display"
    )
    mpr_window_level_preset: int = Field(
        default=1, description="Window/level preset key for MPR views"
    )

    # Field validators for JSON string inputs
    @field_validator("meshes", mode="before")
    @classmethod
    def validate_meshes(cls, v):
        if isinstance(v, str):
            return MeshListAdapter.validate_json(v)
        return v

    @field_validator("volumes", mode="before")
    @classmethod
    def validate_volumes(cls, v):
        if isinstance(v, str):
            return VolumeListAdapter.validate_json(v)
        return v

    @field_validator("segmentations", mode="before")
    @classmethod
    def validate_segmentations(cls, v):
        if isinstance(v, str):
            return SegmentationListAdapter.validate_json(v)
        return v

    # VTK objects as private attributes
    _renderer: vtk.vtkRenderer = PrivateAttr(default_factory=vtk.vtkRenderer)
    _renderWindow: vtk.vtkRenderWindow = PrivateAttr(
        default_factory=vtk.vtkRenderWindow
    )
    _renderWindowInteractor: vtk.vtkRenderWindowInteractor = PrivateAttr(
        default_factory=vtk.vtkRenderWindowInteractor
    )

    # MPR render windows
    _axial_renderWindow: vtk.vtkRenderWindow = PrivateAttr(default=None)
    _coronal_renderWindow: vtk.vtkRenderWindow = PrivateAttr(default=None)
    _sagittal_renderWindow: vtk.vtkRenderWindow = PrivateAttr(default=None)

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
    def axial_renderWindow(self) -> vtk.vtkRenderWindow:
        return self._axial_renderWindow

    @property
    def coronal_renderWindow(self) -> vtk.vtkRenderWindow:
        return self._coronal_renderWindow

    @property
    def sagittal_renderWindow(self) -> vtk.vtkRenderWindow:
        return self._sagittal_renderWindow

    @model_validator(mode="after")
    def setup_scene(self):
        # Validate unique labels
        self._validate_unique_labels()

        # Validate active volume label
        self._validate_active_volume_label()

        # Configure VTK objects
        self._renderer.SetBackground(
            *self.background.light,
        )
        self._renderWindow.AddRenderer(self._renderer)
        self._renderWindow.SetOffScreenRendering(True)
        self._renderWindowInteractor.SetRenderWindow(self._renderWindow)
        self._renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        # Configure all objects
        for mesh in self.meshes:
            mesh.configure_actors()
        for volume in self.volumes:
            volume.configure_actors()
        for segmentation in self.segmentations:
            segmentation.configure_actors()

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

    @property
    def nframes(self) -> int:
        ns = []
        ns += [len(m.actors) for m in self.meshes]
        ns += [len(v.actors) for v in self.volumes]
        ns += [len(s.actors) for s in self.segmentations]
        if not len(ns) > 0:
            logging.warning("No objects were found to display.")
            return 1
        result = int(max(ns))
        ns = np.array(ns)
        if not np.all(ns == ns[0]):
            logging.warning(f"Unequal number of frames: {ns}.")
        return result

    def setup_pipeline(self):
        """Add all actors to the renderer and configure initial visibility."""
        # Add mesh actors
        for mesh in self.meshes:
            for actor in mesh.actors:
                self.renderer.AddActor(actor)

        # Add volume actors
        for volume in self.volumes:
            for actor in volume.actors:
                self.renderer.AddVolume(actor)

        # Add segmentation actors
        for segmentation in self.segmentations:
            for actor in segmentation.actors:
                self.renderer.AddActor(actor)

        # Show current frame
        self.show_frame(self.current_frame)
        self.renderer.ResetCamera()

    def setup_mpr_render_windows(self):
        """Initialize MPR render windows when MPR mode is enabled."""
        if self._axial_renderWindow is None:
            # Create axial render window
            self._axial_renderWindow = vtk.vtkRenderWindow()
            axial_renderer = vtk.vtkRenderer()
            axial_renderer.SetBackground(0.0, 0.0, 0.0)  # Black background
            self._axial_renderWindow.AddRenderer(axial_renderer)
            self._axial_renderWindow.SetOffScreenRendering(True)

            # Create and set interactor for axial
            axial_interactor = vtk.vtkRenderWindowInteractor()
            self._axial_renderWindow.SetInteractor(axial_interactor)

            # Create coronal render window
            self._coronal_renderWindow = vtk.vtkRenderWindow()
            coronal_renderer = vtk.vtkRenderer()
            coronal_renderer.SetBackground(0.0, 0.0, 0.0)  # Black background
            self._coronal_renderWindow.AddRenderer(coronal_renderer)
            self._coronal_renderWindow.SetOffScreenRendering(True)

            # Create and set interactor for coronal
            coronal_interactor = vtk.vtkRenderWindowInteractor()
            self._coronal_renderWindow.SetInteractor(coronal_interactor)

            # Create sagittal render window
            self._sagittal_renderWindow = vtk.vtkRenderWindow()
            sagittal_renderer = vtk.vtkRenderer()
            sagittal_renderer.SetBackground(0.0, 0.0, 0.0)  # Black background
            self._sagittal_renderWindow.AddRenderer(sagittal_renderer)
            self._sagittal_renderWindow.SetOffScreenRendering(True)

            # Create and set interactor for sagittal
            sagittal_interactor = vtk.vtkRenderWindowInteractor()
            self._sagittal_renderWindow.SetInteractor(sagittal_interactor)

    def hide_all_frames(self):
        for a in self.renderer.GetActors():
            a.SetVisibility(False)
        for a in self.renderer.GetVolumes():
            a.SetVisibility(False)

    def show_frame(self, frame: int):
        for mesh in self.meshes:
            if mesh.visible:
                mesh.actors[frame % len(mesh.actors)].SetVisibility(True)
        for volume in self.volumes:
            if volume.visible:
                volume.actors[frame % len(volume.actors)].SetVisibility(True)
        for segmentation in self.segmentations:
            if segmentation.visible:
                actor = segmentation.actors[frame % len(segmentation.actors)]
                actor.SetVisibility(True)
