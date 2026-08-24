"""Per-object visibility, transfer function and clipping controls."""

# Third Party
from trame.widgets import vuetify3 as vuetify

# Internal
from ...state import ObjectState
from ...volume_property_presets import list_volume_property_presets
from ..common import SLIDER_CLASS, SUBPANEL_CLASS

# on/off icons for each type's clip toggle
CLIP_ICONS = {
    "mesh": ("mdi-content-cut", "mdi-content-cut"),
    "volume": ("mdi-cube-outline", "mdi-cube-off-outline"),
    "segmentation": ("mdi-content-cut", "mdi-content-cut"),
}


def clip_depth_panel(server, scene):
    """The camera's shared near/far range."""
    near, far = scene.renderer.GetActiveCamera().GetClippingRange()

    vuetify.VListSubheader("Camera Depth Range")

    vuetify.VRangeSlider(
        v_model=("clip_depth", [near, far]),
        label="Near / Far",
        title="Near and far clipping planes of the shared camera",
        classes=SLIDER_CLASS,
        min=0.1,
        max=far,
        step=far / 100,
        hide_details=True,
        thumb_label=True,
    )


def appearance_panel(server, scene):
    """One section per object type, each with the same per-object controls."""
    for heading, objects in (
        ("Meshes", scene.meshes),
        ("Volumes", scene.volumes),
        ("Segmentations", scene.segmentations),
    ):
        if not objects:
            continue

        vuetify.VListSubheader(heading, classes="text-caption pl-4")
        for obj in objects:
            object_panel(obj, CLIP_ICONS[obj.kind])


def object_panel(obj, clip_icons):
    """Visibility, transfer function and clipping controls for one object."""
    keys = ObjectState.of(obj)

    vuetify.VCheckbox(
        v_model=keys.visibility,
        on_icon="mdi-eye",
        off_icon="mdi-eye-off",
        classes="mx-1",
        hide_details=True,
        label=obj.label,
    )

    if obj.kind == "volume":
        transfer_function_panel(keys)

    if not obj.clipping_enabled:
        return

    on_icon, off_icon = clip_icons
    vuetify.VCheckbox(
        v_model=(keys.clipping, obj.clipping_enabled),
        on_icon=on_icon,
        off_icon=off_icon,
        classes="mx-1 ml-4",
        hide_details=True,
        label=f"Crop {obj.label}" if obj.kind == "volume" else "Crop",
    )

    if obj.actors:
        clip_bounds_panel(keys, obj.combined_bounds)


def transfer_function_panel(keys):
    """Preset picker for a volume, in a collapsed panel."""
    with vuetify.VExpansionPanels(
        v_model=keys.preset_panel,
        flat=True,
        classes=SUBPANEL_CLASS,
    ):
        with vuetify.VExpansionPanel():
            vuetify.VExpansionPanelTitle("Transfer Function")
            with vuetify.VExpansionPanelText():
                with vuetify.VRadioGroup(v_model=keys.preset):
                    for (
                        preset_key,
                        preset_desc,
                    ) in list_volume_property_presets().items():
                        vuetify.VRadio(label=preset_desc, value=preset_key)


def clip_bounds_panel(keys, bounds):
    """The x/y/z clip range sliders, seeded from the object's bounds."""
    with vuetify.VExpansionPanels(
        v_model=keys.clip_panel,
        multiple=True,
        flat=True,
        classes=SUBPANEL_CLASS,
    ):
        with vuetify.VExpansionPanel():
            vuetify.VExpansionPanelTitle("Crop Bounds")
            with vuetify.VExpansionPanelText():
                for key, axis, low in zip(keys.clip_bounds, "XYZ", (0, 2, 4)):
                    minimum, maximum = bounds[low], bounds[low + 1]
                    vuetify.VRangeSlider(
                        v_model=(key, [minimum, maximum]),
                        label=f"{axis} Range",
                        classes=SLIDER_CLASS,
                        min=minimum,
                        max=maximum,
                        step=(maximum - minimum) / 100,
                        hide_details=True,
                        thumb_label=False,
                    )
