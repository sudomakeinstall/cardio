"""What the volume rendering draws: visibility, transfer functions, cropping."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ...camera import DEPTH_STEP, depth_top
from ...state import ObjectState
from ..common import RENDERING_ACTIVE, SLIDER_CLASS, target_group

# What kind of thing a row is, in the only place a row has room to say it: two
# kinds may carry the same label, and the eye beside either is the same eye.
KIND_ICONS = {
    "mesh": "mdi-triangle-outline",
    "volume": "mdi-cube-outline",
    "segmentation": "mdi-shape-outline",
}


def volume_rendering_panel(server, scene):
    """Everything that reaches the volume rendering's renderer and no other.

    The whole group goes away with the view it acts on, rather than each
    control in it saying so for itself.
    """
    with target_group(
        "Volume Rendering",
        "mdi-cube-scan",
        "Controls the 3D rendering only",
        v_if=RENDERING_ACTIVE,
    ):
        clip_depth_slider(scene)
        for obj in scene.renderables:
            object_row(obj)


def clip_depth_slider(scene):
    """The camera's shared near/far range."""
    _, far = scene.renderer.GetActiveCamera().GetClippingRange()

    vuetify.VRangeSlider(
        v_model=("clip_depth",),
        label="Depth",
        title="Near and far clipping planes of the shared camera",
        classes=SLIDER_CLASS,
        min=DEPTH_STEP,
        max=depth_top(far),
        step=DEPTH_STEP,
        hide_details=True,
        thumb_label=True,
    )


def object_row(obj):
    """One object on one line: whether it is drawn, and whether it is cropped.

    The two toggles are icons rather than labelled checkboxes because the row
    is already labelled, once, by the object they both act on.
    """
    keys = ObjectState.of(obj)
    cropping = obj.clipping_enabled
    hidden = obj.kind == "volume" or (cropping and obj.actors)

    with vuetify.VRow(no_gutters=True, classes="align-center flex-nowrap"):
        with vuetify.VCol(cols="auto"):
            vuetify.VCheckbox(
                v_model=keys.visibility,
                true_icon="mdi-eye",
                false_icon="mdi-eye-off",
                title=f"Draw {obj.label} in the rendering",
                density="compact",
                hide_details=True,
            )

        with vuetify.VCol(classes="ps-1 text-body-2 text-truncate"):
            vuetify.VIcon(KIND_ICONS[obj.kind], size="x-small", classes="mr-2")
            html.Span(obj.label)

        if cropping:
            with vuetify.VCol(cols="auto"):
                vuetify.VCheckbox(
                    v_model=(keys.clipping,),
                    true_icon="mdi-crop",
                    false_icon="mdi-crop-free",
                    title=f"Crop {obj.label} to the bounds it opens to",
                    density="compact",
                    hide_details=True,
                )

        if hidden:
            with vuetify.VCol(cols="auto"):
                vuetify.VBtn(
                    icon=(
                        f"{keys.detail_panel} ? 'mdi-chevron-up' : 'mdi-chevron-down'",
                    ),
                    click=f"{keys.detail_panel} = !{keys.detail_panel}",
                    title=f"More for {obj.label}",
                    variant="text",
                    density="compact",
                )

    if hidden:
        object_details(obj, keys)


def object_details(obj, keys):
    """What the row's chevron opens to, indented under the row it belongs to."""
    with html.Div(v_show=keys.detail_panel, classes="ms-6 mb-2"):
        if obj.kind == "volume":
            preset_select(keys)

        if obj.clipping_enabled and obj.actors:
            clip_bounds_sliders(keys, obj.combined_bounds)


def preset_select(keys):
    """The transfer function, as one line rather than a list of radios."""
    vuetify.VSelect(
        v_model=keys.preset,
        items=("volume_preset_items",),
        item_title="title",
        item_value="value",
        label="Transfer Function",
        hide_details=True,
        density="compact",
        classes="mb-2",
    )


def clip_bounds_sliders(keys, bounds):
    """The x/y/z clip ranges, seeded from the object's own extent.

    Grey while the crop is off, which is when moving them changes nothing.
    """
    for key, axis, low in zip(keys.clip_bounds, "XYZ", (0, 2, 4)):
        minimum, maximum = bounds[low], bounds[low + 1]
        vuetify.VRangeSlider(
            v_model=(key,),
            label=f"{axis} Range",
            classes=SLIDER_CLASS,
            min=minimum,
            max=maximum,
            step=(maximum - minimum) / 100,
            hide_details=True,
            thumb_label=False,
            disabled=(f"!{keys.clipping}",),
        )
