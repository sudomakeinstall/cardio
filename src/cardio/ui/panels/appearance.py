"""What the volume rendering draws: visibility, transfer functions, cropping."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ...camera import DEPTH_STEP, depth_top
from ...state import ObjectState
from ..common import (
    EYE_ICONS,
    KIND_ICONS,
    RENDERING_ACTIVE,
    SLIDER_CLASS,
    object_row,
    row_cell,
    target_group,
)


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
            rendering_row(obj)


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


def rendering_row(obj):
    """One object on one line: whether it is drawn, and whether it is cropped.

    The two toggles are icons rather than labelled checkboxes because the row
    is already labelled, once, by the object they both act on. The eye ends the
    row it shares with the eye in the slice group, where the same icon means
    the same thing about a different view.
    """
    keys = ObjectState.of(obj)
    cropping = obj.clipping_enabled
    hidden = obj.kind == "volume" or (cropping and obj.actors)
    eye_on, eye_off = EYE_ICONS

    with object_row(obj.label, KIND_ICONS[obj.kind]):
        if cropping:
            with row_cell():
                vuetify.VCheckbox(
                    v_model=(keys.clipping,),
                    true_icon="mdi-crop",
                    false_icon="mdi-crop-free",
                    title=f"Crop {obj.label} to the bounds it opens to",
                    density="compact",
                    hide_details=True,
                )

        if hidden:
            with row_cell():
                vuetify.VBtn(
                    icon=(
                        f"{keys.detail_panel} ? 'mdi-chevron-up' : 'mdi-chevron-down'",
                    ),
                    click=f"{keys.detail_panel} = !{keys.detail_panel}",
                    title=f"More for {obj.label}",
                    variant="text",
                    density="compact",
                )

        with row_cell():
            vuetify.VCheckbox(
                v_model=keys.visibility,
                true_icon=eye_on,
                false_icon=eye_off,
                title=f"Draw {obj.label} in the rendering",
                density="compact",
                hide_details=True,
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
