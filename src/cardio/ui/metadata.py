"""The per-object metadata sheet."""

# Third Party
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Internal
from ..metadata import describe_scene
from .common import sheet_dialog


def metadata_dialog(scene):
    """What each object in the scene is, toggled with the `i` key.

    Nothing here changes after load, so every object's tables are rendered
    once and the dropdown switches between them in the browser: choosing one
    costs no round trip to the server.
    """
    entries = describe_scene(scene)

    with sheet_dialog(
        "metadata_overlay_visible",
        "Scene Metadata",
        scene.view.metadata_visible,
    ):
        if not entries:
            html.P("No objects in the scene.", classes="text-caption")
            return

        several = len(entries) > 1

        if several:
            vuetify.VSelect(
                v_model=("metadata_object", entries[0].key),
                # A list prop has to arrive as state; passed inline, trame reads
                # it as a (name, default) pair and chokes on the dicts.
                items=(
                    "metadata_pages",
                    [{"title": entry.title, "value": entry.key} for entry in entries],
                ),
                label="Object",
                density="compact",
                hide_details=True,
                classes="mb-4",
            )

        for entry in entries:
            shown = f"metadata_object === '{entry.key}'" if several else True
            with html.Div(v_if=shown):
                for section in entry.sections:
                    html.H3(section.title, classes="text-h6 mb-3")
                    with vuetify.VTable(density="compact", classes="mb-4"):
                        with html.Thead():
                            with html.Tr():
                                html.Th("Field")
                                html.Th("Value")
                        with html.Tbody():
                            for row in section.rows:
                                with html.Tr():
                                    html.Td(row.name)
                                    # Direction matrices arrive as three lines
                                    html.Td(row.value, style="white-space: pre-line;")
