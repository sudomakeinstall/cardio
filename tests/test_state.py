"""Test the per-object state key registry.

These keys are the contract between Logic (which registers and reads them) and
the UI (which binds them). A mismatch is a silent no-op at runtime, so the
spellings are pinned here.
"""

# Third Party
import pytest

# Internal
from cardio.mesh import Mesh
from cardio.object import Object
from cardio.segmentation import Segmentation
from cardio.state import ObjectState
from cardio.volume import Volume

MESH = ObjectState(kind="mesh", label="BL")
VOLUME = ObjectState(kind="volume", label="CCTA")
SEG = ObjectState(kind="segmentation", label="labels")


def test_kinds_are_declared_on_each_subclass():
    assert Object.kind == "object"
    assert Mesh.kind == "mesh"
    assert Volume.kind == "volume"
    assert Segmentation.kind == "segmentation"


def test_per_type_keys_carry_the_kind():
    assert MESH.visibility == "mesh_visibility_BL"
    assert VOLUME.visibility == "volume_visibility_CCTA"
    assert SEG.visibility == "segmentation_visibility_labels"

    assert MESH.clipping == "mesh_clipping_BL"
    assert VOLUME.clipping == "volume_clipping_CCTA"
    assert SEG.clipping == "segmentation_clipping_labels"


def test_clip_keys_are_keyed_by_label_alone():
    assert MESH.clip_panel == "clip_panel_BL"
    assert MESH.clip_bounds == ("clip_x_BL", "clip_y_BL", "clip_z_BL")


def test_clip_controls_covers_the_toggle_and_all_three_axes():
    assert MESH.clip_controls == [
        "mesh_clipping_BL",
        "clip_x_BL",
        "clip_y_BL",
        "clip_z_BL",
    ]


def test_volume_and_segmentation_specific_keys():
    assert VOLUME.preset == "volume_preset_CCTA"
    assert VOLUME.preset_panel == "preset_panel_CCTA"
    assert SEG.mpr_overlay == "mpr_segmentation_overlay_labels"


def test_two_kinds_sharing_a_label_do_not_collide():
    """A mesh and a volume may both be called "heart"."""
    mesh = ObjectState(kind="mesh", label="heart")
    volume = ObjectState(kind="volume", label="heart")

    assert mesh.visibility != volume.visibility
    assert mesh.clipping != volume.clipping


def test_of_reads_kind_and_label_off_an_object():
    class FakeObject:
        kind = "volume"
        label = "CCTA"

    assert ObjectState.of(FakeObject()) == VOLUME


@pytest.mark.parametrize(
    "key",
    ["visibility", "clipping", "clip_panel", "clip_x", "preset", "mpr_overlay"],
)
def test_every_key_is_a_valid_python_identifier(key):
    """trame state names are also attribute names, so they must not contain
    anything a label validator would have rejected."""
    assert getattr(VOLUME, key).isidentifier()
