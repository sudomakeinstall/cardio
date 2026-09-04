"""The same physical field, stored two ways, must produce the same picture.

An obliquely acquired image and an axis-aligned one covering the same anatomy
carry different voxel grids and different direction matrices.  Resliced at the
same world pose they have to agree, because the pose is stated in world (LPS)
coordinates and the grid is an implementation detail of how the data was
sampled.  Without ``SetOutputDirection`` in ``build_pipeline`` the oblique cut
inherits the acquisition's rotation and this fails.
"""

# Third Party
import itk
import numpy as np
import pytest
import vtk.util.numpy_support as vtk_np

# Internal
from cardio.orientation import ensure_right_handed
from cardio.reslice import VIEW_TRANSFORMS, VIEWS, ResliceSet
from cardio.volume import Volume

CENTRE = np.array([4.0, -3.0, 12.0])
SIZE = np.array([48, 48, 48])
SPACING = np.array([1.0, 1.0, 1.0])

# Interpolating a smooth field onto a rotated grid and back costs a little
# accuracy; two griddings of the same field agree to about this much.
TOLERANCE = 0.06


def oblique_direction(yaw_degrees: float = 35.0, pitch_degrees: float = 20.0):
    yaw, pitch = np.radians(yaw_degrees), np.radians(pitch_degrees)
    about_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    about_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    return about_z @ about_x


def field(points: np.ndarray) -> np.ndarray:
    """An anisotropic blob in world space, so no two axes can be confused."""
    offset = points - CENTRE
    return 1000.0 * np.exp(
        -(
            (offset[..., 0] / 14.0) ** 2
            + (offset[..., 1] / 9.0) ** 2
            + (offset[..., 2] / 20.0) ** 2
        )
    )


def sampled_image(direction: np.ndarray):
    """``field`` sampled onto a grid with the given direction, centred on the blob."""
    origin = CENTRE - direction @ (SPACING * (SIZE - 1) / 2.0)
    kk, jj, ii = np.meshgrid(*[np.arange(s) for s in SIZE[::-1]], indexing="ij")
    index = np.stack([ii, jj, kk], axis=-1).astype(np.float64)
    world = origin + (index * SPACING) @ direction.T

    image = itk.image_from_array(np.ascontiguousarray(field(world).astype(np.float32)))
    image.SetOrigin(origin.tolist())
    image.SetSpacing(SPACING.tolist())
    image.SetDirection(itk.matrix_from_array(direction))
    return itk.vtk_image_from_image(image)


def cut(image_data, view: str, origin):
    """One view's reslice output, through the app's own pipeline."""
    reslice_set = ResliceSet(image_data, interpolation="linear", background_level=0.0)
    reslice_set.set_pose(list(origin))
    return reslice_set[view]["reslice"].GetOutput()


def sample_cut(output, offsets: np.ndarray) -> np.ndarray:
    """Bilinearly sample a cut at in-plane offsets from the pose origin.

    The two griddings crop to different extents and land on different sub-pixel
    phases, so the cuts can only be compared at stated positions rather than at
    matching array indices.
    """
    columns, rows, _ = output.GetDimensions()
    values = vtk_np.vtk_to_numpy(output.GetPointData().GetScalars()).reshape(
        rows, columns
    )

    grid_origin = np.array(output.GetOrigin())[:2]
    grid_spacing = np.array(output.GetSpacing())[:2]
    index = (offsets - grid_origin) / grid_spacing

    low = np.floor(index).astype(int)
    frac = index - low
    assert (low >= 0).all() and (low[..., 0] + 1 < columns).all(), "offset outside cut"
    assert (low[..., 1] + 1 < rows).all(), "offset outside cut"

    u, v = low[..., 0], low[..., 1]
    fu, fv = frac[..., 0], frac[..., 1]
    return (
        values[v, u] * (1 - fu) * (1 - fv)
        + values[v, u + 1] * fu * (1 - fv)
        + values[v + 1, u] * (1 - fu) * fv
        + values[v + 1, u + 1] * fu * fv
    )


def in_plane_offsets(half_width: float = 12.0, count: int = 25) -> np.ndarray:
    steps = np.linspace(-half_width, half_width, count)
    du, dv = np.meshgrid(steps, steps, indexing="xy")
    return np.stack([du, dv], axis=-1)


def world_points(view: str, offsets: np.ndarray, origin) -> np.ndarray:
    """Where a cut's in-plane offsets land in world (LPS) coordinates."""
    planar = np.concatenate([offsets, np.zeros_like(offsets[..., :1])], axis=-1)
    return origin + planar @ VIEW_TRANSFORMS[view].T


def test_oblique_and_axis_aligned_cuts_agree():
    aligned = sampled_image(np.eye(3))
    oblique = sampled_image(oblique_direction())
    offsets = in_plane_offsets()

    for view in VIEWS:
        from_aligned = sample_cut(cut(aligned, view, CENTRE), offsets)
        from_oblique = sample_cut(cut(oblique, view, CENTRE), offsets)

        difference = np.abs(from_aligned - from_oblique).max()
        assert difference < TOLERANCE * field(CENTRE[None])[0], (
            f"{view}: cuts differ by {difference:.2f}"
        )


def test_cut_matches_the_analytic_field():
    """Both griddings reproduce the field itself, not merely each other."""
    offsets = in_plane_offsets()

    for name, direction in (("aligned", np.eye(3)), ("oblique", oblique_direction())):
        image_data = sampled_image(direction)

        for view in VIEWS:
            sampled = sample_cut(cut(image_data, view, CENTRE), offsets)
            expected = field(world_points(view, offsets, CENTRE))

            difference = np.abs(sampled - expected).max()
            assert difference < TOLERANCE * field(CENTRE[None])[0], (
                f"{name} {view}: cut differs from the field by {difference:.2f}"
            )


# --- handedness ---------------------------------------------------------------

LEFT_HANDED = {
    "flipped x": np.diag([-1.0, 1.0, 1.0]),
    "flipped z": np.diag([1.0, 1.0, -1.0]),
    "swapped axes": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "oblique": oblique_direction() @ np.diag([1.0, 1.0, -1.0]),
}


def itk_image(direction, size=(5, 6, 7), spacing=(0.8, 1.5, 2.5)):
    array = np.arange(np.prod(size[::-1]), dtype=np.float32).reshape(size[::-1])
    image = itk.image_from_array(np.ascontiguousarray(array))
    image.SetOrigin([-11.0, 7.0, 3.0])
    image.SetSpacing(list(spacing))
    image.SetDirection(itk.matrix_from_array(np.asarray(direction, dtype=np.float64)))
    return image


@pytest.mark.parametrize("name", sorted(LEFT_HANDED))
def test_left_handed_directions_are_made_right_handed(name):
    image = itk_image(LEFT_HANDED[name])
    assert np.linalg.det(itk.array_from_matrix(image.GetDirection())) < 0

    corrected = ensure_right_handed(image)

    assert np.linalg.det(itk.array_from_matrix(corrected.GetDirection())) > 0


@pytest.mark.parametrize("name", sorted(LEFT_HANDED))
def test_every_voxel_keeps_its_world_position_and_value(name):
    """The correction renumbers voxels; it must not move or resample them."""
    image = itk_image(LEFT_HANDED[name])
    corrected = ensure_right_handed(image)

    size = list(corrected.GetLargestPossibleRegion().GetSize())
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                index = itk.Index[3]([i, j, k])
                point = corrected.TransformIndexToPhysicalPoint(index)
                source = image.TransformPhysicalPointToIndex(point)

                assert corrected.GetPixel(index) == image.GetPixel(source)
                np.testing.assert_allclose(
                    list(image.TransformIndexToPhysicalPoint(source)), list(point)
                )


def test_right_handed_images_pass_through_untouched():
    image = itk_image(np.eye(3))

    assert ensure_right_handed(image) is image


def test_left_handed_input_still_cuts_correctly():
    """The whole point: a left-handed image must reslice like any other."""
    direction = oblique_direction() @ np.diag([1.0, 1.0, -1.0])
    origin = CENTRE - direction @ (SPACING * (SIZE - 1) / 2.0)

    kk, jj, ii = np.meshgrid(*[np.arange(s) for s in SIZE[::-1]], indexing="ij")
    index = np.stack([ii, jj, kk], axis=-1).astype(np.float64)
    image = itk.image_from_array(
        np.ascontiguousarray(
            field(origin + (index * SPACING) @ direction.T).astype(np.float32)
        )
    )
    image.SetOrigin(origin.tolist())
    image.SetSpacing(SPACING.tolist())
    image.SetDirection(itk.matrix_from_array(direction))

    image_data = itk.vtk_image_from_image(ensure_right_handed(image))
    offsets = in_plane_offsets()

    for view in VIEWS:
        sampled = sample_cut(cut(image_data, view, CENTRE), offsets)
        expected = field(world_points(view, offsets, CENTRE))
        assert np.abs(sampled - expected).max() < TOLERANCE * field(CENTRE[None])[0]


def test_volume_actors_are_never_left_handed(tmp_path):
    """What the volume mapper is handed, since it draws nothing for left-handed input.

    No rendering here -- this is the property that failure reduces to, and it
    holds without a GL context.
    """
    image = itk_image(oblique_direction() @ np.diag([1.0, 1.0, -1.0]), size=(8, 8, 8))
    itk.imwrite(image, str(tmp_path / "0.nii.gz"))

    volume = Volume(label="v", directory=tmp_path)

    matrix = volume.mpr_image_data(0).GetDirectionMatrix()
    direction = np.array(
        [[matrix.GetElement(i, j) for j in range(3)] for i in range(3)]
    )
    assert np.linalg.det(direction) > 0
