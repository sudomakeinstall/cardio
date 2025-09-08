#!/usr/bin/env python

import itk
import numpy as np
import vtk
from trame.app import get_server
from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vuetify, vtk as vtk_widgets

import cardio

window = 600
level = 200
background = [0.0, 0.0, 0.0]
vr_preset = "vascular_closed"

server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

nifti_file_path = "ct_scan.nii.gz"

image = itk.imread(nifti_file_path)
image = cardio.utils.reset_direction(image)
image = itk.vtk_image_from_image(image)

axial_reslice = vtk.vtkImageReslice()
axial_reslice.SetInputData(image)
axial_reslice.SetOutputDimensionality(2)
axial_reslice.SetInterpolationModeToLinear()

sagittal_reslice = vtk.vtkImageReslice()
sagittal_reslice.SetInputData(image)
sagittal_reslice.SetOutputDimensionality(2)
sagittal_reslice.SetInterpolationModeToLinear()

coronal_reslice = vtk.vtkImageReslice()
coronal_reslice.SetInputData(image)
coronal_reslice.SetOutputDimensionality(2)
coronal_reslice.SetInterpolationModeToLinear()

axial_actor = vtk.vtkImageActor()
axial_actor.GetMapper().SetInputConnection(axial_reslice.GetOutputPort())
axial_property = axial_actor.GetProperty()
axial_property.SetColorWindow(window)
axial_property.SetColorLevel(level)

axial_renderer = vtk.vtkRenderer()
axial_render_window = vtk.vtkRenderWindow()
axial_render_window.AddRenderer(axial_renderer)
axial_interactor = vtk.vtkRenderWindowInteractor()
axial_render_window.SetInteractor(axial_interactor)
axial_render_window.SetOffScreenRendering(True)
axial_renderer.AddActor(axial_actor)
axial_renderer.SetBackground(background)

sagittal_actor = vtk.vtkImageActor()
sagittal_actor.GetMapper().SetInputConnection(sagittal_reslice.GetOutputPort())
sagittal_property = sagittal_actor.GetProperty()
sagittal_property.SetColorWindow(window)
sagittal_property.SetColorLevel(level)

sagittal_renderer = vtk.vtkRenderer()
sagittal_render_window = vtk.vtkRenderWindow()
sagittal_render_window.AddRenderer(sagittal_renderer)
sagittal_interactor = vtk.vtkRenderWindowInteractor()
sagittal_render_window.SetInteractor(sagittal_interactor)
sagittal_render_window.SetOffScreenRendering(True)
sagittal_renderer.AddActor(sagittal_actor)
sagittal_renderer.SetBackground(background)  # Black background

coronal_actor = vtk.vtkImageActor()
coronal_actor.GetMapper().SetInputConnection(coronal_reslice.GetOutputPort())
coronal_property = coronal_actor.GetProperty()
coronal_property.SetColorWindow(window)
coronal_property.SetColorLevel(level)

coronal_renderer = vtk.vtkRenderer()
coronal_render_window = vtk.vtkRenderWindow()
coronal_render_window.AddRenderer(coronal_renderer)
coronal_interactor = vtk.vtkRenderWindowInteractor()
coronal_render_window.SetInteractor(coronal_interactor)
coronal_render_window.SetOffScreenRendering(True)
coronal_renderer.AddActor(coronal_actor)
coronal_renderer.SetBackground(background)  # Black background

volume_mapper = vtk.vtkSmartVolumeMapper()
volume_mapper.SetInputData(image)

volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume_prop = cardio.volume_property_presets.load_volume_property_preset(vr_preset)
volume.SetProperty(volume_prop.vtk_property)

volume_renderer = vtk.vtkRenderer()
volume_render_window = vtk.vtkRenderWindow()
volume_render_window.AddRenderer(volume_renderer)
volume_interactor = vtk.vtkRenderWindowInteractor()
volume_render_window.SetInteractor(volume_interactor)
volume_render_window.SetOffScreenRendering(False)  # Enable interaction
volume_renderer.AddVolume(volume)
volume_renderer.SetBackground(background)

volume_renderer.ResetCamera()
volume_renderer.GetActiveCamera().Elevation(-90)

# Initialize the volume interactor
volume_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

center = image.GetCenter()
state.axial_slice = center[2]  # Z position for axial view
state.sagittal_slice = center[0]  # X position for sagittal view
state.coronal_slice = -center[1]  # Y position for coronal view (LPS->LAS: flip Y)

bounds = image.GetBounds()
x_min, x_max = bounds[0], bounds[1]
y_min, y_max = bounds[2], bounds[3]
z_min, z_max = bounds[4], bounds[5]

state.axial_min = z_min
state.axial_max = z_max
state.sagittal_min = x_min
state.sagittal_max = x_max
# LPS->LAS: flip Y range for coronal (Posterior->Anterior)
state.coronal_min = -y_max  # Most anterior (was most posterior)
state.coronal_max = -y_min  # Most posterior (was most anterior)

state.image_window = window
state.image_level = level

# Dynamic rotation sequence - list of rotation steps
state.rotation_sequence = []

# Initialize rotation labels and angle states
for i in range(20):
    setattr(state, f"rotation_axis_{i}", f"Rotation {i+1}")
    setattr(state, f"rotation_angle_{i}", 0)

initial_values = {
    "axial_slice": center[2],
    "sagittal_slice": center[0],
    "coronal_slice": -center[1],  # LPS->LAS: flip Y
    "image_window": window,
    "image_level": level,
}


@ctrl.set("reset_all")
def reset_to_initial():
    """Reset all sliders to their initial values."""
    state.axial_slice = initial_values["axial_slice"]
    state.sagittal_slice = initial_values["sagittal_slice"]
    state.coronal_slice = initial_values["coronal_slice"]
    state.image_window = initial_values["image_window"]
    state.image_level = initial_values["image_level"]
    reset_rotations()  # Reuse the existing reset_rotations function


def update_rotation_labels():
    """Update the rotation axis labels for display."""
    for i, rotation in enumerate(state.rotation_sequence):
        setattr(state, f"rotation_axis_{i}", f"{rotation['axis']} ({i+1})")
    # Clear unused labels
    for i in range(len(state.rotation_sequence), 20):
        setattr(state, f"rotation_axis_{i}", f"Rotation {i+1}")


@ctrl.set("add_x_rotation")
def add_x_rotation():
    """Add a new X rotation slider."""
    import copy

    current_sequence = copy.deepcopy(
        state.rotation_sequence
    )  # Deep copy to preserve angles
    current_sequence.append({"axis": "X", "angle": 0})
    state.rotation_sequence = current_sequence
    update_rotation_labels()


@ctrl.set("add_y_rotation")
def add_y_rotation():
    """Add a new Y rotation slider."""
    import copy

    current_sequence = copy.deepcopy(
        state.rotation_sequence
    )  # Deep copy to preserve angles
    current_sequence.append({"axis": "Y", "angle": 0})
    state.rotation_sequence = current_sequence
    update_rotation_labels()


@ctrl.set("add_z_rotation")
def add_z_rotation():
    """Add a new Z rotation slider."""
    import copy

    current_sequence = copy.deepcopy(
        state.rotation_sequence
    )  # Deep copy to preserve angles
    current_sequence.append({"axis": "Z", "angle": 0})
    state.rotation_sequence = current_sequence
    update_rotation_labels()


@ctrl.set("remove_rotation_event")
def remove_rotation_event(index):
    """Remove a rotation at given index."""
    sequence = list(state.rotation_sequence)
    if 0 <= index < len(sequence):
        sequence.pop(index)
        state.rotation_sequence = sequence
        update_rotation_labels()


@ctrl.set("reset_rotations")
def reset_rotations():
    """Reset all rotations."""
    state.rotation_sequence = []
    # Reset all individual angle states
    for i in range(20):
        setattr(state, f"rotation_angle_{i}", 0)
    update_rotation_labels()


@state.change(
    "axial_slice",
    "sagittal_slice",
    "coronal_slice",
    "rotation_sequence",
    "rotation_angle_0",
    "rotation_angle_1",
    "rotation_angle_2",
    "rotation_angle_3",
    "rotation_angle_4",
    "rotation_angle_5",
    "rotation_angle_6",
    "rotation_angle_7",
    "rotation_angle_8",
    "rotation_angle_9",
    "rotation_angle_10",
    "rotation_angle_11",
    "rotation_angle_12",
    "rotation_angle_13",
    "rotation_angle_14",
    "rotation_angle_15",
    "rotation_angle_16",
    "rotation_angle_17",
    "rotation_angle_18",
    "rotation_angle_19",
    "image_window",
    "image_level",
)
def update_all_views(**kwargs):
    # Calculate cumulative rotation matrix from sequence
    def create_rotation_matrix(axis, angle_degrees):
        """Create rotation matrix for given axis and angle."""
        angle = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        if axis == "X":
            return np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        elif axis == "Y":
            return np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        elif axis == "Z":
            return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        return np.eye(3)

    # Build cumulative rotation matrix using individual angle states
    cumulative_rotation = np.eye(3)
    for i, rotation in enumerate(state.rotation_sequence):
        angle = getattr(state, f"rotation_angle_{i}", 0)
        rotation_matrix = create_rotation_matrix(rotation["axis"], angle)
        cumulative_rotation = cumulative_rotation @ rotation_matrix

    def create_reslice_matrix(normal, up, origin):
        """Create a 4x4 reslice matrix from normal vector, up vector, and origin"""
        # Normalize vectors
        normal = normal / np.linalg.norm(normal)
        up = up / np.linalg.norm(up)
        # Right-handed coordinate system: right = normal × up
        right = np.cross(normal, up)
        right = right / np.linalg.norm(right)
        # Recompute up to ensure orthogonality: up = right × normal
        up = np.cross(right, normal)

        # Create 4x4 matrix - VTK uses column vectors
        matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            matrix.SetElement(i, 0, right[i])  # X direction (right)
            matrix.SetElement(i, 1, up[i])  # Y direction (up)
            matrix.SetElement(i, 2, normal[i])  # Z direction (normal)
            matrix.SetElement(i, 3, origin[i])  # Origin
        matrix.SetElement(3, 3, 1.0)
        return matrix

    # Apply cumulative rotation to base LAS view vectors
    # Base LAS vectors (before rotation)
    base_axial_normal = np.array([0.0, 0.0, 1.0])  # Z axis (Superior)
    base_axial_up = np.array([0.0, -1.0, 0.0])  # -Y axis (Anterior)
    base_sagittal_normal = np.array([1.0, 0.0, 0.0])  # X axis (Left)
    base_sagittal_up = np.array([0.0, 0.0, 1.0])  # Z axis (Superior)
    base_coronal_normal = np.array([0.0, 1.0, 0.0])  # Y axis (Posterior in data)
    base_coronal_up = np.array([0.0, 0.0, 1.0])  # Z axis (Superior)

    # Apply rotation to get current view vectors
    axial_normal = cumulative_rotation @ base_axial_normal
    axial_up = cumulative_rotation @ base_axial_up
    sagittal_normal = cumulative_rotation @ base_sagittal_normal
    sagittal_up = cumulative_rotation @ base_sagittal_up
    coronal_normal = cumulative_rotation @ base_coronal_normal
    coronal_up = cumulative_rotation @ base_coronal_up

    # Create reslice matrices with rotated vectors
    axial_origin = [center[0], center[1], state.axial_slice]
    axial_matrix = create_reslice_matrix(axial_normal, axial_up, axial_origin)
    axial_reslice.SetResliceAxes(axial_matrix)

    sagittal_origin = [state.sagittal_slice, center[1], center[2]]
    sagittal_matrix = create_reslice_matrix(
        sagittal_normal, sagittal_up, sagittal_origin
    )
    sagittal_reslice.SetResliceAxes(sagittal_matrix)

    # Coronal view: LPS->LAS Y coordinate conversion
    coronal_y_data = -state.coronal_slice  # Convert LAS Y to LPS Y for data access
    coronal_origin = [center[0], coronal_y_data, center[2]]
    coronal_matrix = create_reslice_matrix(coronal_normal, coronal_up, coronal_origin)
    coronal_reslice.SetResliceAxes(coronal_matrix)

    # Update display properties
    axial_property.SetColorWindow(state.image_window)
    axial_property.SetColorLevel(state.image_level)
    sagittal_property.SetColorWindow(state.image_window)
    sagittal_property.SetColorLevel(state.image_level)
    coronal_property.SetColorWindow(state.image_window)
    coronal_property.SetColorLevel(state.image_level)

    # Viewport backgrounds always black
    axial_renderer.SetBackground(background)
    sagittal_renderer.SetBackground(background)
    coronal_renderer.SetBackground(background)
    volume_renderer.SetBackground(background)

    # Reset cameras and update all views
    axial_renderer.ResetCamera()
    sagittal_renderer.ResetCamera()
    coronal_renderer.ResetCamera()

    ctrl.axial_update()
    ctrl.sagittal_update()
    ctrl.coronal_update()
    ctrl.volume_update()


with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("NIFTI Slice Viewer")

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VSwitch(
            v_model=("$vuetify.theme.dark", True),
            label="Dark Mode",
            dense=True,
            hide_details=True,
            style="max-width: 150px;",
        )

    with layout.drawer:
        with vuetify.VContainer(classes="pa-4"):
            # Reset button
            vuetify.VBtn(
                "Reset All",
                color="primary",
                small=True,
                click=ctrl.reset_all,
                style="width: 100%; margin-bottom: 16px;",
            )

            vuetify.VDivider(classes="my-2")
            vuetify.VSubheader("Slice Positions")
            vuetify.VSlider(
                v_model=("axial_slice", state.axial_slice),
                min=f"{state.axial_min}",
                max=f"{state.axial_max}",
                label="Axial (I ← → S)",
                dense=True,
                hide_details=False,
                thumb_label=True,
            )
            vuetify.VSlider(
                v_model=("sagittal_slice", state.sagittal_slice),
                min=f"{state.sagittal_min}",
                max=f"{state.sagittal_max}",
                label="Sagittal (R ← → L)",
                dense=True,
                hide_details=False,
                thumb_label=True,
            )
            vuetify.VSlider(
                v_model=("coronal_slice", state.coronal_slice),
                min=f"{state.coronal_min}",
                max=f"{state.coronal_max}",
                label="Coronal (P ← → A)",
                dense=True,
                hide_details=False,
                thumb_label=True,
            )

            vuetify.VDivider(classes="my-4")
            vuetify.VSubheader("View Rotation")

            # Add rotation buttons
            with vuetify.VRow(no_gutters=True, classes="mb-2"):
                with vuetify.VCol(cols=4):
                    vuetify.VBtn(
                        "Add X", small=True, click=ctrl.add_x_rotation, block=True
                    )
                with vuetify.VCol(cols=4):
                    vuetify.VBtn(
                        "Add Y", small=True, click=ctrl.add_y_rotation, block=True
                    )
                with vuetify.VCol(cols=4):
                    vuetify.VBtn(
                        "Add Z", small=True, click=ctrl.add_z_rotation, block=True
                    )

            vuetify.VBtn(
                "Reset Rotations",
                small=True,
                click=ctrl.reset_rotations,
                color="warning",
                block=True,
                classes="mb-2",
            )

            # Dynamic rotation sliders - increased limit
            with vuetify.VContainer(classes="pa-0"):
                for i in range(20):  # Support up to 20 rotations
                    with vuetify.VRow(
                        no_gutters=True,
                        classes="align-center mb-1",
                        v_if=f"rotation_sequence && rotation_sequence.length > {i}",
                    ):
                        with vuetify.VCol(cols=10):
                            vuetify.VSlider(
                                v_model=(f"rotation_angle_{i}", 0),
                                min=-180,
                                max=180,
                                label=(f"rotation_axis_{i}", f"Rotation {i+1}"),
                                dense=True,
                                hide_details=True,
                                thumb_label=True,
                            )
                        with vuetify.VCol(cols=2):
                            vuetify.VBtn(
                                "×",
                                small=True,
                                icon=True,
                                color="error",
                                click=(ctrl.remove_rotation_event, f"[{i}]"),
                            )

            vuetify.VDivider(classes="my-4")
            vuetify.VSubheader("Image Display")
            vuetify.VSlider(
                v_model=("image_window", 600),
                min=1,
                max=4000,
                label="Window",
                dense=True,
                hide_details=False,
                thumb_label=True,
            )
            vuetify.VSlider(
                v_model=("image_level", 200),
                min=-1000,
                max=2000,
                label="Level",
                dense=True,
                hide_details=False,
                thumb_label=True,
            )

    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
            # First row: Axial and Volume (50% height)
            with vuetify.VRow(no_gutters=True, style="height: 50%;"):
                with vuetify.VCol(cols=6, classes="pa-0", style="height: 100%;"):
                    axial_view = vtk_widgets.VtkRemoteView(
                        axial_render_window, style="height: 100%;"
                    )
                with vuetify.VCol(cols=6, classes="pa-0", style="height: 100%;"):
                    volume_view = vtk_widgets.VtkRemoteView(
                        volume_render_window,
                        style="height: 100%;",
                        interactive_ratio=1,  # Enable full interaction
                    )

            # Second row: Coronal and Sagittal (50% height)
            with vuetify.VRow(no_gutters=True, style="height: 50%;"):
                with vuetify.VCol(cols=6, classes="pa-0", style="height: 100%;"):
                    coronal_view = vtk_widgets.VtkRemoteView(
                        coronal_render_window, style="height: 100%;"
                    )
                with vuetify.VCol(cols=6, classes="pa-0", style="height: 100%;"):
                    sagittal_view = vtk_widgets.VtkRemoteView(
                        sagittal_render_window, style="height: 100%;"
                    )

            # Store view update functions in controller
            ctrl.axial_update = axial_view.update
            ctrl.sagittal_update = sagittal_view.update
            ctrl.coronal_update = coronal_view.update
            ctrl.volume_update = volume_view.update
            ctrl.axial_reset_camera = axial_view.reset_camera
            ctrl.sagittal_reset_camera = sagittal_view.reset_camera
            ctrl.coronal_reset_camera = coronal_view.reset_camera
            ctrl.volume_reset_camera = volume_view.reset_camera

# Initial call to set up all views
update_all_views()

# -----------------------------------------------------------------------------
# 5. Start server
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
