import logging
import os

import itk
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction, vtkPlanes
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper

from . import Object
from .transfer_functions import load_preset




def blend_transfer_functions(tfs, scalar_range=(-2000, 2000), num_samples=512):
    """
    Blend multiple transfer functions using volume rendering emission-absorption model.

    The volume rendering integral: I = ∫ C(s) * μ(s) * T(s) ds
    where C(s) = emission color, μ(s) = opacity, T(s) = transmission

    For discrete transfer functions, this becomes:
    - Total emission = Σ(color_i * opacity_i)
    - Total absorption = Σ(opacity_i)
    - Final color = total_emission / total_absorption
    """
    if len(tfs) == 1:
        return tfs[0]

    sample_points = np.linspace(
        start=scalar_range[0],
        stop=scalar_range[1],
        num=num_samples,
    )

    # Initialize arrays to store blended values
    blended_opacity = []
    blended_color = []

    for scalar_val in sample_points:
        # Accumulate emission and absorption for volume rendering
        total_emission = [0.0, 0.0, 0.0]
        total_absorption = 0.0

        for otf, ctf in tfs:
            # Get opacity and color for this scalar value
            layer_opacity = otf.GetValue(scalar_val)
            layer_color = [0.0, 0.0, 0.0]
            ctf.GetColor(scalar_val, layer_color)

            # Volume rendering accumulation:
            # Emission = color * opacity (additive)
            # Absorption = opacity (multiplicative through transmission)
            for i in range(3):
                total_emission[i] += layer_color[i] * layer_opacity

            total_absorption += layer_opacity

        # Clamp values to reasonable ranges
        total_absorption = min(total_absorption, 1.0)
        for i in range(3):
            total_emission[i] = min(total_emission[i], 1.0)

        # For the final color, normalize emission by absorption if absorption > 0
        if total_absorption > 0.001:  # Avoid division by zero
            final_color = [total_emission[i] / total_absorption for i in range(3)]
        else:
            final_color = [0.0, 0.0, 0.0]

        # Clamp final colors
        final_color = [min(c, 1.0) for c in final_color]

        blended_opacity.append(total_absorption)
        blended_color.append(final_color)

    # Create new VTK transfer functions with blended values
    blended_otf = vtkPiecewiseFunction()
    blended_ctf = vtkColorTransferFunction()

    for i, scalar_val in enumerate(sample_points):
        blended_otf.AddPoint(scalar_val, blended_opacity[i])
        blended_ctf.AddRGBPoint(
            scalar_val, blended_color[i][0], blended_color[i][1], blended_color[i][2]
        )

    return blended_otf, blended_ctf


def reset_direction(image):
    origin = np.asarray(itk.origin(image))
    spacing = np.asarray(itk.spacing(image))
    size = np.asarray(itk.size(image))
    direction = np.asarray(image.GetDirection())

    direction[direction == 1] = 0
    origin += np.dot(size, np.dot(np.diag(spacing), direction))
    direction = np.identity(3)

    origin = itk.Point[itk.F, 3](origin)
    spacing = itk.spacing(image)
    size = itk.size(image)
    direction = itk.matrix_from_array(direction)

    interpolator = itk.LinearInterpolateImageFunction.New(image)
    output = itk.resample_image_filter(
        image,
        size=size,
        interpolator=interpolator,
        output_spacing=spacing,
        output_origin=origin,
        output_direction=direction,
    )

    return output


class Volume(Object):
    def __init__(self, cfg: str, renderer: vtkRenderer):
        super().__init__(cfg, renderer)
        self.actors: list[vtkVolume] = []

        # Load preset configuration
        self.preset = load_preset(cfg["transfer_function_preset"])

        # Clipping configuration
        self.clipping_enabled: bool = cfg["clipping_enabled"]
        self.clipping_planes: vtkPlanes = None

        frame = 0
        while os.path.exists(self.path_for_frame(frame)):
            logging.info(f"{self.label}: Loading frame {frame}.")

            image = itk.imread(self.path_for_frame(frame))
            image = reset_direction(image)
            image = itk.vtk_image_from_image(image)

            mapper = vtkGPUVolumeRayCastMapper()
            mapper.SetInputData(image)

            actor = vtkVolume()
            actor.SetMapper(mapper)

            self.actors += [actor]
            frame += 1

        self.setup_property()

    def setup_pipeline(self, frame: int):
        for a in self.actors:
            self.renderer.AddVolume(a)
            a.SetVisibility(False)
            a.SetProperty(self.property)
        if self.visible:
            self.actors[frame].SetVisibility(True)

        # Set up clipping after actors are configured
        self.setup_clipping()

        self.renderer.ResetCamera()

    def setup_property(self):
        tfs = [tf.vtk_functions for tf in self.preset.transfer_functions]

        # Blend all transfer functions into a single composite
        blended_otf, blended_ctf = blend_transfer_functions(tfs)

        self.property = vtkVolumeProperty()
        self.property.SetScalarOpacity(blended_otf)
        self.property.SetColor(blended_ctf)
        self.property.ShadeOn()
        self.property.SetInterpolationTypeToLinear()
        self.property.SetAmbient(self.preset.ambient)
        self.property.SetDiffuse(self.preset.diffuse)
        self.property.SetSpecular(self.preset.specular)

    def setup_clipping(self):
        """Set up volume clipping planes."""
        if not self.actors:
            return

        # Get volume bounds for initial clipping planes
        bounds = self.actors[0].GetBounds()

        # Create clipping planes
        self.clipping_planes = vtkPlanes()
        self._create_clipping_planes_from_bounds(bounds)

    def _create_clipping_planes_from_bounds(self, bounds):
        """Create 6 clipping planes from box bounds."""
        import vtkmodules.vtkCommonCore as vtk

        # Create 6 planes for the box faces
        planes = []
        normals = [
            [1, 0, 0],
            [-1, 0, 0],  # x-min, x-max
            [0, 1, 0],
            [0, -1, 0],  # y-min, y-max
            [0, 0, 1],
            [0, 0, -1],  # z-min, z-max
        ]
        origins = [
            [bounds[0], 0, 0],
            [bounds[1], 0, 0],  # x-min, x-max
            [0, bounds[2], 0],
            [0, bounds[3], 0],  # y-min, y-max
            [0, 0, bounds[4]],
            [0, 0, bounds[5]],  # z-min, z-max
        ]

        points = vtk.vtkPoints()
        norms = vtk.vtkDoubleArray()
        norms.SetNumberOfComponents(3)
        norms.SetName("Normals")

        for i, (normal, origin) in enumerate(zip(normals, origins)):
            points.InsertNextPoint(origin)
            norms.InsertNextTuple(normal)

        self.clipping_planes.SetPoints(points)
        self.clipping_planes.SetNormals(norms)

    def toggle_clipping(self, enabled: bool):
        """Enable or disable volume clipping."""
        self.clipping_enabled = enabled

        if enabled and self.clipping_planes:
            # Apply clipping to all actors
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.SetClippingPlanes(self.clipping_planes)
        else:
            # Remove clipping from all actors
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.RemoveAllClippingPlanes()

    def reset_clipping_bounds(self):
        """Reset the clipping bounds to the volume bounds."""
        if not self.actors:
            return

        bounds = self.actors[0].GetBounds()
        self.update_clipping_bounds(bounds)

    def update_clipping_bounds(self, bounds):
        """Update clipping bounds from UI controls."""
        if not self.clipping_planes:
            return

        # Update clipping planes with new bounds
        self._create_clipping_planes_from_bounds(bounds)

        # Apply to all volume actors
        if self.clipping_enabled:
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.SetClippingPlanes(self.clipping_planes)
