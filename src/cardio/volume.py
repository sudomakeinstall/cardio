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

        self.ambient: float = cfg["ambient"]
        self.diffuse: float = cfg["diffuse"]
        self.specular: float = cfg["specular"]

        # Clipping configuration
        self.clipping_enabled: bool = cfg["clipping_enabled"]
        self.clipping_planes: vtkPlanes = None

        frame = 0
        while os.path.exists(self.path_for_frame(frame)):
            logging.info(f"{self.label}: Loading frame {frame}.")
            mapper = vtkGPUVolumeRayCastMapper()
            actor = vtkVolume()
            image = itk.imread(self.path_for_frame(frame))
            image = reset_direction(image)
            image = itk.vtk_image_from_image(image)
            mapper.SetInputData(image)
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
        a = 0
        b = 500
        c = 1000
        d = 1150

        # Create transfer mapping scalar value to opacity.
        opacityTransferFunction = vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(a, 0.00)
        opacityTransferFunction.AddPoint(b, 0.15)
        opacityTransferFunction.AddPoint(c, 0.15)
        opacityTransferFunction.AddPoint(d, 0.85)

        # Create transfer mapping scalar value to color.
        colorTransferFunction = vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(a, 0.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(b, 1.0, 0.5, 0.3)
        colorTransferFunction.AddRGBPoint(c, 1.0, 0.5, 0.3)
        colorTransferFunction.AddRGBPoint(d, 1.0, 1.0, 0.9)

        volumeGradientOpacity = vtkPiecewiseFunction()
        volumeGradientOpacity.AddPoint(0, 0.0)
        volumeGradientOpacity.AddPoint(90, 0.5)
        volumeGradientOpacity.AddPoint(100, 1.0)

        # The property describes how the data will look.
        self.property = vtkVolumeProperty()
        self.property.SetColor(colorTransferFunction)
        self.property.SetScalarOpacity(opacityTransferFunction)
        self.property.SetGradientOpacity(volumeGradientOpacity)
        self.property.ShadeOn()
        self.property.SetInterpolationTypeToLinear()
        self.property.SetAmbient(self.ambient)
        self.property.SetDiffuse(self.diffuse)
        self.property.SetSpecular(self.specular)

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
