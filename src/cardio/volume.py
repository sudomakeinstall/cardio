import logging
import os

import itk
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
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
