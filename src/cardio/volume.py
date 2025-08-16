import logging
import os

import itk
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction, vtkBox, vtkPlanes, vtkPlane
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper
from vtkmodules.vtkInteractionWidgets import (
    vtkBoxWidget2,
    vtkBoxRepresentation
)

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
        self.clipping_widget: vtkBoxWidget2 = None
        self.clipping_box: vtkBox = None
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
        self.setup_box_clipping()

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

    def setup_box_clipping(self):
        """Set up interactive box widget for volume clipping."""
        if not self.actors:
            return

        # Get volume bounds for initial box positioning
        bounds = self.actors[0].GetBounds()

        # Create clipping planes
        self.clipping_planes = vtkPlanes()
        self._create_clipping_planes_from_bounds(bounds)

        # Create the box widget
        self.clipping_widget = vtkBoxWidget2()
        box_rep = vtkBoxRepresentation()
        box_rep.SetPlaceFactor(1.0)
        box_rep.PlaceWidget(bounds)
        box_rep.SetInsideOut(True)  # Clip outside the box

        # Make the box widget more visible and interactive
        box_rep.HandlesOn()
        box_rep.SetHandleSize(0.01)

        self.clipping_widget.SetRepresentation(box_rep)

        # Set the interactor - it should be available from the renderer
        interactor = self.renderer.GetRenderWindow().GetInteractor()
        if interactor:
            self.clipping_widget.SetInteractor(interactor)

            # Add observer for interaction events
            self.clipping_widget.AddObserver("InteractionEvent", self.on_box_changed)

            if self.clipping_enabled:
                self.clipping_widget.On()
                # Enable widget processing
                self.clipping_widget.SetProcessEvents(1)

    def _create_clipping_planes_from_bounds(self, bounds):
        """Create 6 clipping planes from box bounds."""
        import vtkmodules.vtkCommonCore as vtk

        # Create 6 planes for the box faces
        planes = []
        normals = [
            [1, 0, 0], [-1, 0, 0],   # x-min, x-max
            [0, 1, 0], [0, -1, 0],   # y-min, y-max
            [0, 0, 1], [0, 0, -1]    # z-min, z-max
        ]
        origins = [
            [bounds[0], 0, 0], [bounds[1], 0, 0],  # x-min, x-max
            [0, bounds[2], 0], [0, bounds[3], 0],  # y-min, y-max
            [0, 0, bounds[4]], [0, 0, bounds[5]]   # z-min, z-max
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

    def on_box_changed(self, widget, event):
        """Callback for when the clipping box changes."""
        if not self.clipping_widget or not self.clipping_planes:
            return

        # Get the current box bounds from the widget
        box_rep = self.clipping_widget.GetRepresentation()
        bounds = [0] * 6
        box_rep.GetBounds(bounds)

        # Regenerate clipping planes from new bounds
        self._create_clipping_planes_from_bounds(bounds)

        # Apply clipping to all volume actors
        for actor in self.actors:
            mapper = actor.GetMapper()
            mapper.SetClippingPlanes(self.clipping_planes)

        # Request a render
        self.renderer.GetRenderWindow().Render()

    def toggle_clipping(self, enabled: bool):
        """Enable or disable box clipping."""
        self.clipping_enabled = enabled

        if not self.clipping_widget:
            return

        if enabled:
            self.clipping_widget.On()
            # Apply clipping to all actors
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.SetClippingPlanes(self.clipping_planes)
        else:
            self.clipping_widget.Off()
            # Remove clipping from all actors
            for actor in self.actors:
                mapper = actor.GetMapper()
                mapper.RemoveAllClippingPlanes()

        self.renderer.GetRenderWindow().Render()

    def reset_clipping_box(self):
        """Reset the clipping box to the volume bounds."""
        if not self.actors or not self.clipping_widget:
            return

        bounds = self.actors[0].GetBounds()
        box_rep = self.clipping_widget.GetRepresentation()
        box_rep.PlaceWidget(bounds)

        # Trigger the callback to update clipping
        self.on_box_changed(self.clipping_widget, "InteractionEvent")

    def enable_widget_interaction(self):
        """Enable widget interaction after the scene is fully initialized."""
        if self.clipping_widget and self.clipping_enabled:
            self.clipping_widget.On()
            self.clipping_widget.SetProcessEvents(1)

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
