"""Test blend_transfer_functions module."""

import pytest as pt
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

from cardio.blend_transfer_functions import blend_transfer_functions


def test_blend_single_transfer_function():
    """Test that blending a single transfer function returns it unchanged."""
    # Create a simple transfer function pair
    otf = vtkPiecewiseFunction()
    otf.AddPoint(-1000, 0.0)
    otf.AddPoint(0, 0.8)
    otf.AddPoint(1000, 0.0)

    ctf = vtkColorTransferFunction()
    ctf.AddRGBPoint(-1000, 1.0, 0.0, 0.0)
    ctf.AddRGBPoint(0, 0.0, 1.0, 0.0)
    ctf.AddRGBPoint(1000, 0.0, 0.0, 1.0)

    tfs = [(otf, ctf)]

    # Blend should return the same transfer functions
    result_otf, result_ctf = blend_transfer_functions(tfs)

    assert result_otf is otf
    assert result_ctf is ctf


def test_blend_multiple_transfer_functions():
    """Test blending multiple transfer functions."""
    # Create first transfer function pair
    otf1 = vtkPiecewiseFunction()
    otf1.AddPoint(-1000, 0.0)
    otf1.AddPoint(0, 0.5)
    otf1.AddPoint(1000, 0.0)

    ctf1 = vtkColorTransferFunction()
    ctf1.AddRGBPoint(-1000, 1.0, 0.0, 0.0)
    ctf1.AddRGBPoint(0, 1.0, 0.0, 0.0)
    ctf1.AddRGBPoint(1000, 1.0, 0.0, 0.0)

    # Create second transfer function pair
    otf2 = vtkPiecewiseFunction()
    otf2.AddPoint(-1000, 0.0)
    otf2.AddPoint(0, 0.3)
    otf2.AddPoint(1000, 0.0)

    ctf2 = vtkColorTransferFunction()
    ctf2.AddRGBPoint(-1000, 0.0, 0.0, 1.0)
    ctf2.AddRGBPoint(0, 0.0, 0.0, 1.0)
    ctf2.AddRGBPoint(1000, 0.0, 0.0, 1.0)

    tfs = [(otf1, ctf1), (otf2, ctf2)]

    # Blend the transfer functions
    result_otf, result_ctf = blend_transfer_functions(tfs)

    # Verify result types
    assert isinstance(result_otf, vtkPiecewiseFunction)
    assert isinstance(result_ctf, vtkColorTransferFunction)

    # Test that opacity values are combined (should be sum of both)
    combined_opacity = result_otf.GetValue(0)
    assert combined_opacity == pt.approx(0.8, abs=0.1)  # 0.5 + 0.3

    # Test that the blended function has reasonable color values
    color = [0.0, 0.0, 0.0]
    result_ctf.GetColor(0, color)
    # Should be a purple-ish color (blend of red and blue)
    assert 0.0 <= color[0] <= 1.0
    assert 0.0 <= color[1] <= 1.0
    assert 0.0 <= color[2] <= 1.0


def test_blend_transfer_functions_custom_range():
    """Test blending with custom scalar range."""
    # Create transfer function pair
    otf = vtkPiecewiseFunction()
    otf.AddPoint(0, 0.0)
    otf.AddPoint(50, 0.8)
    otf.AddPoint(100, 0.0)

    ctf = vtkColorTransferFunction()
    ctf.AddRGBPoint(0, 1.0, 0.0, 0.0)
    ctf.AddRGBPoint(50, 0.0, 1.0, 0.0)
    ctf.AddRGBPoint(100, 0.0, 0.0, 1.0)

    tfs = [(otf, ctf)]

    # Blend with custom range
    result_otf, result_ctf = blend_transfer_functions(
        tfs, scalar_range=(0, 100), num_samples=100
    )

    # Verify the result matches input in this case
    assert result_otf is otf
    assert result_ctf is ctf


def test_blend_transfer_functions_num_samples():
    """Test blending with different number of samples."""
    # Create two similar transfer function pairs
    otf1 = vtkPiecewiseFunction()
    otf1.AddPoint(-100, 0.2)
    otf1.AddPoint(0, 0.6)
    otf1.AddPoint(100, 0.2)

    ctf1 = vtkColorTransferFunction()
    ctf1.AddRGBPoint(-100, 0.5, 0.5, 0.5)
    ctf1.AddRGBPoint(0, 0.8, 0.8, 0.8)
    ctf1.AddRGBPoint(100, 0.5, 0.5, 0.5)

    otf2 = vtkPiecewiseFunction()
    otf2.AddPoint(-100, 0.1)
    otf2.AddPoint(0, 0.2)
    otf2.AddPoint(100, 0.1)

    ctf2 = vtkColorTransferFunction()
    ctf2.AddRGBPoint(-100, 0.2, 0.2, 0.8)
    ctf2.AddRGBPoint(0, 0.3, 0.3, 0.9)
    ctf2.AddRGBPoint(100, 0.2, 0.2, 0.8)

    tfs = [(otf1, ctf1), (otf2, ctf2)]

    # Test with different sample counts
    for num_samples in [64, 256, 1024]:
        result_otf, result_ctf = blend_transfer_functions(
            tfs, scalar_range=(-100, 100), num_samples=num_samples
        )

        assert isinstance(result_otf, vtkPiecewiseFunction)
        assert isinstance(result_ctf, vtkColorTransferFunction)

        # Verify that the functions have the expected number of points
        assert result_otf.GetSize() == num_samples
        assert result_ctf.GetSize() == num_samples
