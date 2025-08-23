# System
import numpy as np

# Third Party
import vtk


def blend_transfer_functions(tfs, scalar_range=(-2000, 2000), num_samples=512):
    """
    Blend multiple transfer functions using volume rendering emission-absorption model.

    Based on the volume rendering equation from:
    - Levoy, M. "Display of Surfaces from Volume Data" IEEE Computer Graphics and Applications, 1988
    - Kajiya, J.T. & Von Herzen, B.P. "Ray tracing volume densities" ACM SIGGRAPH Computer Graphics, 1984
    - Engel, K. et al. "Real-time Volume Graphics" A K Peters, 2006, Chapter 2

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
    blended_otf = vtk.vtkPiecewiseFunction()
    blended_ctf = vtk.vtkColorTransferFunction()

    for i, scalar_val in enumerate(sample_points):
        blended_otf.AddPoint(scalar_val, blended_opacity[i])
        blended_ctf.AddRGBPoint(
            scalar_val, blended_color[i][0], blended_color[i][1], blended_color[i][2]
        )

    return blended_otf, blended_ctf
