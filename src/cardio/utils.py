"""Utility functions shared across cardio classes."""

import enum


class InterpolatorType(enum.Enum):
    """Interpolation methods for image resampling."""

    LINEAR = "linear"
    NEAREST = "nearest"


class AngleUnit(enum.Enum):
    """Units for angle measurements."""

    DEGREES = "degrees"
    RADIANS = "radians"


def calculate_combined_bounds(actors):
    """Calculate combined bounds encompassing all VTK actors.

    Args:
        actors: List of VTK actors or list of lists/dicts of actors

    Returns:
        List of 6 floats: [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    if not actors:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Flatten actors if nested (for segmentations with frame_actors dict)
    flat_actors = []
    for item in actors:
        if isinstance(item, dict):
            flat_actors.extend(item.values())
        elif isinstance(item, list):
            flat_actors.extend(item)
        else:
            flat_actors.append(item)

    if not flat_actors:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Start with first actor's bounds
    combined = list(flat_actors[0].GetBounds())

    # Expand to encompass all actors
    for actor in flat_actors[1:]:
        bounds = actor.GetBounds()
        combined[0] = min(combined[0], bounds[0])  # xmin
        combined[1] = max(combined[1], bounds[1])  # xmax
        combined[2] = min(combined[2], bounds[2])  # ymin
        combined[3] = max(combined[3], bounds[3])  # ymax
        combined[4] = min(combined[4], bounds[4])  # zmin
        combined[5] = max(combined[5], bounds[5])  # zmax

    return combined
