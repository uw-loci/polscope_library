"""Geometric handling of slow-axis orientation.

Orientation is **axial**, not vectorial: theta and theta + pi describe the same
physical slow axis, and the value lives in ``[0, pi)``. Two consequences drive
this whole module.

**It cannot be resampled as an ordinary scalar.** The mean of 179 degrees and 1
degree is 90 degrees -- perpendicular to the truth, and entirely plausible
looking. Anything that averages, blends or downsamples orientation pixels must
go through the doubled-angle vector (sin 2t, cos 2t), which is continuous
across the wrap.

**It is expressed in the frame of the image.** Moving the pixels is not enough:
mirroring an image mirrors the physical directions it depicts, so the angle has
to move with them. See :func:`transform_orientation`.

Retardance carries neither constraint -- it is an ordinary non-negative scalar.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_orientation",
    "orientation_to_vector",
    "transform_orientation",
    "vector_to_orientation",
]


def orientation_to_vector(orientation_rad):
    """Doubled-angle representation ``(sin 2t, cos 2t)``, safe to interpolate.

    The doubling makes theta and theta + pi map to the same point, so the
    representation is continuous where the angle itself wraps.
    """
    o = np.asarray(orientation_rad, dtype=np.float64)
    return np.sin(2.0 * o), np.cos(2.0 * o)


def vector_to_orientation(sin2t, cos2t):
    """Inverse of :func:`orientation_to_vector`, returning ``[0, pi)``."""
    return (np.arctan2(np.asarray(sin2t, dtype=np.float64), np.asarray(cos2t, dtype=np.float64)) % (2.0 * np.pi)) / 2.0


def average_orientation(orientations, weights=None, axis=0):
    """Circular mean of axial orientations, optionally retardance-weighted.

    Weighting by retardance is usually what you want: orientation is undefined
    where there is no birefringence, so unweighted averaging lets pure noise
    pull the result.

    Parameters
    ----------
    orientations : array_like
        Orientations in radians.
    weights : array_like, optional
        Non-negative weights, broadcastable to ``orientations``. Retardance is
        the natural choice.
    axis : int
        Axis to reduce.
    """
    s, c = orientation_to_vector(orientations)
    if weights is None:
        return vector_to_orientation(np.mean(s, axis=axis), np.mean(c, axis=axis))
    w = np.asarray(weights, dtype=np.float64)
    total = np.sum(w, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.sum(s * w, axis=axis) / total
        cm = np.sum(c * w, axis=axis) / total
    return vector_to_orientation(sm, cm)


def transform_orientation(orientation_rad, *, flip_x=False, flip_y=False, rotate_deg=0.0):
    """Adjust orientation values to match a geometric transform of the pixels.

    Apply this **alongside** the pixel transform, never instead of it. Moving
    the pixels alone leaves the angles describing the old frame, which is the
    silent-failure case: the map still looks like a plausible orientation map.

    The mirror rule is worth stating because it surprises people: a **single**
    mirror negates the angle, but mirroring **both** axes does not change it at
    all, because that composition is a 180 degree rotation and orientation is
    axial (theta and theta + pi are the same axis).

    Parameters
    ----------
    orientation_rad : array_like
        Orientation in radians, any range; the result is wrapped to ``[0, pi)``.
    flip_x, flip_y : bool
        Whether the pixels were mirrored left-right / top-bottom.
    rotate_deg : float
        Rotation applied to the pixels, in degrees, in the same sense as the
        orientation convention (counter-clockwise on screen for the y-down
        image frame these values are measured in).

    Returns
    -------
    numpy.ndarray
        Transformed orientation in ``[0, pi)``.
    """
    o = np.asarray(orientation_rad, dtype=np.float64)
    if flip_x != flip_y:  # exactly one mirror
        o = -o
    return (o + np.deg2rad(rotate_deg)) % np.pi
