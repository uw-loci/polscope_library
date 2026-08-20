"""Birefringence reconstruction for the LC-PolScope.

Portions of this module are a port of upstream code:

    Copyright (c) 2025, Chan Zuckerberg Biohub  (waveorder)
    Copyright (c) 2020, Chan Zuckerberg Biohub  (recOrder)
    Licensed under the BSD 3-Clause License. Full texts in ``licenses/``.

The one function most callers need is :func:`reconstruct`: give it a stack of
polarization-state images and the calibration, get back retardance, orientation,
transmittance and depolarization.

This is a numpy port of the *birefringence* path only, taken from
``waveorder.models.inplane_oriented_thick_pol3d.apply_inverse_transfer_function``
and ``recOrder.cli.apply_inverse_models.birefringence`` (both Chan Zuckerberg
Biohub, BSD-3-Clause). See ``THIRD_PARTY_NOTICES.md``.

**No phase reconstruction.** Upstream sits alongside quantitative phase
imaging, which is what pulls in PyTorch and its FFT machinery. We deliberately
do not port it: this package reconstructs birefringence and nothing else, which
is why it needs only numpy and runs on Python 3.10.

Per-tile is equivalent to whole-mosaic
--------------------------------------
Every operation here is pointwise. The Stokes inversion is a matrix multiply
per pixel, and measured-background correction applies a per-pixel Mueller
matrix. Nothing is a neighbourhood operation, so reconstructing tile-by-tile
during acquisition gives bit-identical results to reconstructing an assembled
mosaic afterwards. That is what makes it safe to run this inside a tile loop.
"""

from typing import NamedTuple, Optional, Sequence

import numpy as np

from . import stokes as _stokes

__all__ = ["Birefringence", "reconstruct", "stokes_from_intensities"]


class Birefringence(NamedTuple):
    """Result of a birefringence reconstruction.

    Attributes
    ----------
    retardance_nm : numpy.ndarray
        Retardance in nanometres. Capped at a quarter wave by the algorithm --
        see :func:`polscope_library.stokes.estimate_adr_from_stokes`.
    orientation_rad : numpy.ndarray
        Slow-axis orientation in radians, in ``[0, pi)``.

        **Axial data -- do not average or interpolate it as an ordinary
        scalar.** 0 and pi are the same physical orientation, so the mean of
        179 degrees and 1 degree is 90 degrees: perpendicular to the truth, and
        entirely plausible-looking. Anything that averages pixels later --
        pyramid downsampling, tile blending, binning -- must operate on
        :attr:`orientation_sin2`/:attr:`orientation_cos2` instead, or on a hue
        encoding, and recover the angle afterwards.
    transmittance : numpy.ndarray
        Unitless.
    depolarization : numpy.ndarray
        Unitless, 0 to 1.
    """

    retardance_nm: np.ndarray
    orientation_rad: np.ndarray
    transmittance: np.ndarray
    depolarization: np.ndarray

    @property
    def orientation_sin2(self) -> np.ndarray:
        """``sin(2 * orientation)`` -- the averaging-safe encoding.

        Averaging this together with :attr:`orientation_cos2` is a vector
        average, which is the correct circular mean for axial data. The
        magnitude of the resultant is a free coherence measure: low magnitude
        means the orientations being combined disagreed.
        """
        return np.sin(2 * self.orientation_rad)

    @property
    def orientation_cos2(self) -> np.ndarray:
        """``cos(2 * orientation)``. See :attr:`orientation_sin2`."""
        return np.cos(2 * self.orientation_rad)


def _as_state_stack(intensities: Sequence[np.ndarray] | np.ndarray, what: str) -> np.ndarray:
    """Coerce a per-state sequence or an array into a ``(states, ...)`` array."""
    arr = np.asarray(intensities, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"{what} must have a leading state axis plus at least one spatial axis; got shape {arr.shape}")
    return arr


def stokes_from_intensities(
    intensities: Sequence[np.ndarray] | np.ndarray,
    swing: float,
    scheme: str = "5-State",
) -> np.ndarray:
    """Convert measured polarization-state intensities to Stokes parameters.

    Parameters
    ----------
    intensities : sequence of arrays, or array with leading state axis
        **In calibration order.** For ``"5-State"`` that is extinction, +S1,
        +S2, -S1, -S2. Order is positional: supplying a permutation yields a
        rotated or mirrored orientation with no error raised.
    swing : float
        Swing in waves, from the calibration.
    scheme : {"4-State", "5-State"}

    Returns
    -------
    numpy.ndarray
        Shape ``(4, ...)`` -- S0, S1, S2, S3.
    """
    data = _as_state_stack(intensities, "intensities")
    i2s = _stokes.calculate_intensity_to_stokes_matrix(swing, scheme=scheme)
    if data.shape[0] != i2s.shape[1]:
        raise ValueError(
            f"scheme {scheme!r} expects {i2s.shape[1]} states but {data.shape[0]} were supplied. "
            "Check that the state stack matches the calibration scheme."
        )
    return _stokes.mmul(i2s, data)


def reconstruct(
    intensities: Sequence[np.ndarray] | np.ndarray,
    swing: float,
    wavelength_nm: float,
    scheme: str = "5-State",
    background_intensities: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    rotate_orientation: bool = False,
    flip_orientation: bool = False,
    denom_floor: float = 0.0,
) -> Birefringence:
    """Reconstruct birefringence from polarization-state images.

    Parameters
    ----------
    intensities : sequence of arrays, or array with leading state axis
        Sample measurements, in calibration order. All states must have been
        acquired at the **same exposure and gain** -- the inversion treats them
        as samples of one radiometric scale, so a per-state difference biases
        retardance and orientation with no visible symptom.
    swing : float
        Swing in waves, from the calibration (e.g. ``0.03``).
    wavelength_nm : float
        Illumination wavelength in nanometres (e.g. ``549``).
    scheme : {"4-State", "5-State"}
        Must match the calibration that produced the states.
    background_intensities : optional
        A specimen-free stack of the same states, acquired the same way. If
        given, the instrument's own residual birefringence is removed by
        applying the inverse of the Mueller matrix that background implies.

        Capture it **specimen-free and slightly defocused (10-20 um)**: dust on
        the slide or coverslip that lands in the reference reappears as
        artificial negative spots in every corrected image.
    rotate_orientation, flip_orientation : bool
        Add 90 degrees to, or mirror, the orientation map. Intended for
        matching a target of known orientation, not for casual use.
    denom_floor : float
        Passed to :func:`polscope_library.stokes.mueller_from_stokes` for the
        background correction; see its note on near-circular backgrounds.

    Returns
    -------
    Birefringence
    """
    if scheme not in _stokes.SCHEMES:
        raise ValueError(f"{scheme!r} is not implemented, use one of {_stokes.SCHEMES}")

    data_stokes = stokes_from_intensities(intensities, swing, scheme=scheme)

    if background_intensities is None:
        corrected = data_stokes
    else:
        bg = _as_state_stack(background_intensities, "background_intensities")
        if bg.shape[0] != data_stokes.shape[0] and bg.shape[0] != np.shape(intensities)[0]:
            raise ValueError("background_intensities must have the same number of states as intensities")
        bg_stokes = stokes_from_intensities(bg, swing, scheme=scheme)
        inverse_bg_mueller = _stokes.mueller_from_stokes(*bg_stokes, direction="inverse", denom_floor=denom_floor)
        corrected = _stokes.mmul(inverse_bg_mueller, data_stokes)

    retardance_rad, orientation, transmittance, depolarization = _stokes.estimate_adr_from_stokes(*corrected)
    orientation = _stokes.apply_orientation_offset(orientation, rotate=rotate_orientation, flip=flip_orientation)

    return Birefringence(
        retardance_nm=_stokes.radians_to_nanometers(retardance_rad, wavelength_nm),
        orientation_rad=orientation,
        transmittance=transmittance,
        depolarization=depolarization,
    )
