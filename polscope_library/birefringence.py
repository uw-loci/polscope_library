"""Birefringence reconstruction for the LC-PolScope.

Portions of this module are a port of upstream code:

    Copyright (c) 2025, Chan Zuckerberg Biohub  (waveorder)
    Copyright (c) 2020, Chan Zuckerberg Biohub  (recOrder)
    Licensed under the BSD 3-Clause License. Full texts in ``third_party_licenses/``.

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

from . import openpolscope as _ops
from . import stokes as _stokes

__all__ = ["RECONSTRUCTION_METHODS", "Birefringence", "reconstruct", "stokes_from_intensities"]

#: ``"qlipp"`` is the default: the waveorder/recOrder Stokes inversion with a
#: Mueller-matrix background correction. ``"openpolscope"`` reproduces
#: OpenPolScope 3.20b exactly; see :mod:`polscope_library.openpolscope` for
#: what the two share and where they part.
RECONSTRUCTION_METHODS = ("qlipp", "openpolscope")


class Birefringence(NamedTuple):
    """Result of a birefringence reconstruction.

    Attributes
    ----------
    retardance_nm : numpy.ndarray
        Retardance in nanometres. With ``method="qlipp"`` it is capped at a
        quarter wave by the algorithm -- see
        :func:`polscope_library.stokes.estimate_adr_from_stokes`. With
        ``method="openpolscope"`` it reaches a half wave, and is ``+inf`` on
        dead pixels (all swing states zero).
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
        Unitless. All-NaN when ``method="openpolscope"``, which does not
        compute it.
    depolarization : numpy.ndarray
        Unitless, 0 to 1. All-NaN when ``method="openpolscope"``.
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
    method: str = "qlipp",
    black_level: float = 0.0,
    azimuth_reference_deg: float = 0.0,
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
        Illumination wavelength in nanometres (e.g. ``546``).
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
        ``"qlipp"`` only.
    method : {"qlipp", "openpolscope"}
        ``"qlipp"`` (default) is the waveorder/recOrder Stokes inversion.
        ``"openpolscope"`` reproduces OpenPolScope 3.20b bit-for-bit, for
        matching data processed there; 5-State only. The two are identical
        without a background and differ in how one is applied -- see
        :mod:`polscope_library.openpolscope`.
    black_level : float
        Camera offset in counts, subtracted from every state (sample and
        background) and clamped at zero before reconstruction, by either
        method. Retardance and orientation are invariant to a uniform offset
        except through that clamp, so leaving this at 0 is safe for them; it
        does change ``depolarization`` under ``"qlipp"``.
    azimuth_reference_deg : float
        Added to every orientation. OpenPolScope's "Azimuth Reference";
        honoured by both methods.

    Returns
    -------
    Birefringence
    """
    if method not in RECONSTRUCTION_METHODS:
        raise ValueError(f"method must be one of {RECONSTRUCTION_METHODS}, got {method!r}")
    if black_level:
        intensities = np.maximum(0.0, _as_state_stack(intensities, "intensities") - float(black_level))
        if background_intensities is not None:
            background_intensities = np.maximum(
                0.0, _as_state_stack(background_intensities, "background_intensities") - float(black_level)
            )

    if method == "openpolscope":
        if scheme != "5-State":
            raise NotImplementedError("method='openpolscope' is 5-State only; use method='qlipp' for 4-State data")
        retardance_rad, orientation, _dead = _ops.reconstruct_openpolscope(
            intensities,
            swing,
            background_intensities=background_intensities,
            black_level=0.0,  # already applied above, identically for both methods
            azimuth_reference_deg=azimuth_reference_deg,
            mirror=flip_orientation,
        )
        # mirror is OpenPolScope's own flip and was applied inside; only the
        # rotation is left to the common helper.
        orientation = _stokes.apply_orientation_offset(orientation, rotate=rotate_orientation, flip=False)
        not_computed = np.full(np.shape(retardance_rad), np.nan)
        return Birefringence(
            retardance_nm=_stokes.radians_to_nanometers(retardance_rad, wavelength_nm),
            orientation_rad=orientation,
            transmittance=not_computed,
            depolarization=not_computed,
        )

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
    if azimuth_reference_deg:
        orientation = orientation + np.radians(float(azimuth_reference_deg))
    orientation = _stokes.apply_orientation_offset(orientation, rotate=rotate_orientation, flip=flip_orientation)

    return Birefringence(
        retardance_nm=_stokes.radians_to_nanometers(retardance_rad, wavelength_nm),
        orientation_rad=orientation,
        transmittance=transmittance,
        depolarization=depolarization,
    )
