"""Stokes-parameter polarimetry for the LC-PolScope, in numpy.

Portions of this module are a port of ``waveorder/stokes.py``:

    Copyright (c) 2025, Chan Zuckerberg Biohub
    Licensed under the BSD 3-Clause License. Full text in
    ``third_party_licenses/waveorder-LICENSE.txt``.

This module is a **port of the polarization subset** of
``waveorder/stokes.py`` (Chan Zuckerberg Biohub, BSD-3-Clause) from PyTorch to
numpy. See ``THIRD_PARTY_NOTICES.md`` for the upstream notice and the exact
commits the port was taken from.

Why port rather than depend
---------------------------
``waveorder`` requires Python >= 3.12 and ``torch >= 2.4.1``. Torch is there for
the *phase* reconstruction's FFT machinery, which we deliberately do not use --
this project reconstructs birefringence only. The polarization math is
elementary linear algebra, and every torch operation it uses has an exact numpy
equivalent:

    torch.linalg.pinv  -> np.linalg.pinv     torch.einsum    -> np.einsum
    torch.linalg.inv   -> np.linalg.inv      torch.moveaxis  -> np.moveaxis
    torch.remainder    -> np.remainder       torch.clone     -> np.copy

``np.linalg.inv`` broadcasts over leading axes exactly as the torch version
does, ``np.einsum`` takes the same subscripts, and ``np.remainder`` shares
torch's sign convention (result takes the sign of the divisor), which matters
for wrapping orientation into ``[0, pi)``. So the port is mechanical, and the
result runs on Python 3.10 with numpy alone.

The physics
-----------
An LC-PolScope measures N intensities, one per calibrated polarization state.
Those relate to the sample's Stokes parameters by a *system matrix* built from
the calibration swing and scheme::

    intensities = S2I @ stokes          (forward)
    stokes      = I2S @ intensities     (inverse, I2S = pinv(S2I))

The sample is then modelled as an attenuating depolarizing retarder (ADR), and
its four parameters -- retardance, slow-axis orientation, transmittance and
depolarization -- are recovered from the Stokes vector in one step.

**State order is part of the algorithm.** The system matrix rows correspond to
specific polarization states in a fixed order; feeding measurements in a
different order yields a rotated or mirrored orientation with no error raised.
See ``calculate_stokes_to_intensity_matrix`` for the orders.
"""

import numpy as np

__all__ = [
    "calculate_stokes_to_intensity_matrix",
    "calculate_intensity_to_stokes_matrix",
    "mmul",
    "stokes_after_adr",
    "estimate_adr_from_stokes",
    "mueller_from_stokes",
    "apply_orientation_offset",
    "radians_to_nanometers",
    "SCHEMES",
]

#: The only calibration schemes the system matrix is defined for.
SCHEMES = ("4-State", "5-State")


def calculate_stokes_to_intensity_matrix(swing: float, scheme: str = "5-State") -> np.ndarray:
    """Polarimeter system matrix for a given swing and calibration scheme.

    Parameters
    ----------
    swing : float
        Swing in waves (e.g. ``0.03``). Periodic on the integers, so
        ``swing=0.03`` and ``swing=1.03`` give the same matrix.
    scheme : {"4-State", "5-State"}
        The calibration scheme the data was acquired under.

    Returns
    -------
    numpy.ndarray
        Shape ``(5, 4)`` for ``"5-State"``, ``(4, 4)`` for ``"4-State"``.
        Rows are states, columns are ``(S0, S1, S2, S3)``.

    Notes
    -----
    **The row order is the state order.** For ``"5-State"`` the rows are
    extinction, +S1, +S2, -S1, -S2 -- that is, extinction plus four states at
    0/90/180/270 degrees in the S1-S2 plane. For ``"4-State"`` they are
    extinction plus three states 120 degrees apart. Measurements must be
    supplied in the same order.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"{scheme!r} is not implemented, use one of {SCHEMES}")

    chi = 2 * np.pi * swing
    if scheme == "5-State":
        return np.array(
            [
                [1, 0, 0, -1],  # extinction
                [1, np.sin(chi), 0, -np.cos(chi)],  # +S1
                [1, 0, np.sin(chi), -np.cos(chi)],  # +S2
                [1, -np.sin(chi), 0, -np.cos(chi)],  # -S1
                [1, 0, -np.sin(chi), -np.cos(chi)],  # -S2
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            [1, 0, 0, -1],
            [1, np.sin(chi), 0, -np.cos(chi)],
            [
                1,
                -0.5 * np.sin(chi),
                np.sqrt(3) * np.cos(chi / 2) * np.sin(chi / 2),
                -np.cos(chi),
            ],
            [
                1,
                -0.5 * np.sin(chi),
                -np.sqrt(3) * np.cos(chi / 2) * np.sin(chi / 2),
                -np.cos(chi),
            ],
        ],
        dtype=np.float64,
    )


def calculate_intensity_to_stokes_matrix(swing: float, scheme: str = "5-State") -> np.ndarray:
    """Inverse system matrix: measured intensities to Stokes parameters.

    The Moore-Penrose pseudo-inverse of
    :func:`calculate_stokes_to_intensity_matrix`, so the 5-state scheme is
    least-squares over its one redundant measurement.

    Returns
    -------
    numpy.ndarray
        Shape ``(4, 5)`` for ``"5-State"``, ``(4, 4)`` for ``"4-State"``.
    """
    return np.linalg.pinv(calculate_stokes_to_intensity_matrix(swing, scheme=scheme))


def mmul(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Matrix-multiply a per-pixel stack of vectors.

    Used both for applying an intensity-to-Stokes matrix to measured
    intensities and for applying a Mueller matrix to Stokes vectors.

    Parameters
    ----------
    matrix : numpy.ndarray
        Shape ``(N, M, ...)`` -- trailing axes may be per-pixel or absent.
    vector : numpy.ndarray
        Shape ``(M, ...)`` with trailing axes matching.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, ...)``.
    """
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError(
            f"matrix.shape[1]={matrix.shape[1]} does not match vector.shape[0]={vector.shape[0]}. "
            "A common cause is supplying the wrong number of polarization states for the scheme."
        )
    return np.einsum("NM...,M...->N...", matrix, vector)


def stokes_after_adr(retardance, orientation, transmittance, depolarization):
    """Forward model: Stokes parameters after circularly polarized light passes
    through an attenuating depolarizing retarder.

    The inverse of :func:`estimate_adr_from_stokes`. Mainly useful for tests and
    for generating synthetic data with a known answer.

    Parameters
    ----------
    retardance, orientation, transmittance, depolarization : array_like
        Identical shapes. Retardance and orientation in radians.

    Returns
    -------
    tuple of numpy.ndarray
        ``(s0, s1, s2, s3)``.
    """
    retardance = np.asarray(retardance, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    transmittance = np.asarray(transmittance, dtype=np.float64)
    depolarization = np.asarray(depolarization, dtype=np.float64)

    # Copy, so that a caller mutating s0 downstream does not also mutate the
    # transmittance array they passed in.
    s0 = np.array(transmittance, copy=True)
    s1 = transmittance * depolarization * np.sin(retardance) * np.sin(2 * orientation)
    s2 = -transmittance * depolarization * np.sin(retardance) * np.cos(2 * orientation)
    s3 = transmittance * depolarization * np.cos(retardance)
    return s0, s1, s2, s3


def _s12_to_orientation(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Slow-axis orientation from S1 and S2, wrapped into ``[0, pi)``.

    Orientation is *axial*: 0 and pi are the same physical orientation, because
    the measurement gives orientation but not direction. The factor of two is
    what makes that true.
    """
    return (np.arctan2(s1, -s2) % (2 * np.pi)) / 2


def estimate_adr_from_stokes(s0, s1, s2, s3):
    """Recover retardance, orientation, transmittance and depolarization.

    The inverse of :func:`stokes_after_adr`, and the step that turns Stokes
    parameters into the quantities we actually want.

    Returns
    -------
    tuple of numpy.ndarray
        ``(retardance, orientation, transmittance, depolarization)``.
        Retardance and orientation in radians; orientation in ``[0, pi)``.

    Notes
    -----
    **Retardance is recovered through an arcsine and is therefore limited to
    ``[0, pi/2]``** -- a quarter wave. Beyond that the value folds back rather
    than continuing to climb, so a strongly birefringent sample reads as less
    retarding than it is. This is a property of the upstream algorithm, not of
    the port. At 549 nm a quarter wave is about 137 nm.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    s3 = np.asarray(s3, dtype=np.float64)

    len_pol = np.sqrt(s1**2 + s2**2 + s3**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        retardance = np.arcsin(np.sqrt(s1**2 + s2**2) / len_pol)
        depolarization = len_pol / s0
    orientation = _s12_to_orientation(s1, s2)
    transmittance = np.array(s0, copy=True)
    return retardance, orientation, transmittance, depolarization


def mueller_from_stokes(s0, s1, s2, s3, direction: str = "forward", denom_floor: float = 0.0):
    """Mueller matrix of the attenuating depolarizing retarder that produced a
    given Stokes vector.

    Used for measured-background correction: the background's Stokes parameters
    describe the instrument's own residual birefringence, and the inverse of
    that Mueller matrix removes its contribution from the sample data.

    Parameters
    ----------
    s0, s1, s2, s3 : array_like
        Stokes parameters, identical shapes.
    direction : {"forward", "inverse"}
        ``"inverse"`` returns the matrix inverse, taken over the leading 4x4.
    denom_floor : float, optional
        Lower bound applied to ``s1**2 + s2**2`` before dividing. Default 0.0
        reproduces upstream behaviour exactly.

        **This is a deliberate, optional divergence from upstream.** The forward
        matrix divides by ``s1**2 + s2**2``, which goes to zero for perfectly
        circular light -- and a *background* field is close to circular almost
        by construction, since that is what a good calibration produces. The
        residual instrument birefringence we are trying to measure is exactly
        what keeps it non-zero, so the division is well-posed in principle but
        numerically delicate in practice, and a pixel that happens to sit at
        zero yields NaN that then spreads through the corrected image. Set a
        small floor (e.g. ``1e-12``) to bound it.

    Returns
    -------
    numpy.ndarray
        Shape ``(4, 4) + s0.shape``.
    """
    if direction not in ("forward", "inverse"):
        raise ValueError("direction must be 'forward' or 'inverse'")

    s0 = np.asarray(s0, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    s3 = np.asarray(s3, dtype=np.float64)

    if direction == "forward":
        denom = s1**2 + s2**2
        if denom_floor > 0:
            denom = np.maximum(denom, denom_floor)
        M = np.zeros((4, 4) + np.shape(s0), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            M[0, 0] = s0
            M[1, 1] = (s0 * s2**2 + s1**2 * s3) / denom
            M[1, 2] = s1 * s2 * (s3 - s0) / denom
            M[1, 3] = s1
            M[2, 1] = M[1, 2]
            M[2, 2] = (s0 * s1**2 + s2**2 * s3) / denom
            M[2, 3] = s2
            M[3, 1] = -M[1, 3]
            M[3, 2] = -M[2, 3]
            M[3, 3] = s3
        return M

    M = mueller_from_stokes(s0, s1, s2, s3, direction="forward", denom_floor=denom_floor)
    # np.linalg.inv inverts over the LAST two axes and broadcasts over the rest,
    # so move the 4x4 to the back, invert, and move it forward again.
    M_flip = np.moveaxis(M, (0, 1), (-2, -1))
    M_inv_flip = np.linalg.inv(M_flip)
    return np.moveaxis(M_inv_flip, (-2, -1), (0, 1))


def apply_orientation_offset(orientation, rotate: bool = False, flip: bool = False) -> np.ndarray:
    """Rotate and/or flip an orientation map, keeping it in ``[0, pi)``.

    With both flags false this still wraps the input into ``[0, pi)``, which is
    why it is applied unconditionally.

    Parameters
    ----------
    orientation : array_like
        Orientations in radians.
    rotate : bool
        Add 90 degrees.
    flip : bool
        Negate (mirror about the x axis).
    """
    out = np.array(orientation, dtype=np.float64, copy=True)
    if rotate:
        out += np.pi / 2
    if flip:
        out *= -1
    return np.remainder(out, np.pi)


def radians_to_nanometers(retardance_rad, wavelength_nm: float):
    """Convert retardance from radians to nanometres.

    Parameters
    ----------
    retardance_rad : array_like
        Retardance in radians, as returned by :func:`estimate_adr_from_stokes`.
    wavelength_nm : float
        Illumination wavelength in nanometres (549 on our scope). Retardance in
        physical units is meaningless without it, which is why it is required
        rather than defaulted.
    """
    return np.asarray(retardance_rad, dtype=np.float64) * wavelength_nm / (2 * np.pi)
