"""OpenPolScope-exact birefringence reconstruction.

Selected with ``reconstruct(..., method="openpolscope")``. It reproduces the
output of OpenPolScope 3.20b (the Pol-Acquisition / Birefringence plugins from
the Marine Biological Laboratory, R. Oldenbourg) bit-for-bit on this
laboratory's reference data -- see ``tests/test_openpolscope.py``, which
carries a crop of that data and checks every pixel.

Why this exists alongside the QLIPP path
----------------------------------------
Without background correction the two methods are the same function: on the
reference data their retardance agrees to machine precision over 2.5 million
pixels. They differ in exactly two places:

1. **Background correction.** OpenPolScope subtracts the background in the
   normalised difference space it works in (``a/den - bg_a``). QLIPP inverts
   the background's Mueller matrix and applies it to the sample's Stokes
   vector. On a nearly clear reference slide that is a 0.53% scale difference;
   on strongly birefringent tissue nobody has measured it yet.
2. **Past a quarter wave.** OpenPolScope detects the fold (its ``den`` goes
   negative there) and reports ``pi - ret``, so retardance keeps climbing up
   to a half wave. QLIPP's arcsine cannot represent that region and folds back.

QLIPP is the better model where the two differ and stays the default. This
path is for matching data that was, or will be, processed in OpenPolScope.

Provenance
----------
The method is the published one: Shribak & Oldenbourg, "Techniques for fast
and sensitive measurements of two-dimensional birefringence distributions",
Applied Optics 42(16):3009-3017 (2003), https://doi.org/10.1364/AO.42.003009.
The behavioural details that the paper leaves open -- the black-level clamp,
the sign test for the fold-back, which difference pair feeds the azimuth when
a background is present, the azimuth offset convention -- were established by
reproducing OpenPolScope's own output on our data, not by copying its code.
No OpenPolScope source is included here; it is not distributed under an open
licence.

One thing is verified empirically rather than read: how OpenPolScope treats a
background pixel whose normalising denominator is zero. This module skips the
subtraction there (treats the background term as 0), which reproduces the
reference output exactly; that output may simply contain no such pixel.

State order
-----------
Takes states in the library's order, ``(extinction, +S1, +S2, -S1, -S2)``, the
same contract as every other entry point here. OpenPolScope's own acquisition
order differs (its pairs are states 1&4 and 2&3); the caller remaps, exactly as
it already must for the QLIPP path. On the QPSC rig that remap is the
``state_order`` key in the microscope YAML.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["reconstruct_openpolscope"]


def _differences(states: np.ndarray, black_level: float):
    """The normalised differences OpenPolScope works in.

    Returns ``(a_raw, b_raw, den, a, b, swing_sum)`` where ``a``/``b`` are the
    raw differences divided by ``den`` (0 where ``den`` is 0, which the caller
    overrides) and ``swing_sum`` is the sum of the four swing states, which
    OpenPolScope uses as a dead-pixel mask.
    """
    q = np.maximum(0.0, states - float(black_level))
    swing_sum = q[1] + q[2] + q[3] + q[4]
    den = (swing_sum - 4.0 * q[0]) / 2.0
    # In library order +S1 is q[1], -S1 is q[3], +S2 is q[2], -S2 is q[4].
    # OpenPolScope's "p4 - p1" and "p2 - p3" land on these once its own
    # acquisition order is remapped to ours.
    a_raw = q[3] - q[1]
    b_raw = q[2] - q[4]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(den != 0.0, a_raw / den, 0.0)
        b = np.where(den != 0.0, b_raw / den, 0.0)
    return a_raw, b_raw, den, a, b, swing_sum


def reconstruct_openpolscope(
    intensities: Sequence[np.ndarray] | np.ndarray,
    swing_waves: float,
    *,
    background_intensities: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    black_level: float = 0.0,
    azimuth_reference_deg: float = 0.0,
    mirror: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retardance and orientation the way OpenPolScope 3.20b computes them.

    Parameters
    ----------
    intensities
        Five states in library order ``(extinction, +S1, +S2, -S1, -S2)``.
    swing
        Swing in waves.
    background_intensities
        Same five states on a specimen-free field, or ``None``.
    black_level
        Camera offset in counts. Subtracted from every state and clamped at
        zero before anything else, as OpenPolScope does. Retardance and
        orientation are invariant to a uniform offset except through that
        clamp, so this mostly matters for pixels darker than the offset.
    azimuth_reference_deg
        OpenPolScope's "Azimuth Reference (degrees)": added to every
        orientation.
    mirror
        OpenPolScope's "mirror" setting: orientation becomes ``180 - theta``.

    Returns
    -------
    (retardance_rad, orientation_rad, dead)
        Retardance in radians on ``[0, pi]`` -- note the upper bound, a half
        wave, which is twice what the QLIPP path can express. Orientation in
        radians on ``[0, pi)``. ``dead`` is True where all four swing states
        are zero after black-level subtraction; OpenPolScope writes a saturated
        retardance there, and the retardance returned here is ``+inf`` so that
        a caller scaling to integers saturates the same way.
    """
    states = np.asarray(intensities, dtype=np.float64)
    if states.ndim < 2:
        raise ValueError(f"intensities must have a leading state axis plus spatial axes; got shape {states.shape}")
    if states.shape[0] != 5:
        raise NotImplementedError(
            f"method='openpolscope' implements OpenPolScope's 5-frame algorithm only; got {states.shape[0]} "
            "states. Its 4-frame path exists in the source but has not been verified against any real "
            "4-frame output, so it is deliberately not offered -- use method='qlipp' for 4-State data."
        )

    a_raw, b_raw, den, a, b, swing_sum = _differences(states, black_level)

    if background_intensities is None:
        # No background: the azimuth is taken from the raw differences. Same
        # angle modulo 180 as the normalised ones, so no subtlety here.
        azimuth_from_raw = np.ones(den.shape, dtype=bool)
    else:
        bg = np.asarray(background_intensities, dtype=np.float64)
        if bg.shape != states.shape:
            raise ValueError(f"background_intensities shape {bg.shape} must match intensities shape {states.shape}")
        _, _, _, bg_a, bg_b, _ = _differences(bg, black_level)
        a = a - bg_a
        b = b - bg_b
        # With a background present OpenPolScope still takes the azimuth from
        # the UNCORRECTED differences wherever the denominator is tiny or
        # negative, and from the corrected ones elsewhere.
        azimuth_from_raw = (np.abs(den) < 1.0) | (den < 0.0)

    k = np.tan(np.pi * float(swing_waves))
    retardance = np.arctan(k * np.hypot(a, b))
    # den < 0 means the extinction state is brighter than the mean swing
    # state, which happens only past a quarter wave: the fringe has turned
    # over, and the answer is on the other branch.
    retardance = np.where(den < 0.0, np.pi - retardance, retardance)
    # Exactly zero denominator: OpenPolScope reports a quarter wave outright.
    retardance = np.where(den == 0.0, np.pi / 2.0, retardance)
    dead = swing_sum == 0.0
    retardance = np.where(dead, np.inf, retardance)

    a_az = np.where(azimuth_from_raw, a_raw, a)
    b_az = np.where(azimuth_from_raw, b_raw, b)
    # OpenPolScope: (azimRef + 90 + 90 * atan2(a, b) / pi) mod 180, in degrees.
    azimuth_deg = (float(azimuth_reference_deg) + 90.0 + np.degrees(np.arctan2(a_az, b_az)) / 2.0) % 180.0
    azimuth_deg = np.where((a_az == 0.0) & (b_az == 0.0), 0.0, azimuth_deg)
    if mirror:
        azimuth_deg = 180.0 - azimuth_deg
    orientation = np.remainder(np.radians(azimuth_deg), np.pi)

    return retardance, orientation, dead
