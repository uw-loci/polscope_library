"""Forward model of the liquid-crystal universal compensator.

The reconstruction side of this package starts *after* calibration: given the
five calibrated states, :func:`polscope_library.calculate_stokes_to_intensity_matrix`
knows what polarization each one produces, parameterised by swing alone.
Calibration is the problem of *finding* the crystal settings that produce those
states, so it needs the one thing that matrix does not express -- how a pair of
LC retardances maps to a polarization state.

Neither waveorder nor recOrder models this. recOrder discovers the states
empirically, by driving the crystals and minimising measured intensity, and
never writes down the relationship. That works, but it means nothing about the
calibration can be checked without a microscope. With a forward model the whole
optimizer becomes testable on a simulated instrument.

The optical train
-----------------
Linear polarizer at 0 deg, LC-A with its axis at 45 deg, LC-B with its axis at
0 deg, then the specimen and a circular analyser::

    S_in = (1, 1, 0, 0)                     after the polarizer
    LC-A (45 deg): S2 fixed, (S3, S1) rotate by delta_A
    LC-B (0 deg):  S1 fixed, (S2, S3) rotate by delta_B

which reduces to :func:`compensator_stokes` below.

Verified: at the ideal palette (extinction at LC-A = 1/4 wave, LC-B = 1/2 wave,
plus and minus the swing on each crystal in turn) this reproduces the rows of
``calculate_stokes_to_intensity_matrix`` to machine precision, for both the
4-State and 5-State schemes. That ties the new model to one already validated
against real OpenPolScope output, so it needs no ground truth of its own.

What this model does NOT yet explain: a real calibration off the rig swings
LC-A symmetrically but LC-B by +0.048 / -0.014 waves about its extinction
value. Residual instrument birefringence does not account for it -- it moves
the extinction point while leaving the swing states symmetric. See
``tests/test_compensator.py`` for the open hypotheses. Treat simulated
palettes as idealised in that one respect.

It also *explains* a structure previously taken as an empirical rule: a valid
palette repeats the base LC-A value on exactly three states and the base LC-B
value on three, because S1 depends only on LC-A while S2 and S3 depend on both.
The states that swing LC-A therefore all share the extinction LC-B, and vice
versa. A palette without that structure did not come from a calibration.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EXTINCTION_LCA_WAVES",
    "EXTINCTION_LCB_WAVES",
    "compensator_stokes",
    "ideal_palette",
    "retardances_for_stokes",
]

# Nominal extinction: a quarter wave on LC-A and a half wave on LC-B send the
# polarizer's linear state to circular, which the circular analyser blocks.
# Real calibrations land near but not exactly here -- the rig measured
# (0.248, 0.451) -- because residual birefringence in the optics shifts it.
# These are only the starting point of the search.
EXTINCTION_LCA_WAVES = 0.25
EXTINCTION_LCB_WAVES = 0.50


def compensator_stokes(lca_waves, lcb_waves):
    """Stokes vector produced by the compensator at the given LC retardances.

    Parameters
    ----------
    lca_waves, lcb_waves : float or array_like
        Retardance of each liquid crystal, in waves. Broadcast together.

    Returns
    -------
    numpy.ndarray
        Stokes vector ``(S0, S1, S2, S3)`` along the LAST axis, normalised so
        ``S0 == 1``. Fully polarized: ``S1**2 + S2**2 + S3**2 == 1``.
    """
    delta_a = 2.0 * np.pi * np.asarray(lca_waves, dtype=np.float64)
    delta_b = 2.0 * np.pi * np.asarray(lcb_waves, dtype=np.float64)
    sin_a = np.sin(delta_a)
    return np.stack(
        [
            np.ones_like(delta_a * delta_b),
            np.broadcast_to(np.cos(delta_a), np.broadcast(delta_a, delta_b).shape),
            -sin_a * np.sin(delta_b),
            sin_a * np.cos(delta_b),
        ],
        axis=-1,
    )


def retardances_for_stokes(stokes):
    """Invert :func:`compensator_stokes`: the LC settings producing a state.

    Used to derive a scheme's ideal palette from its system matrix, so the
    starting guess for a calibration comes from the optics rather than from a
    table of constants.

    Parameters
    ----------
    stokes : array_like
        ``(S0, S1, S2, S3)`` along the last axis. Only the normalised
        orientation matters; ``S0`` is ignored.

    Returns
    -------
    (lca_waves, lcb_waves) : tuple of numpy.ndarray
        Retardances in waves, each wrapped to ``[0, 1)``.

    Notes
    -----
    ``delta_A`` is taken on the principal branch of ``arccos``, i.e.
    ``sin(delta_A) >= 0``. The compensator is periodic and the reflected
    solution is equally valid optically, so this picks the representative in
    the range the hardware actually spans.
    """
    s = np.asarray(stokes, dtype=np.float64)
    s1, s2, s3 = s[..., 1], s[..., 2], s[..., 3]
    delta_a = np.arccos(np.clip(s1, -1.0, 1.0))
    # atan2 recovers delta_B without dividing by sin(delta_A), which vanishes
    # when the compensator sits at a full or half wave on LC-A.
    delta_b = np.arctan2(-s2, s3)
    return (delta_a / (2.0 * np.pi)) % 1.0, (delta_b / (2.0 * np.pi)) % 1.0


def ideal_palette(swing, scheme="5-State", lca_ext=EXTINCTION_LCA_WAVES, lcb_ext=EXTINCTION_LCB_WAVES):
    """Nominal LC settings for a scheme, as the starting point for calibration.

    Derived by inverting the scheme's system matrix through
    :func:`retardances_for_stokes` rather than hard-coded, so the two can never
    disagree. For 5-State the result is the familiar plus/minus swing on each
    crystal; for 4-State the states are not axis-aligned and the arithmetic is
    not obvious by inspection, which is precisely why deriving it is better than
    writing it down.

    Parameters
    ----------
    swing : float
        Swing in waves.
    scheme : {"4-State", "5-State"}
    lca_ext, lcb_ext : float
        Measured extinction point, if known. The palette is expressed relative
        to it, so a rig whose extinction sits away from the nominal quarter/half
        wave still gets a usable starting guess.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_states, 2)`` of ``(lca_waves, lcb_waves)``, extinction first.
    """
    from polscope_library import calculate_stokes_to_intensity_matrix

    matrix = calculate_stokes_to_intensity_matrix(swing, scheme)
    lca, lcb = retardances_for_stokes(matrix)
    # Re-reference onto the measured extinction point. Row 0 is extinction by
    # construction in both schemes.
    lca = (lca - lca[0] + lca_ext) % 1.0
    lcb = (lcb - lcb[0] + lcb_ext) % 1.0
    return np.stack([lca, lcb], axis=-1)
