"""A microscope made of arithmetic, so the calibration can be tested without one.

This is the payoff for having a forward model. :class:`SimulatedPolScope`
satisfies :class:`~polscope_library.calibration.instrument.LiquidCrystalInstrument`
by computing what a camera would read, so a calibration can be run end to end,
scored against the answer it was supposed to find, and regression-tested in
milliseconds.

It ships in the library rather than in ``tests/`` for three reasons: it is pure
numpy, the server will want it for a dry-run mode, and it is the executable
statement of what we believe the instrument does.

Defaults are tuned to the real rig, not to a convenient ideal --
``polarization=0.9925`` reproduces the extinction ratio of 267 measured after
calibration on 2026-08-25, so a test asserting "extinction ratio above 200" is
asserting something the hardware actually achieves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compensator import compensator_stokes
from .instrument import LCAxis, RetardanceLimits

__all__ = ["SimulatedPolScope"]


@dataclass
class SimulatedPolScope:
    """A polarizing microscope that exists only as numbers.

    Parameters
    ----------
    residual : numpy.ndarray, optional
        3x3 rotation acting on ``(S1, S2, S3)`` -- the instrument's own
        birefringence, everything between the compensator and the analyser.
        Identity puts extinction at exactly (0.25, 0.5) waves; a real rig sits
        a little off, and this is what moves it.
    analyzer : tuple of float
        ``(e1, e2, h)`` of an imperfect analyser, projecting
        ``0.5 * (S0 + e1*S1 + e2*S2 + h*S3)``. The ideal circular analyser is
        ``(0, 0, 1)``.
    polarization : float
        Fraction of light that stays polarized. Sets how dark extinction can
        get, and therefore the achievable extinction ratio -- 1.0 would be
        infinite. See the module docstring for the default.
    transmittance : float
        Camera counts at full transmission.
    black_level : float
        Camera offset, added to every frame including dark ones.
    read_noise : float
        Standard deviation of additive Gaussian noise, in counts.
    hysteresis : float
        Fraction of each commanded step the crystal fails to complete, so
        ``get_retardance`` differs from what was asked. Models settle error.
    sample_mueller : numpy.ndarray, optional
        4x4 Mueller matrix of a specimen in the beam. ``None`` is a clear
        field, which is what a calibration runs on.
    """

    residual: np.ndarray = field(default_factory=lambda: np.eye(3))
    analyzer: tuple = (0.0, 0.0, 1.0)
    polarization: float = 0.9925
    transmittance: float = 4000.0
    black_level: float = 102.0
    read_noise: float = 0.0
    hysteresis: float = 0.0
    sample_mueller: np.ndarray | None = None
    limits: RetardanceLimits = field(default_factory=RetardanceLimits)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # Starting state: crystals parked, nothing measured yet.
    _lca: float = 0.25
    _lcb: float = 0.50
    #: Number of frames snapped. A calibration that needs 10x the exposures is
    #: 10x the bench time, so tests assert on this.
    exposures: int = 0

    # -- LiquidCrystalInstrument ------------------------------------------

    def set_retardance(self, axis: LCAxis, waves: float) -> None:
        target = self.limits.clamp(waves)
        current = self._lca if axis == "LCA" else self._lcb
        # Incomplete travel, not random error: the crystal lands short of the
        # commanded value in the direction it was moving.
        reached = current + (1.0 - self.hysteresis) * (target - current)
        if axis == "LCA":
            self._lca = reached
        elif axis == "LCB":
            self._lcb = reached
        else:
            raise ValueError(f"unknown LC axis {axis!r}")

    def get_retardance(self, axis: LCAxis) -> float:
        if axis == "LCA":
            return self._lca
        if axis == "LCB":
            return self._lcb
        raise ValueError(f"unknown LC axis {axis!r}")

    def measure_intensity(self) -> float:
        self.exposures += 1
        stokes = compensator_stokes(self._lca, self._lcb)
        s0, pol = stokes[0], self.residual @ stokes[1:]
        if self.sample_mueller is not None:
            s0, pol = _apply_mueller(self.sample_mueller, s0, pol)
        e1, e2, h = self.analyzer
        fraction = 0.5 * (s0 + self.polarization * (e1 * pol[0] + e2 * pol[1] + h * pol[2]))
        return self._read(self.transmittance * fraction)

    def measure_dark(self) -> float:
        self.exposures += 1
        return self._read(0.0)

    def _read(self, signal: float) -> float:
        counts = signal + self.black_level
        if self.read_noise:
            counts += self.rng.normal(0.0, self.read_noise)
        return float(counts)

    # -- convenience for tests and dry runs --------------------------------

    def intensity_at(self, lca: float, lcb: float) -> float:
        """Intensity at a given pair, leaving the crystals where they were.

        For mapping the cost surface without perturbing a run in progress.
        """
        saved, self._lca, self._lcb = (self._lca, self._lcb), lca, lcb
        try:
            return self.measure_intensity()
        finally:
            self._lca, self._lcb = saved


def _apply_mueller(mueller, s0, pol):
    stokes = np.asarray(mueller, dtype=np.float64) @ np.concatenate([[s0], pol])
    return float(stokes[0]), stokes[1:]
