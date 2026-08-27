"""The cost functions a search minimises, and the record of what it tried.

This is the only place that touches the instrument during a search. The
searches themselves see nothing but floats, which is what lets them be tested
against a simulator -- and what recOrder cannot do, because its optimizers hold
the Micro-Manager core directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .instrument import LCAxis, LiquidCrystalInstrument, other_axis

__all__ = ["CalibrationSession", "TracePoint"]

#: LC-B excursion per unit of LC-A, for the constrained 4-State moves.
#: recOrder's value, from a simulation of the compensator's ellipticity. Pure
#: geometry gives tan(60 deg) = 1.7321; the 3.5% excess is thought to absorb
#: the difference between the two crystals' response curves.
DEFAULT_RATIO = 1.793


@dataclass(frozen=True)
class TracePoint:
    """One measurement, kept so a calibration can be explained afterwards."""

    state: str
    lca: float
    lcb: float
    intensity: float
    residual: float


@dataclass
class CalibrationSession:
    """Drives the instrument on behalf of a search.

    Parameters
    ----------
    instrument
        Anything satisfying :class:`LiquidCrystalInstrument`.
    swing
        Swing in waves.
    ratio
        LC-B excursion per unit LC-A for constrained moves.
    """

    instrument: LiquidCrystalInstrument
    swing: float
    ratio: float = DEFAULT_RATIO

    lca_ext: Optional[float] = None
    lcb_ext: Optional[float] = None
    trace: List[TracePoint] = field(default_factory=list)
    #: Name of the state currently being solved, for the trace.
    state: str = ""

    # -- instrument access -------------------------------------------------

    def set_lc(self, axis: LCAxis, waves: float) -> float:
        """Command one crystal and return where it actually landed."""
        self.instrument.set_retardance(axis, float(waves))
        return self.instrument.get_retardance(axis)

    def get_lc(self, axis: LCAxis) -> float:
        return float(self.instrument.get_retardance(axis))

    def measure(self, reference: float = 0.0) -> float:
        """Snap, record the point, and return the intensity."""
        intensity = float(self.instrument.measure_intensity())
        self.trace.append(
            TracePoint(self.state, self.get_lc("LCA"), self.get_lc("LCB"), intensity, intensity - reference)
        )
        return intensity

    # -- cost functions ----------------------------------------------------

    def objective(self, x, axis: LCAxis, reference: float) -> float:
        """How far one crystal's setting puts the intensity from a reference.

        ``x`` is coerced to a scalar because scipy hands minimisers a 0-d array
        and recOrder unwraps it by testing ``isinstance(x, list)``, which is
        true for neither.
        """
        self.set_lc(axis, _scalar(x))
        return abs(self.measure(reference) - reference)

    def constrained_objective(self, x, axis: LCAxis, reference: float, slope: float) -> float:
        """Cost of a joint move: one crystal free, the other following a line.

        The partner crystal is driven to ``lcb_ext + slope * (x - lca_ext)``.

        ``slope`` is signed and is *derived* from the scheme's ideal palette,
        not transcribed. recOrder hardcodes ``ratio = 1.793`` with a separate
        sign per state, and the sign convention has to agree with whichever way
        the compensator's crystals are oriented -- an agreement nothing checks.
        Deriving the whole slope from the system matrix makes it agree by
        construction, and reports the magnitude rather than assuming it.

        The instrument clamps the partner if the line leaves its travel, which
        is why :meth:`LiquidCrystalInstrument.set_retardance` must clamp rather
        than raise: recOrder raises here, aborting the run and discarding every
        state already solved.
        """
        value = _scalar(x)
        self.set_lc(axis, value)
        partner = self.lcb_ext + slope * (value - self.lca_ext)
        self.set_lc(other_axis(axis), partner)
        return abs(self.measure(reference) - reference)

    def intensity_at(self, lca: float, lcb: float, reference: float = 0.0) -> float:
        """Set both crystals and measure. Used by the coarse extinction grid."""
        self.set_lc("LCA", lca)
        self.set_lc("LCB", lcb)
        return self.measure(reference)


def _scalar(x) -> float:
    return float(np.asarray(x).reshape(()))
