"""How the search for each state is conducted.

recOrder ships two optimizers and lets you pick between them. Underneath they
are the *same* numerical method -- ``fminbound`` and
``minimize_scalar(method="bounded")`` both call scipy's bounded Brent. What
differs is policy: one takes a single pass with a fine sweep for extinction,
the other iterates until the residual is small enough. So they are named for
that here, rather than for a numerical distinction they do not have.

Both are kept because they behave differently on a real rig -- the iterative
one re-reads the crystals each round and can walk its window toward a solution
that a single pass leaves on the table, at the cost of more exposures. Choosing
between them is the operator's, not ours.

Every defect the audit found in either is fixed; see the class docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np

from .instrument import RetardanceLimits
from .session import CalibrationSession
from .states import StateMove

__all__ = ["STRATEGIES", "IterativeRefineSearch", "SearchStrategy", "SinglePassSearch", "StateOutcome"]


@dataclass(frozen=True)
class StateOutcome:
    """Where a state ended up."""

    lca: float
    lcb: float
    intensity: float
    #: Signed difference from the reference the state was matched to.
    residual: float


class SearchStrategy(Protocol):
    def solve(
        self, session: CalibrationSession, move: StateMove, *, bound: float, reference: float
    ) -> StateOutcome: ...


def _bounded_min(fn, lo: float, hi: float, args: tuple) -> float:
    """Minimise a scalar function on an interval. Both strategies use this."""
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(fn, bounds=(lo, hi), args=args, method="bounded")
    return float(result.x)


def _window(centre: float, half_width: float, limits: RetardanceLimits) -> tuple:
    """A search interval around a point, clipped to the crystal's travel.

    recOrder hardcodes 0.01/1.6 here, and its unreachable ``mode == "voltage"``
    branch clips to 2.2 -- a voltage figure applied to a value in waves.
    """
    lo = max(limits.min_waves, centre - half_width)
    hi = min(limits.max_waves, centre + half_width)
    if hi <= lo:  # centre pinned against a rail
        lo, hi = limits.min_waves, limits.max_waves
    return lo, hi


@dataclass
class SinglePassSearch:
    """One pass per state; a coarse grid plus a fine sweep for extinction.

    recOrder's ``min_scalar``, its default. Fixed here:

    * State I120 passed ``"LCB"`` to the constrained cost where I60 passed
      ``"LCA"``, so LC-A never moved during that optimisation and the returned
      value -- an LC-B setting -- was recorded as LC-A. The crystal now comes
      from the scheme table, which both strategies read.
    * The "pick the best iterate" step indexed row 2 of the results array
      instead of column 2, so it minimised over an arbitrary row and could
      select nothing at all.
    * ``thresh`` and ``n_iter`` were accepted and silently ignored. They are
      not accepted here; :class:`IterativeRefineSearch` is what uses them.
    * The reported intensity was ``|I - reference| + reference``, which equals
      the measured value only when the intensity exceeds the reference. The
      optimum is now re-measured.
    """

    #: Half-width of the fine sweep around the coarse extinction result.
    fine_bound: float = 0.01
    #: Coarse grid over LC-A and LC-B, in waves.
    grid_step: float = 0.1
    grid_lca: tuple = (0.01, 0.5)
    grid_lcb: tuple = (0.25, 0.75)

    def solve(
        self, session: CalibrationSession, move: StateMove, *, seed: tuple, bound: float, reference: float
    ) -> StateOutcome:
        if move.kind == "extinction":
            return self._extinction(session, reference)
        if move.kind == "open_loop":
            return self._open_loop(session, seed, reference)
        if move.kind == "scan":
            return self._scan(session, move, seed, bound=bound, reference=reference)
        if move.kind == "constrained":
            return self._constrained(session, move, seed, bound=bound, reference=reference)
        raise ValueError(f"unknown move kind {move.kind!r} for state {move.name!r}")

    # -- per kind ----------------------------------------------------------

    def _extinction(self, session: CalibrationSession, reference: float) -> StateOutcome:
        limits = session.instrument.limits
        best = (np.inf, None, None)
        for lca in np.arange(*self.grid_lca, self.grid_step):
            for lcb in np.arange(*self.grid_lcb, self.grid_step):
                value = session.intensity_at(float(lca), float(lcb), reference)
                if value < best[0]:
                    best = (value, float(lca), float(lcb))
        _, lca, lcb = best
        session.set_lc("LCA", lca)
        session.set_lc("LCB", lcb)

        # Coarse then fine, alternating crystals. Coordinate descent, as
        # recOrder does it -- neither strategy minimises jointly.
        for half_width in (self.grid_step, self.fine_bound):
            for axis in ("LCA", "LCB"):
                lo, hi = _window(session.get_lc(axis), half_width, limits)
                session.set_lc(axis, _bounded_min(session.objective, lo, hi, (axis, reference)))

        return self._settle(session, reference)

    def _open_loop(self, session: CalibrationSession, seed: tuple, reference: float) -> StateOutcome:
        lca, lcb = seed
        session.set_lc("LCA", lca)
        session.set_lc("LCB", lcb)
        return self._settle(session, reference)

    def _scan(
        self, session: CalibrationSession, move: StateMove, seed: tuple, *, bound: float, reference: float
    ) -> StateOutcome:
        lca, lcb = seed
        session.set_lc("LCA", lca)
        session.set_lc("LCB", lcb)
        # Bounds follow the scanned crystal. recOrder's broken optimizer used
        # LC-A's bounds for the LC-B scans of states 45 and 135.
        lo, hi = _window(session.get_lc(move.axis), bound, session.instrument.limits)
        session.set_lc(move.axis, _bounded_min(session.objective, lo, hi, (move.axis, reference)))
        return self._settle(session, reference)

    def _constrained(
        self, session: CalibrationSession, move: StateMove, seed: tuple, *, bound: float, reference: float
    ) -> StateOutcome:
        lca, lcb = seed
        session.set_lc("LCA", lca)
        session.set_lc("LCB", lcb)
        # Slope of the line this state lies on, taken from where the seed sits
        # relative to extinction -- derived, not a transcribed constant.
        slope = (lcb - session.lcb_ext) / (lca - session.lca_ext) if lca != session.lca_ext else 0.0
        lo, hi = _window(session.get_lc(move.axis), bound, session.instrument.limits)
        best = _bounded_min(session.constrained_objective, lo, hi, (move.axis, reference, slope))
        session.constrained_objective(best, move.axis, reference, slope)
        return self._settle(session, reference)

    @staticmethod
    def _settle(session: CalibrationSession, reference: float) -> StateOutcome:
        """Measure where the crystals actually ended up.

        Re-measured rather than reconstructed from the optimiser's cost, and
        read back from the instrument rather than assumed, so the recorded
        palette is what the hardware will reproduce.
        """
        intensity = session.measure(reference)
        return StateOutcome(session.get_lc("LCA"), session.get_lc("LCB"), intensity, intensity - reference)


@dataclass
class IterativeRefineSearch:
    """Repeat the single pass, re-centring on the crystals, until close enough.

    recOrder's ``brent``, which raises on every path it has: pre-refactor call
    arity on ``get_lc``/``set_lc``, a ``(device, property)`` tuple passed where
    a crystal name is required, LC-A bounds used for LC-B scans, a missing
    argument to the constrained cost, and an LC-A result written where LC-B's
    was meant. None of that survives here, because the work is delegated to
    :class:`SinglePassSearch` and the differences are policy only.

    Its selection step is also fixed: recOrder minimised over ``|lca|``, logged
    the columns under the wrong names, then returned the last iterate anyway,
    making the whole "best iterate" search dead code. This returns the best.

    Parameters
    ----------
    n_iter
        Maximum passes.
    thresh_percent
        Stop once the residual is within this percentage of the reference.
        recOrder accepts both of these on the other strategy and ignores them.
    """

    n_iter: int = 5
    thresh_percent: float = 1.0
    inner: SinglePassSearch = None

    def __post_init__(self):
        if self.inner is None:
            self.inner = SinglePassSearch()

    def solve(
        self, session: CalibrationSession, move: StateMove, *, seed: tuple, bound: float, reference: float
    ) -> StateOutcome:
        best = None
        for _ in range(max(1, self.n_iter)):
            outcome = self.inner.solve(session, move, seed=seed, bound=bound, reference=reference)
            if best is None or abs(outcome.residual) < abs(best.residual):
                best = outcome
            if move.kind == "open_loop":
                break  # nothing to refine: the seed defines the state
            if reference and 100.0 * abs(outcome.residual) / abs(reference) <= self.thresh_percent:
                break
        return best


#: Selectable by name, so the choice can live in configuration.
STRATEGIES: Dict[str, type] = {
    "single_pass": SinglePassSearch,
    "iterative": IterativeRefineSearch,
}
