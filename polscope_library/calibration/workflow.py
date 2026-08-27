"""Running a whole calibration.

Sequences the scheme's states, feeds each search its reference, and reports
what it found -- including when it found something poor, because a marginal
calibration the operator can see beats one thrown away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .compensator import ideal_palette
from .instrument import LiquidCrystalInstrument
from .session import DEFAULT_RATIO, CalibrationSession, TracePoint
from .states import moves_for
from .strategies import STRATEGIES, SearchStrategy, SinglePassSearch

__all__ = ["CalibrationResult", "CalibrationSettings", "assess", "calibrate", "extinction_ratio", "resolve_black_level"]


@dataclass
class CalibrationSettings:
    swing: float = 0.03
    scheme: str = "5-State"
    wavelength_nm: float = 546.0
    #: Explicit black level. ``None`` measures one; see :func:`resolve_black_level`.
    black_level: Optional[float] = None
    #: Frames to average when measuring the black level.
    dark_frames: int = 20
    #: Half-width of the search window around each seed, in waves.
    search_bound: float = 0.05
    ratio: float = DEFAULT_RATIO


@dataclass
class CalibrationResult:
    palette: Dict[str, Tuple[float, float]]
    swing: float
    scheme: str
    wavelength_nm: float
    black_level: float
    black_level_source: str
    extinction_ratio: float
    assessment: str
    intensities: Dict[str, float]
    trace: List[TracePoint] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    exposures: int = 0

    @property
    def states(self) -> List[str]:
        """Channel names in scheme order, which is the order to acquire in."""
        return list(self.palette)


def extinction_ratio(swing: float, black_level: float, i_extinction: float, i_elliptical: float) -> float:
    """How much brighter a swing state is than extinction, per recOrder.

    Not rounded -- recOrder rounds to 2 decimals inside the calculation, which
    belongs at display time. Verified against the rig: its reported 267.54
    reproduces as 263.74 from the reference-slide intensities, the residual
    being that the bench figure is measured on a blank field.
    """
    denominator = i_extinction - black_level
    if denominator <= 0:
        return float("nan")
    return (1.0 / np.sin(np.pi * swing) ** 2) * (i_elliptical - i_extinction) / denominator + 1.0


def assess(ratio: float) -> str:
    """recOrder's acceptance bands."""
    if not np.isfinite(ratio):
        return "unmeasurable"
    if ratio >= 100:
        return "good"
    if ratio >= 80:
        return "acceptable"
    return "poor"


def resolve_black_level(instrument, settings: CalibrationSettings, warnings: List[str]) -> Tuple[float, str]:
    """Black level, by priority: configured, else measured, else assumed zero.

    Never a shutter and never a prompt. recOrder asks the operator to close one
    by hand and blocks on stdin from a worker thread; on a rig without a
    shutter that is an unanswerable question, and it is what stopped recOrder
    being usable here.
    """
    if settings.black_level is not None:
        return float(settings.black_level), "configured"
    try:
        frames = [float(instrument.measure_dark()) for _ in range(max(1, settings.dark_frames))]
        return float(np.mean(frames)), "measured"
    except NotImplementedError:
        warnings.append(
            "No black level configured and the instrument cannot darken the field; assuming 0. "
            "The extinction ratio will be overstated -- a 50-count error moves it by roughly 10%."
        )
        return 0.0, "assumed"


def calibrate(
    instrument: LiquidCrystalInstrument,
    settings: Optional[CalibrationSettings] = None,
    *,
    strategy: Optional[SearchStrategy] = None,
) -> CalibrationResult:
    """Find the LC settings for every state of a scheme.

    Parameters
    ----------
    instrument
        Anything satisfying the seam -- a microscope, or
        :class:`~polscope_library.calibration.simulator.SimulatedPolScope`.
    strategy
        A search, or a name from
        :data:`~polscope_library.calibration.strategies.STRATEGIES`. Defaults
        to a single pass.
    """
    settings = settings or CalibrationSettings()
    if isinstance(strategy, str):
        strategy = STRATEGIES[strategy]()
    strategy = strategy or SinglePassSearch()

    warnings: List[str] = []
    black_level, black_source = resolve_black_level(instrument, settings, warnings)

    session = CalibrationSession(instrument, swing=settings.swing, ratio=settings.ratio)
    palette: Dict[str, Tuple[float, float]] = {}
    intensities: Dict[str, float] = {}
    reference = black_level
    i_extinction = i_elliptical = float("nan")

    moves = moves_for(settings.scheme)
    for index, move in enumerate(moves):
        session.state = move.name
        # Seeds come from the system matrix, re-referenced onto the extinction
        # point once it is known. Not transcribed constants: the cost is
        # symmetric about extinction, so a seed on the wrong side of a crystal
        # converges perfectly onto the mirror-image state and the orientation
        # map comes out reflected, with nothing to indicate it.
        if move.kind == "extinction":
            seed = (0.0, 0.0)  # unused; the search grids the whole range
        else:
            seed = tuple(ideal_palette(settings.swing, settings.scheme, session.lca_ext, session.lcb_ext)[index])
        # The search window must not reach the state's mirror image. Both lie
        # on the reference intensity level -- they are the antipodal Stokes
        # pair -- so a window wide enough to contain both lets the search
        # converge perfectly onto the wrong one, and the orientation map comes
        # out reflected. The mirror sits at twice the seed's displacement from
        # extinction, so half of that is the natural bound; recOrder reaches
        # the same numbers by hardcoding swing for 5-State and swing/ratio for
        # 4-State, which is this rule with the arithmetic already done.
        bound = settings.search_bound
        if move.kind in ("scan", "constrained"):
            axis_index = 0 if move.axis == "LCA" else 1
            centre = session.lca_ext if move.axis == "LCA" else session.lcb_ext
            displacement = abs(seed[axis_index] - centre)
            if displacement > 0:
                bound = min(bound, displacement)
        outcome = strategy.solve(session, move, seed=seed, bound=bound, reference=reference)
        palette[move.channel] = (outcome.lca, outcome.lcb)
        intensities[move.channel] = outcome.intensity

        if move.kind == "extinction":
            session.lca_ext, session.lcb_ext = outcome.lca, outcome.lcb
            i_extinction = outcome.intensity
        elif move.kind == "open_loop":
            # This state's intensity is what every later state is matched to,
            # so the swing is defined here rather than searched for.
            i_elliptical = outcome.intensity
            reference = i_elliptical

    ratio = extinction_ratio(settings.swing, black_level, i_extinction, i_elliptical)
    grade = assess(ratio)
    if grade == "poor":
        warnings.append(
            f"Extinction ratio {ratio:.1f} is below 80 (recOrder band: poor). The calibration is "
            "recorded anyway -- inspect it, and consider re-running on a cleaner blank field."
        )

    return CalibrationResult(
        palette=palette,
        swing=settings.swing,
        scheme=settings.scheme,
        wavelength_nm=settings.wavelength_nm,
        black_level=black_level,
        black_level_source=black_source,
        extinction_ratio=ratio,
        assessment=grade,
        intensities=intensities,
        trace=session.trace,
        warnings=warnings,
        exposures=getattr(instrument, "exposures", 0),
    )
