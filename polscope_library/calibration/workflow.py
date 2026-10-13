"""Running a whole calibration.

Sequences the scheme's states, feeds each search its reference, and reports
what it found -- including when it found something poor, because a marginal
calibration the operator can see beats one thrown away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .compensator import compensator_stokes, ideal_palette
from .instrument import LiquidCrystalInstrument
from .session import DEFAULT_RATIO, CalibrationSession, TracePoint
from .states import moves_for
from .strategies import STRATEGIES, SearchStrategy, SinglePassSearch

__all__ = [
    "CalibrationResult",
    "CalibrationSettings",
    "InsufficientSignalError",
    "assess",
    "calibrate",
    "check_palette_geometry",
    "extinction_ratio",
    "extinction_ratio_openpolscope",
    "resolve_black_level",
    "swing_state_distances",
]


class InsufficientSignalError(RuntimeError):
    """The field is too dark to calibrate; raised before the swing states are searched.

    A search on a dark field does not fail -- it converges on noise and hands
    back a palette that looks like any other. The first rig runs (2026-10-12
    and -13) did exactly that: frame means of 1 to 17 counts, a swing state
    0.002 waves from extinction, and an extinction ratio of 17 that was pure
    black-level arithmetic. This is raised instead, as soon as the extinction
    grid has shown how bright the field gets.
    """


@dataclass
class CalibrationSettings:
    swing_waves: float = 0.03
    scheme: str = "5-State"
    wavelength_nm: float = 546.0
    #: Explicit black level. ``None`` measures one; see :func:`resolve_black_level`.
    black_level: Optional[float] = None
    #: Frames to average when measuring the black level.
    dark_frames: int = 20
    #: Half-width of the search window around each seed, in waves.
    search_bound_waves: float = 0.05
    ratio: float = DEFAULT_RATIO
    #: Refuse the run when the brightest frame of the extinction grid averages
    #: fewer counts than this (:class:`InsufficientSignalError`). ``None``
    #: disables the check. The library has no idea what a count is worth, so
    #: the caller sets this from the camera -- the microscope server uses a
    #: tenth of full scale.
    min_grid_peak: Optional[float] = None


@dataclass
class CalibrationResult:
    palette_waves: Dict[str, Tuple[float, float]]
    swing_waves: float
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
        return list(self.palette_waves)


def extinction_ratio(swing_waves: float, black_level: float, i_extinction: float, i_elliptical: float) -> float:
    """How much brighter a swing state is than extinction, per recOrder.

    Not rounded -- recOrder rounds to 2 decimals inside the calculation, which
    belongs at display time. Verified against the rig: its reported 267.54
    reproduces as 263.74 from the reference-slide intensities, the residual
    being that the bench figure is measured on a blank field.
    """
    denominator = i_extinction - black_level
    if denominator <= 0:
        return float("nan")
    return (1.0 / np.sin(np.pi * swing_waves) ** 2) * (i_elliptical - i_extinction) / denominator + 1.0


def extinction_ratio_openpolscope(
    swing_waves: float, black_level: float, i_extinction: float, swing_intensities: Sequence[float]
) -> float:
    """The extinction ratio as OpenPolScope 3.20b computes it.

    Differs from :func:`extinction_ratio` (recOrder's) in two ways: it averages
    over EVERY swing state rather than taking one elliptical state, and it has
    no ``+ 1`` term::

        ER = (sum(I_swing) - n * I_ext) / (n * (I_ext - black) * sin(pi * swing) ** 2)

    With all swing states equal the two differ by exactly 1, which at this
    instrument's ~270 is below the run-to-run noise -- but a number read off
    the OpenPolScope screen and one from recOrder are not the same quantity,
    and this is the one to compare against the former.

    Established by decompiling OpenPolScope's ``calculate(float[], float,
    float)`` and checking it against the rig's reported value; the formula is
    too short to be copyrightable expression and is restated here in our own
    terms.
    """
    values = [float(v) for v in swing_intensities]
    n = len(values)
    if n == 0:
        raise ValueError("swing_intensities must contain at least one state")
    denominator = n * (float(i_extinction) - float(black_level)) * np.sin(np.pi * float(swing_waves)) ** 2
    if denominator <= 0:
        return float("nan")
    return (sum(values) - n * float(i_extinction)) / denominator


def assess(ratio: float) -> str:
    """recOrder's acceptance bands."""
    if not np.isfinite(ratio):
        return "unmeasurable"
    if ratio >= 100:
        return "good"
    if ratio >= 80:
        return "acceptable"
    return "poor"


def swing_state_distances(palette_waves: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Poincare-sphere distance of each state from the extinction state.

    The first entry of the palette is the extinction state; every other entry
    is a swing state. Distances are computed through the compensator forward
    model, so this reads the palette as polarization rather than as numbers.
    """
    items = list(palette_waves.items())
    if not items:
        return {}
    ext = np.asarray(compensator_stokes(*items[0][1]), dtype=np.float64)
    out = {}
    for channel, (lca_waves, lcb_waves) in items[1:]:
        delta = np.asarray(compensator_stokes(lca_waves, lcb_waves), dtype=np.float64) - ext
        out[channel] = float(np.linalg.norm(delta[1:]))
    return out


def check_palette_geometry(
    palette_waves: Dict[str, Tuple[float, float]], swing_waves: float, tolerance: float = 0.1
) -> List[str]:
    """Verify the swing states sit where the scheme says they must.

    Every swing state is the same angular step away from extinction, just in a
    different direction on the Poincare sphere, so all of them must be the
    same distance from it: exactly ``2*sin(pi*swing)``.

    This is worth checking separately from the extinction ratio, which only
    describes the extinction state and is blind to where the other four
    landed. A palette can post an excellent extinction ratio while its swing
    states are misplaced, and nothing downstream would notice -- the
    reconstruction builds its instrument matrix from ``swing`` and ``scheme``
    and never reads the palette, so a misplaced state is silently treated as
    if it were where it should be. That is a real error in the data which no
    reconstruction, ours or anyone's, can see.

    **A real calibration may legitimately fail this, so read a warning as
    "the ideal-geometry assumption is strained here", not as "the palette is
    broken".** Both OpenPolScope palettes we have from this instrument fail
    it, on the LC-B pair, and differently from each other: the one read off
    the screen gives 0.3004 / 0.0879, while the one in OpenPolScope's own
    registry gives 0.1193 / 0.1506 against an expected 0.1882. Two out of two
    is a poor basis for calling the other program wrong. The likelier reading
    is that a calibration searches for the values that produce the wanted
    states on the REAL optics -- imperfect crystal orientation, residual
    retardance elsewhere in the path -- and those are not the ideal ones. A
    calibration that returned exactly the ideal values would not be measuring
    anything.

    It is still worth reporting, because our reconstruction assumes the ideal
    geometry: it builds the instrument matrix from ``swing`` and ``scheme``
    and never reads the palette. So the size of this deviation is the size of
    the assumption we are making, and nothing else surfaces it.

    ``tolerance`` is a fraction of the expected distance.
    """
    expected = 2.0 * np.sin(np.pi * float(swing_waves))
    warnings: List[str] = []
    if expected <= 0:
        return warnings
    for channel, distance in swing_state_distances(palette_waves).items():
        error = abs(distance - expected) / expected
        if error > tolerance:
            warnings.append(
                f"Swing state {channel} sits {distance:.4f} from extinction on the "
                f"Poincare sphere, against {expected:.4f} for an ideal compensator at "
                f"swing {swing_waves} ({error * 100:.0f}% off). This is not proof of a bad "
                "calibration -- real optics are not ideal, and both OpenPolScope "
                "palettes from this instrument deviate here too. It is reported "
                "because the reconstruction builds its instrument matrix from the "
                "swing and never reads the palette, so this figure is the size of "
                "the assumption that inversion is making."
            )
    return warnings


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

    session = CalibrationSession(instrument, swing_waves=settings.swing_waves, ratio=settings.ratio)
    palette_waves: Dict[str, Tuple[float, float]] = {}
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
            seed = tuple(
                ideal_palette(settings.swing_waves, settings.scheme, session.lca_ext_waves, session.lcb_ext_waves)[
                    index
                ]
            )
        # The search window must not reach the state's mirror image. Both lie
        # on the reference intensity level -- they are the antipodal Stokes
        # pair -- so a window wide enough to contain both lets the search
        # converge perfectly onto the wrong one, and the orientation map comes
        # out reflected. The mirror sits at twice the seed's displacement from
        # extinction, so half of that is the natural bound; recOrder reaches
        # the same numbers by hardcoding swing for 5-State and swing/ratio for
        # 4-State, which is this rule with the arithmetic already done.
        bound_waves = settings.search_bound_waves
        if move.kind in ("scan", "constrained"):
            axis_index = 0 if move.axis == "LCA" else 1
            centre = session.lca_ext_waves if move.axis == "LCA" else session.lcb_ext_waves
            displacement = abs(seed[axis_index] - centre)
            if displacement > 0:
                bound_waves = min(bound_waves, displacement)
        outcome = strategy.solve(session, move, seed=seed, bound_waves=bound_waves, reference=reference)
        palette_waves[move.channel] = (outcome.lca_waves, outcome.lcb_waves)
        intensities[move.channel] = outcome.intensity

        if move.kind == "extinction":
            session.lca_ext_waves, session.lcb_ext_waves = outcome.lca_waves, outcome.lcb_waves
            i_extinction = outcome.intensity
            if settings.min_grid_peak is not None:
                grid_peak = max(p.intensity for p in session.trace if p.state == move.name)
                if grid_peak < settings.min_grid_peak:
                    raise InsufficientSignalError(
                        f"the field is too dark to calibrate: the brightest frame of the "
                        f"extinction search averaged {grid_peak:.1f} counts, below the "
                        f"{settings.min_grid_peak:.1f} needed to tell the states apart "
                        f"(extinction found at {i_extinction:.1f}). Raise the exposure or the "
                        "illumination and run again."
                    )
        elif move.kind == "open_loop":
            # This state's intensity is what every later state is matched to,
            # so the swing is defined here rather than searched for.
            i_elliptical = outcome.intensity
            reference = i_elliptical

    ratio = extinction_ratio(settings.swing_waves, black_level, i_extinction, i_elliptical)
    grade = assess(ratio)
    warnings.extend(check_palette_geometry(palette_waves, settings.swing_waves))
    if grade == "poor":
        warnings.append(
            f"Extinction ratio {ratio:.1f} is below 80 (recOrder band: poor). The calibration is "
            "recorded anyway -- inspect it, and consider re-running on a cleaner blank field."
        )

    return CalibrationResult(
        palette_waves=palette_waves,
        swing_waves=settings.swing_waves,
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
