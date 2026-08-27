"""Measuring a crystal's own voltage-to-retardance curve, from fringes.

Driving a liquid crystal by voltage needs a curve relating the two. The one in
circulation ships with recOrder and appears to be a stock file rather than a
calibration of any particular device -- the copies on our rig were installed
from its driver archives, and two variants disagree enough that one crashes
recOrder's own reader.

The instrument can measure its own, with no reference sample, because the
compensator and analyser are already a polarimeter. Park one crystal at a half
wave and sweep the other on a clear field: the transmitted intensity is

    I(V)  proportional to  1 - sin(2 * pi * retardance(V))

so it oscillates as the retardance sweeps, and **its turning points sit at
retardances known exactly from the physics**. Minima occur where the sine is
+1, at 0.25 waves and every whole wave above; maxima where it is -1, at 0.75
waves and every whole wave above. Consecutive extrema are half a wave apart.

That is the whole method. Find the extrema, label them, and interpolate. No
calibrated retarder, no reference slide -- the fringe pattern is the standard.

Two properties worth having in mind:

* The result is in **waves at whatever wavelength is actually in the beam**,
  which is what the calibration works in. So a self-measured curve is correct
  for the filter that happens to be fitted, without anyone having to know
  which one that is.
* Minima and maxima are distinguishable, and land on different residues (0.25
  versus 0.75 modulo one wave). That is what fixes the absolute retardance
  rather than leaving it known only up to an offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["FringeLandmark", "RetardanceCurve", "curve_from_sweep", "find_fringe_landmarks"]


@dataclass(frozen=True)
class FringeLandmark:
    """One turning point of the fringe pattern, and the retardance it implies."""

    voltage: float
    retardance_waves: float
    #: ``"min"`` (sine = +1) or ``"max"`` (sine = -1).
    kind: str


@dataclass
class RetardanceCurve:
    """Voltage to retardance, and back, for one crystal.

    Monotone decreasing: more voltage, less retardance.
    """

    voltages: np.ndarray
    retardances: np.ndarray
    landmarks: List[FringeLandmark]

    def retardance_for(self, volts) -> np.ndarray:
        """Retardance in waves at a voltage. Clamped outside the measured range."""
        return np.interp(np.asarray(volts, dtype=np.float64), self.voltages, self.retardances)

    def voltage_for(self, waves) -> np.ndarray:
        """Voltage giving a retardance. Clamped outside the measured range."""
        # np.interp needs increasing x, and retardance decreases with voltage.
        return np.interp(np.asarray(waves, dtype=np.float64), self.retardances[::-1], self.voltages[::-1])

    @property
    def span_waves(self) -> tuple:
        return float(self.retardances.min()), float(self.retardances.max())


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(int(window)) / float(window)
    return np.convolve(values, kernel, mode="same")


def _collapse_runs(found, intensity, min_separation):
    """Merge same-type detections that sit on one turning point."""
    kept = []
    for index, kind in found:
        if kept and index - kept[-1][0] < min_separation:
            previous_index, previous_kind = kept[-1]
            if previous_kind != kind:
                continue
            better_here = (kind == "max" and intensity[index] > intensity[previous_index]) or (
                kind == "min" and intensity[index] < intensity[previous_index]
            )
            kept[-1] = (index if better_here else previous_index, kind)
        else:
            kept.append((index, kind))
    return kept


def _drop_shallow(found, intensity, min_prominence):
    """Remove turning points too shallow to be fringes, then restore alternation.

    Dropped one at a time, least prominent first, because removing an extremum
    changes its neighbours' prominence: a ripple between two real fringes looks
    prominent until the ripple beside it goes.
    """
    span = float(np.max(intensity) - np.min(intensity))
    if span <= 0 or min_prominence <= 0:
        return found

    kept = list(found)
    while len(kept) > 2:
        prominences = []
        for position, (index, _kind) in enumerate(kept):
            neighbours = [kept[position - 1][0]] if position > 0 else []
            if position < len(kept) - 1:
                neighbours.append(kept[position + 1][0])
            prominences.append(min(abs(intensity[index] - intensity[n]) for n in neighbours))
        weakest = int(np.argmin(prominences))
        if prominences[weakest] >= min_prominence * span:
            break
        kept.pop(weakest)

    # A drop can leave two of a kind adjacent; keep the more extreme.
    tidied = []
    for index, kind in kept:
        if tidied and tidied[-1][1] == kind:
            previous_index = tidied[-1][0]
            better_here = (kind == "max" and intensity[index] > intensity[previous_index]) or (
                kind == "min" and intensity[index] < intensity[previous_index]
            )
            tidied[-1] = (index if better_here else previous_index, kind)
        else:
            tidied.append((index, kind))
    return tidied


def find_fringe_landmarks(
    voltages: Sequence[float],
    intensities: Sequence[float],
    *,
    smooth_window: int = 5,
    min_separation: int = 3,
    min_prominence: float = 0.15,
    edge_margin: int = 4,
) -> List[FringeLandmark]:
    """Locate the fringe turning points and label them with their retardance.

    Parameters
    ----------
    voltages, intensities
        The sweep, in increasing voltage. Intensity is the mean of a frame on a
        clear field with the partner crystal held at a half wave.
    smooth_window
        Boxcar width used only for *finding* the extrema. Their positions are
        then refined on the unsmoothed data.
    min_separation
        Samples that must separate two extrema. Guards against noise splitting
        one turning point into several.
    min_prominence
        Height a turning point must have, as a fraction of the sweep's full
        intensity range, to count. Real curves are flat and slightly noisy near
        the ends of the voltage range -- the rig's own calibration file is
        non-monotonic there -- and a bare local-extremum test invents fringes
        in those regions. Since the labelling counts half-wave steps between
        extrema, ONE spurious turning point mislabels every landmark before it.
    edge_margin
        Samples at each end of the sweep in which a turning point is ignored.
        The retardance curve is flat near both voltage rails, so tiny wobbles
        there read as extrema -- and they cannot be rejected by prominence,
        because a sweep that begins just short of a real fringe genuinely sits
        near the top or bottom of the intensity range. A turning point at the
        boundary also cannot be labelled honestly: there is no way to tell a
        real fringe from the shoulder of one lying outside the sweep.

    Returns
    -------
    list of FringeLandmark
        Ordered by increasing voltage, so decreasing retardance.

    Raises
    ------
    ValueError
        If fewer than two extrema are found. One fringe cannot anchor a curve:
        the labelling works by counting half-wave steps between them.
    """
    v = np.asarray(voltages, dtype=np.float64)
    intensity = np.asarray(intensities, dtype=np.float64)
    if v.shape != intensity.shape or v.ndim != 1:
        raise ValueError("voltages and intensities must be matching 1-D sequences")

    smoothed = _smooth(intensity, smooth_window)
    interior = np.arange(1, len(v) - 1)
    is_max = (smoothed[1:-1] > smoothed[:-2]) & (smoothed[1:-1] >= smoothed[2:])
    is_min = (smoothed[1:-1] < smoothed[:-2]) & (smoothed[1:-1] <= smoothed[2:])

    found = sorted([(int(i), "max") for i in interior[is_max]] + [(int(i), "min") for i in interior[is_min]])
    if edge_margin > 0:
        last = len(v) - 1 - edge_margin
        found = [(i, kind) for i, kind in found if edge_margin <= i <= last]
    pruned = _collapse_runs(found, intensity, min_separation)
    pruned = _drop_shallow(pruned, intensity, min_prominence)

    if len(pruned) < 2:
        raise ValueError(
            f"found {len(pruned)} fringe extrema; at least 2 are needed to anchor a curve. "
            "Sweep a wider voltage range, or check that the partner crystal is at a half "
            "wave and the field is clear."
        )

    # Label from the highest voltage, where retardance is lowest. A minimum
    # there is 0.25 waves and a maximum 0.75; each earlier extremum is half a
    # wave more. This is what makes the result absolute rather than relative --
    # and it is the one assumption in the method: THE SWEEP MUST REACH BELOW
    # THE LAST FRINGE. It does on this hardware, where 20 V leaves about 0.07
    # waves, but a sweep stopped early would label every fringe half a wave
    # low, consistently and without any sign that it had.
    last_kind = pruned[-1][1]
    base = 0.25 if last_kind == "min" else 0.75
    landmarks = []
    for step, (index, kind) in enumerate(reversed(pruned)):
        retardance = base + 0.5 * step
        expected = "min" if abs((retardance % 1.0) - 0.25) < 1e-9 else "max"
        if kind != expected:
            raise ValueError(
                f"fringe at {v[index]:.3f} V is a {kind} where a {expected} was expected for "
                f"{retardance:.2f} waves. The extrema are not alternating, which usually means "
                "noise created a spurious one -- try a larger smooth_window."
            )
        landmarks.append(FringeLandmark(float(v[index]), retardance, kind))
    return list(reversed(landmarks))


def _retardance_between_fringes(voltages, intensity, landmarks):
    """Retardance at every sample between the first and last fringe.

    The extrema alone give only a handful of points, and the curve between them
    is steeply nonlinear -- interpolating across a whole fringe costs about 0.1
    waves. But the intensity between extrema is not decoration: the fringe
    pattern is

        I = mid - amplitude * sin(2 * pi * retardance)

    so every sample carries a retardance, ambiguous only in which half-cycle it
    belongs to. The extrema resolve that ambiguity, because each one pins the
    phase exactly. Within a segment the phase falls by pi, and

        retardance = retardance_at_fringe - arccos(+/- u) / (2 * pi)

    with the sign set by whether the segment starts at a minimum or a maximum.

    Returns ``(voltages, retardances)`` over the labelled span only.
    """
    highs = [intensity[np.argmin(np.abs(voltages - lm.voltage))] for lm in landmarks if lm.kind == "max"]
    lows = [intensity[np.argmin(np.abs(voltages - lm.voltage))] for lm in landmarks if lm.kind == "min"]
    if not highs or not lows:
        return None
    top, bottom = float(np.mean(highs)), float(np.mean(lows))
    mid, amplitude = 0.5 * (top + bottom), 0.5 * (top - bottom)
    if amplitude <= 0:
        return None

    out_v, out_r = [], []
    for start, end in pairwise(landmarks):
        i0 = int(np.argmin(np.abs(voltages - start.voltage)))
        i1 = int(np.argmin(np.abs(voltages - end.voltage)))
        if i1 <= i0:
            continue
        u = np.clip((mid - intensity[i0 : i1 + 1]) / amplitude, -1.0, 1.0)
        # arccos is monotone on [0, pi], which is exactly one half-cycle.
        delta = np.arccos(u if start.kind == "min" else -u)
        segment = start.retardance_waves - delta / (2.0 * np.pi)
        out_v.append(voltages[i0 : i1 + 1])
        out_r.append(segment)
    if not out_v:
        return None

    v = np.concatenate(out_v)
    r = np.concatenate(out_r)
    order = np.argsort(v)
    v, r = v[order], r[order]
    keep = np.concatenate([[True], np.diff(v) > 0])
    return v[keep], r[keep]


def curve_from_sweep(
    voltages: Sequence[float],
    intensities: Sequence[float],
    *,
    smooth_window: int = 5,
    min_separation: int = 3,
    min_prominence: float = 0.15,
    edge_margin: int = 4,
    extrapolate_to: Optional[tuple] = None,
) -> RetardanceCurve:
    """Build a voltage-to-retardance curve from one fringe sweep.

    Parameters
    ----------
    voltages, intensities
        The sweep, in increasing voltage.
    extrapolate_to
        ``(min_volts, max_volts)`` to extend the curve to, beyond the outermost
        fringes, by continuing the local slope. Extrapolated ends are less
        trustworthy than the interpolated middle and exist only so the search
        has somewhere to go near the rails.

    Returns
    -------
    RetardanceCurve
    """
    landmarks = find_fringe_landmarks(voltages, intensities, smooth_window=smooth_window, min_separation=min_separation)
    # Prefer the per-sample reconstruction; fall back to joining the fringes if
    # the pattern is too degenerate to normalise.
    dense = _retardance_between_fringes(
        np.asarray(voltages, dtype=np.float64), np.asarray(intensities, dtype=np.float64), landmarks
    )
    if dense is not None:
        v, r = dense
    else:
        v = np.array([lm.voltage for lm in landmarks], dtype=np.float64)
        r = np.array([lm.retardance_waves for lm in landmarks], dtype=np.float64)

    if extrapolate_to is not None:
        low, high = float(extrapolate_to[0]), float(extrapolate_to[1])
        if low < v[0]:
            slope = (r[1] - r[0]) / (v[1] - v[0])
            v = np.concatenate([[low], v])
            r = np.concatenate([[r[0] + slope * (low - v[1])], r])
        if high > v[-1]:
            slope = (r[-1] - r[-2]) / (v[-1] - v[-2])
            v = np.concatenate([v, [high]])
            r = np.concatenate([r, [r[-1] + slope * (high - v[-2])]])

    return RetardanceCurve(voltages=v, retardances=r, landmarks=landmarks)
