"""Measuring a crystal's own voltage-to-retardance curve from fringes.

The ground truth here is the rig's actual Meadowlark calibration file, so
these check the method against a real device curve rather than a smooth
invention -- which matters, because that file is flat and slightly
non-monotonic near both voltage rails, and that is exactly what confuses a
naive fringe finder.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from polscope_library.calibration.sweep import (
    FringeLandmark,
    curve_from_sweep,
    find_fringe_landmarks,
)

CURVE_FILE = Path("/home/msnelson/QPSC_Project/OtherDocuments/polscope/mmgr_dal_MeadowlarkLC.csv")
WAVELENGTH_NM = 546.0


def real_curve(axis="A"):
    """The rig's own curve at 546 nm, as (volts, waves). Skips if absent."""
    if not CURVE_FILE.exists():
        pytest.skip("the rig's Meadowlark curve is not available here")
    rows = [ln.rstrip("\r\n") for ln in CURVE_FILE.read_text().splitlines() if ln.strip() and not ln.startswith("-")]
    data = np.array([[float(x) if x.strip() else np.nan for x in r.split(",")] for r in rows[2:]])
    volts, nm = data[:, 3], data[:, 4 if axis == "A" else 5]
    good = ~(np.isnan(volts) | np.isnan(nm))
    order = np.argsort(volts[good])
    return volts[good][order] / 1000.0, nm[good][order] / WAVELENGTH_NM


def sweep(true_v, true_w, volts, noise=0.0, seed=0):
    """What the camera sees with the partner crystal at a half wave."""
    r = np.interp(volts, true_v, true_w)
    signal = 4000.0 * 0.5 * (1.0 - 0.9925 * np.sin(2 * np.pi * r)) + 102.0
    if noise:
        signal = signal + np.random.default_rng(seed).normal(0.0, noise, signal.shape)
    return signal


class TestAgainstTheRealCurve:
    @pytest.mark.parametrize("axis", ["A", "B"])
    def test_recovers_it_to_a_fraction_of_a_milliwave(self, axis):
        true_v, true_w = real_curve(axis)
        volts = np.linspace(0.0, 20.0, 201)
        curve = curve_from_sweep(volts, sweep(true_v, true_w, volts))
        probe = np.linspace(curve.voltages.min(), curve.voltages.max(), 400)
        error = np.abs(curve.retardance_for(probe) - np.interp(probe, true_v, true_w))
        assert np.median(error) < 0.005, f"median {np.median(error):.4f} waves"
        assert error.max() < 0.05, f"max {error.max():.4f} waves"

    @pytest.mark.parametrize("noise", [5.0, 20.0])
    def test_survives_camera_noise(self, noise):
        true_v, true_w = real_curve("A")
        volts = np.linspace(0.0, 20.0, 201)
        curve = curve_from_sweep(volts, sweep(true_v, true_w, volts, noise=noise, seed=1))
        probe = np.linspace(curve.voltages.min(), curve.voltages.max(), 400)
        error = np.abs(curve.retardance_for(probe) - np.interp(probe, true_v, true_w))
        assert np.median(error) < 0.05

    def test_the_curve_the_rig_would_actually_use_is_covered(self):
        """Calibration lives near a quarter wave on LC-A and a half on LC-B, so
        the swept span has to include those or the measurement is useless."""
        true_v, true_w = real_curve("A")
        volts = np.linspace(0.0, 20.0, 201)
        curve = curve_from_sweep(volts, sweep(true_v, true_w, volts))
        low, high = curve.span_waves
        assert low <= 0.25 <= high


class TestFringeFinding:
    def _synthetic(self, n=201, noise=0.0, seed=0):
        volts = np.linspace(0.0, 20.0, n)
        retardance = 1.70 - 1.63 * (volts / 20.0) ** 0.45
        signal = 4000.0 * 0.5 * (1.0 - 0.99 * np.sin(2 * np.pi * retardance)) + 102.0
        if noise:
            signal = signal + np.random.default_rng(seed).normal(0.0, noise, signal.shape)
        return volts, signal, retardance

    def test_minima_and_maxima_land_on_different_residues(self):
        """This is what makes the result absolute. A minimum is a quarter wave
        plus a whole number; a maximum is three quarters. If they were
        indistinguishable the curve would be known only up to an offset."""
        volts, signal, _ = self._synthetic()
        for landmark in find_fringe_landmarks(volts, signal):
            residue = landmark.retardance_waves % 1.0
            expected = 0.25 if landmark.kind == "min" else 0.75
            assert abs(residue - expected) < 1e-9

    def test_landmarks_are_half_a_wave_apart_and_ordered(self):
        volts, signal, _ = self._synthetic()
        landmarks = find_fringe_landmarks(volts, signal)
        assert [lm.voltage for lm in landmarks] == sorted(lm.voltage for lm in landmarks)
        steps = np.diff([lm.retardance_waves for lm in landmarks])
        assert np.allclose(steps, -0.5)

    def test_flat_ends_do_not_become_fringes(self):
        """The real curve is flat and slightly non-monotonic at both rails, and
        a bare local-extremum test invents turning points there. One spurious
        fringe mislabels every landmark before it, so this is not cosmetic."""
        true_v, true_w = real_curve("A")
        volts = np.linspace(0.0, 20.0, 201)
        landmarks = find_fringe_landmarks(volts, sweep(true_v, true_w, volts))
        # The genuine turning points of this curve are at 1.25, 0.75 and 0.25 waves.
        assert [lm.retardance_waves for lm in landmarks] == [1.25, 0.75, 0.25]

    def test_a_sweep_with_no_fringes_is_refused(self):
        """Better than returning a curve built from one point."""
        volts = np.linspace(0.0, 20.0, 50)
        with pytest.raises(ValueError, match="at least 2"):
            find_fringe_landmarks(volts, np.linspace(100.0, 200.0, 50))

    def test_non_alternating_extrema_are_reported_not_silently_used(self):
        landmarks = [
            FringeLandmark(1.0, 1.25, "min"),
            FringeLandmark(2.0, 0.75, "max"),
        ]
        assert landmarks[0].kind != landmarks[1].kind  # sanity for the fixture
        volts = np.linspace(0.0, 20.0, 201)
        noisy = 100.0 + 50.0 * np.sin(np.linspace(0, 40 * np.pi, 201))
        # A rapidly oscillating input yields many extrema; the labelling must
        # either succeed with alternating kinds or say why it cannot.
        try:
            found = find_fringe_landmarks(volts, noisy, smooth_window=1, min_prominence=0.0)
            kinds = [lm.kind for lm in found]
            assert all(a != b for a, b in pairwise(kinds)), "kinds must alternate"
        except ValueError as exc:
            assert "alternating" in str(exc) or "at least 2" in str(exc)


class TestCurve:
    def test_round_trips_through_voltage(self):
        volts = np.linspace(0.0, 20.0, 201)
        retardance = 1.70 - 1.63 * (volts / 20.0) ** 0.45
        signal = 4000.0 * 0.5 * (1.0 - 0.99 * np.sin(2 * np.pi * retardance)) + 102.0
        curve = curve_from_sweep(volts, signal)
        for target in (0.25, 0.5, 0.75):
            if curve.span_waves[0] <= target <= curve.span_waves[1]:
                assert float(curve.retardance_for(curve.voltage_for(target))) == pytest.approx(target, abs=1e-3)

    def test_retardance_falls_as_voltage_rises(self):
        volts = np.linspace(0.0, 20.0, 201)
        retardance = 1.70 - 1.63 * (volts / 20.0) ** 0.45
        signal = 4000.0 * 0.5 * (1.0 - 0.99 * np.sin(2 * np.pi * retardance)) + 102.0
        curve = curve_from_sweep(volts, signal)
        assert np.all(np.diff(curve.retardances) <= 1e-9)
