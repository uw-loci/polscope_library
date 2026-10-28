"""Which crystal to park where for a fringe sweep, checked against the compensator physics.

The analysis in ``sweep.py`` assumes ``I ~ 1 - sin(2 pi r)``. That holds for LC-A swept with LC-B
at a half wave, but an LC-B sweep with LC-A at a half wave is flat; LC-B needs LC-A at a quarter
wave and its result shifted by a quarter wave. These tests drive the simulated polariscope (the
library's own Stokes model, not a restated formula) through a smooth made-up crystal curve.
"""

import numpy as np
import pytest

from polscope_library.calibration import CalibrationSettings, SimulatedPolScope, calibrate
from polscope_library.calibration.sweep import SWEEP_GEOMETRY, curve_from_sweep


def crystal(volts):
    """A smooth, monotone voltage-to-retardance curve, 1.6 waves at 0 V down to 0.07 at 20 V."""
    return 0.07 + 1.53 / (1.0 + (np.asarray(volts) / 2.2) ** 1.6)


def simulated_sweep(axis, partner_waves, volts):
    scope = SimulatedPolScope(read_noise=0.0)
    geometry = SWEEP_GEOMETRY[axis]
    scope.set_retardance(geometry.partner, partner_waves)
    out = []
    for v in volts:
        scope.set_retardance(axis, float(crystal(v)))
        out.append(scope.measure_intensity())
    return np.array(out)


VOLTS = np.linspace(0.0, 20.0, 801)


@pytest.mark.parametrize("axis", ["LCA", "LCB"])
def test_each_crystal_is_recovered_with_its_own_geometry(axis):
    g = SWEEP_GEOMETRY[axis]
    curve = curve_from_sweep(VOLTS, simulated_sweep(axis, g.partner_waves, VOLTS), offset_waves=g.offset_waves)
    inside = (VOLTS > curve.voltages.min()) & (VOLTS < curve.voltages.max())
    error = np.abs(curve.retardance_for(VOLTS[inside]) - crystal(VOLTS[inside]))
    assert error.max() < 0.01, f"{axis}: worst error {error.max():.4f} waves"


def test_lcb_with_lca_at_a_half_wave_has_no_fringes():
    flat = simulated_sweep("LCB", 0.5, VOLTS)
    assert flat.max() - flat.min() < 0.01 * flat.max()


def test_lcb_without_the_offset_reads_a_quarter_wave_low():
    g = SWEEP_GEOMETRY["LCB"]
    curve = curve_from_sweep(VOLTS, simulated_sweep("LCB", g.partner_waves, VOLTS))
    v = 2.0  # inside the span between the first and last fringe
    assert curve.retardance_for(v) == pytest.approx(crystal(v) - 0.25, abs=0.01)


def test_offset_moves_the_landmarks_too():
    g = SWEEP_GEOMETRY["LCB"]
    curve = curve_from_sweep(VOLTS, simulated_sweep("LCB", 0.25, VOLTS), offset_waves=g.offset_waves)
    for lm in curve.landmarks:
        expected = 0.5 if lm.kind == "min" else 0.0
        assert lm.retardance_waves % 1.0 == pytest.approx(expected, abs=1e-9)


def test_an_imperfect_park_still_recovers_the_curve():
    for axis in ("LCA", "LCB"):
        g = SWEEP_GEOMETRY[axis]
        curve = curve_from_sweep(
            VOLTS, simulated_sweep(axis, g.partner_waves + 0.04, VOLTS), offset_waves=g.offset_waves
        )
        assert curve.retardance_for(2.0) == pytest.approx(crystal(2.0), abs=0.01)


class _ScopeWithVolts(SimulatedPolScope):
    def get_voltage(self, axis):
        return 1.0 + (self._lca if axis == "LCA" else self._lcb)


def test_trace_records_volts_when_the_instrument_offers_them():
    result = calibrate(_ScopeWithVolts(read_noise=0.0), CalibrationSettings(swing_waves=0.03))
    assert result.trace
    for p in result.trace:
        assert p.lca_volts == pytest.approx(1.0 + p.lca_waves)
        assert p.lcb_volts == pytest.approx(1.0 + p.lcb_waves)


def test_trace_volts_are_none_without_get_voltage():
    result = calibrate(SimulatedPolScope(read_noise=0.0), CalibrationSettings(swing_waves=0.03))
    assert all(p.lca_volts is None and p.lcb_volts is None for p in result.trace)


@pytest.mark.parametrize("axis, waves", [("LCA", 0.22), ("LCA", 0.28), ("LCB", 0.415), ("LCB", 0.44), ("LCB", 0.455)])
def test_working_points_below_the_last_fringe_are_measured(axis, waves):
    """Both crystals work below their lowest fringe (0.25 for LC-A, 0.5 for LC-B)."""
    g = SWEEP_GEOMETRY[axis]
    curve = curve_from_sweep(VOLTS, simulated_sweep(axis, g.partner_waves, VOLTS), offset_waves=g.offset_waves)
    grid = np.linspace(0, 20, 200001)
    volts = float(np.interp(waves, crystal(grid)[::-1], grid[::-1]))
    assert curve.retardance_for(volts) == pytest.approx(waves, abs=0.005)
