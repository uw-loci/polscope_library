"""Does the calibration actually find the states it is supposed to?

This is the test recOrder cannot have: with a forward model and a simulated
instrument, "the optimizer works" becomes a claim that either passes or fails
in milliseconds, instead of something you find out at the bench.
"""

import numpy as np
import pytest

from polscope_library.calibration.compensator import compensator_stokes, ideal_palette
from polscope_library.calibration.simulator import SimulatedPolScope
from polscope_library.calibration.states import SCHEMES, moves_for
from polscope_library.calibration.strategies import STRATEGIES, IterativeRefineSearch, SinglePassSearch
from polscope_library.calibration.workflow import (
    CalibrationSettings,
    assess,
    calibrate,
    extinction_ratio,
    resolve_black_level,
)

STRATEGY_NAMES = list(STRATEGIES)


def rotation(axis, degrees):
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    r = np.eye(3)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    r[i, i] = r[j, j] = c
    r[i, j], r[j, i] = -s, s
    return r


def palette_of(result):
    return np.array([result.palette[c] for c in result.palette])


class TestConvergence:
    @pytest.mark.parametrize("scheme", SCHEMES)
    @pytest.mark.parametrize("strategy", STRATEGY_NAMES)
    def test_finds_the_palette_the_system_matrix_asks_for(self, scheme, strategy):
        """On a perfect instrument, exactly. Both schemes, both searches."""
        result = calibrate(
            SimulatedPolScope(read_noise=0.0),
            CalibrationSettings(swing=0.03, scheme=scheme),
            strategy=strategy,
        )
        assert np.abs(palette_of(result) - ideal_palette(0.03, scheme)).max() < 1e-4

    @pytest.mark.parametrize("strategy", STRATEGY_NAMES)
    def test_finds_extinction_where_the_optics_moved_it(self, strategy):
        """Residual birefringence shifts extinction off the nominal quarter and
        half wave. Finding it is the entire reason calibration is a search."""
        scope = SimulatedPolScope(read_noise=0.0, residual=rotation(0, 5.0) @ rotation(1, 3.0))
        result = calibrate(scope, CalibrationSettings(swing=0.03), strategy=strategy)
        lca_ext, lcb_ext = result.palette["State0"]
        assert not np.isclose(lca_ext, 0.25, atol=1e-3), "extinction did not move; test is not testing anything"
        # Every other state must sit correctly about wherever extinction landed.
        want = ideal_palette(0.03, "5-State", lca_ext, lcb_ext)
        assert np.abs(palette_of(result) - want).max() < 1e-3

    def test_survives_noise_and_incomplete_settling(self):
        scope = SimulatedPolScope(
            read_noise=2.0, hysteresis=0.02, residual=rotation(0, 2.0), rng=np.random.default_rng(3)
        )
        result = calibrate(scope, CalibrationSettings(swing=0.03))
        lca_ext, lcb_ext = result.palette["State0"]
        want = ideal_palette(0.03, "5-State", lca_ext, lcb_ext)
        # Looser: with noise the states are found to a fraction of the swing.
        assert np.abs(palette_of(result) - want).max() < 0.02

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_both_strategies_agree(self, scheme):
        kwargs = dict(read_noise=0.0, residual=rotation(1, 3.0))
        a = calibrate(SimulatedPolScope(**kwargs), CalibrationSettings(scheme=scheme), strategy="single_pass")
        b = calibrate(SimulatedPolScope(**kwargs), CalibrationSettings(scheme=scheme), strategy="iterative")
        assert np.abs(palette_of(a) - palette_of(b)).max() < 1e-3

    def test_iterating_costs_more_exposures(self):
        """Bench time is the currency. If refinement were free there would be
        no choice to expose to the operator."""
        single = calibrate(SimulatedPolScope(read_noise=0.0), strategy="single_pass")
        iterative = calibrate(SimulatedPolScope(read_noise=0.0), strategy="iterative")
        assert iterative.exposures > single.exposures


class TestTheMirrorSolution:
    """Each state has an antipodal twin at exactly the same intensity.

    Intensity matching cannot tell them apart, so the search window must never
    reach the twin. This is a real ambiguity in the physics, not a defect --
    but converging onto the wrong branch reflects the orientation map, with
    nothing to indicate it happened.
    """

    def test_the_twin_is_genuinely_indistinguishable(self):
        """Establishes the hazard is real before testing the guard."""
        scope = SimulatedPolScope(read_noise=0.0)
        slope = 1.7449  # the 4-State constraint slope
        near = scope.intensity_at(0.25 - 0.0149, 0.50 - slope * 0.0149)
        far = scope.intensity_at(0.25 + 0.0149, 0.50 + slope * 0.0149)
        assert np.isclose(near, far, atol=1e-6)
        # ...and they are opposite states, not the same one.
        s_near = compensator_stokes(0.25 - 0.0149, 0.50 - slope * 0.0149)
        s_far = compensator_stokes(0.25 + 0.0149, 0.50 + slope * 0.0149)
        assert np.sign(s_near[1]) == -np.sign(s_far[1])

    def test_a_wide_window_does_pick_the_twin(self):
        """Drives the search directly, because calibrate() derives the bound
        and will not let it be widened -- which is the point, but means the
        hazard has to be demonstrated one level down. Without the derived
        bound, the search converges onto the mirror state."""
        from polscope_library.calibration.session import CalibrationSession

        scope = SimulatedPolScope(read_noise=0.0)
        session = CalibrationSession(scope, swing=0.03)
        session.lca_ext, session.lcb_ext = 0.25, 0.50
        i_elliptical = scope.intensity_at(0.22, 0.50)

        move = [m for m in moves_for("5-State") if m.name == "I45"][0]
        seed = tuple(ideal_palette(0.03, "5-State")[2])  # (0.25, 0.53)

        narrow = SinglePassSearch().solve(session, move, seed=seed, bound=0.03, reference=i_elliptical)
        wide = SinglePassSearch().solve(session, move, seed=seed, bound=0.5, reference=i_elliptical)

        assert narrow.lcb == pytest.approx(0.53, abs=1e-3), "the derived bound should hold the correct branch"
        assert abs(wide.lcb - 0.53) > 0.01, "a wide window should reach the twin -- if not, the hazard is gone"
        # Both are equally good by the cost function, which is exactly the problem.
        assert abs(narrow.residual) == pytest.approx(abs(wide.residual), abs=1e-3)


class TestSchemeTable:
    def test_unknown_scheme_says_so(self):
        """recOrder falls through its if-chains and raises UnboundLocalError
        from inside the optimizer, which names nothing."""
        with pytest.raises(ValueError, match="unknown calibration scheme"):
            moves_for("6-State")

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_extinction_is_first_and_channels_are_sequential(self, scheme):
        moves = moves_for(scheme)
        assert moves[0].kind == "extinction"
        assert [m.channel for m in moves] == [f"State{i}" for i in range(len(moves))]

    def test_both_constrained_states_vary_the_same_crystal(self):
        """The defect that motivated the table: recOrder's default optimizer
        passes LC-B for I120 and LC-A for I60, so LC-A never moves during the
        120 degree search and an LC-B value is recorded as LC-A."""
        constrained = [m for m in moves_for("4-State") if m.kind == "constrained"]
        assert len(constrained) == 2
        assert {m.axis for m in constrained} == {"LCA"}
        assert {m.constraint_sign for m in constrained} == {+1, -1}

    def test_exactly_one_open_loop_state_defines_the_reference(self):
        for scheme in SCHEMES:
            kinds = [m.kind for m in moves_for(scheme)]
            assert kinds.count("open_loop") == 1


class TestBlackLevel:
    def test_configured_value_wins(self):
        level, source = resolve_black_level(SimulatedPolScope(), CalibrationSettings(black_level=102.0), [])
        assert (level, source) == (102.0, "configured")

    def test_otherwise_measured_from_a_dark_frame(self):
        scope = SimulatedPolScope(black_level=137.0, read_noise=0.0)
        level, source = resolve_black_level(scope, CalibrationSettings(dark_frames=5), [])
        assert source == "measured"
        assert level == pytest.approx(137.0)

    def test_falls_through_when_the_field_cannot_be_darkened(self):
        """Never a shutter, never a prompt -- recOrder blocks on stdin here,
        from a worker thread, on a rig that has no shutter."""

        class NoDark(SimulatedPolScope):
            def measure_dark(self):
                raise NotImplementedError

        warnings = []
        level, source = resolve_black_level(NoDark(), CalibrationSettings(), warnings)
        assert (level, source) == (0.0, "assumed")
        assert warnings and "overstated" in warnings[0]

    def test_source_is_recorded_in_the_result(self):
        result = calibrate(SimulatedPolScope(read_noise=0.0), CalibrationSettings(black_level=102.0))
        assert result.black_level_source == "configured"


class TestExtinctionRatio:
    def test_reproduces_the_bench_number(self):
        """From the 2026-08-25 reference slide: extinction 1268.03, elliptical
        3981.32, swing 0.03, black 102. OpenPolScope reported 267.54 after
        calibration; 263.74 here, the gap being that the bench figure is
        measured on a blank field rather than on the target."""
        assert extinction_ratio(0.03, 102.0, 1268.03, 3981.32) == pytest.approx(263.74, abs=0.01)

    @pytest.mark.parametrize(
        "ratio,expected", [(300, "good"), (100, "good"), (99, "acceptable"), (80, "acceptable"), (79, "poor")]
    )
    def test_bands(self, ratio, expected):
        assert assess(ratio) == expected

    def test_a_poor_result_is_recorded_and_warned_about_not_refused(self):
        """A marginal calibration the operator can see beats one thrown away."""
        scope = SimulatedPolScope(polarization=0.5, read_noise=0.0)
        result = calibrate(scope, CalibrationSettings(swing=0.03))
        assert result.assessment == "poor"
        assert result.palette, "a poor calibration must still be returned"
        assert any("below 80" in w for w in result.warnings)

    def test_unmeasurable_when_extinction_sits_at_the_black_level(self):
        assert np.isnan(extinction_ratio(0.03, 100.0, 100.0, 200.0))
        assert assess(float("nan")) == "unmeasurable"


class TestResult:
    def test_trace_records_every_exposure(self):
        result = calibrate(SimulatedPolScope(read_noise=0.0))
        assert len(result.trace) == result.exposures - 20  # 20 dark frames
        assert {p.state for p in result.trace} == {m.name for m in moves_for("5-State")}

    def test_states_are_reported_in_acquisition_order(self):
        result = calibrate(SimulatedPolScope(read_noise=0.0))
        assert result.states == ["State0", "State1", "State2", "State3", "State4"]

    def test_strategy_may_be_named_or_constructed(self):
        by_name = calibrate(SimulatedPolScope(read_noise=0.0), strategy="iterative")
        by_object = calibrate(SimulatedPolScope(read_noise=0.0), strategy=IterativeRefineSearch())
        assert np.allclose(palette_of(by_name), palette_of(by_object))
        assert isinstance(SinglePassSearch(), SinglePassSearch)
