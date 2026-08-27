"""The simulated instrument: does it behave like the microscope it stands in for?

Everything downstream is tested against this, so if it is wrong the optimizer
tests are worthless. These pin it to the physics and to the real rig's numbers.
"""

import numpy as np
import pytest

from polscope_library.calibration.instrument import LiquidCrystalInstrument, RetardanceLimits
from polscope_library.calibration.simulator import SimulatedPolScope


def _rotation(axis, degrees):
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    r = np.eye(3)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    r[i, i] = r[j, j] = c
    r[i, j], r[j, i] = -s, s
    return r


class TestSeamConformance:
    def test_satisfies_the_protocol(self):
        assert isinstance(SimulatedPolScope(), LiquidCrystalInstrument)

    def test_commands_outside_the_travel_are_clamped_not_refused(self):
        """The contract that keeps a wandering search from aborting a run
        part-way through, losing the states already solved."""
        scope = SimulatedPolScope(limits=RetardanceLimits(0.001, 1.6))
        scope.set_retardance("LCA", 99.0)
        assert scope.get_retardance("LCA") == pytest.approx(1.6)
        scope.set_retardance("LCA", -5.0)
        assert scope.get_retardance("LCA") == pytest.approx(0.001)

    def test_readback_reflects_incomplete_travel(self):
        """get_retardance must report where the crystal IS, not where it was
        sent -- the recorded palette has to be reproducible tomorrow."""
        scope = SimulatedPolScope(hysteresis=0.1)
        scope.set_retardance("LCA", 0.25)
        scope.set_retardance("LCA", 0.35)
        assert scope.get_retardance("LCA") == pytest.approx(0.25 + 0.9 * 0.10)

    @pytest.mark.parametrize("axis", ["lca", "X", ""])
    def test_unknown_axis_is_rejected(self, axis):
        with pytest.raises(ValueError, match="unknown LC axis"):
            SimulatedPolScope().set_retardance(axis, 0.25)


class TestPhysics:
    def test_extinction_sits_where_the_optics_say(self):
        """With no residual birefringence, minimum transmission is at a quarter
        wave on LC-A and a half wave on LC-B."""
        scope = SimulatedPolScope(read_noise=0.0)
        best = scope.intensity_at(0.25, 0.50)
        for da in (-0.02, 0.0, 0.02):
            for db in (-0.02, 0.0, 0.02):
                if da or db:
                    assert scope.intensity_at(0.25 + da, 0.50 + db) > best

    def test_residual_birefringence_moves_extinction_away_from_nominal(self):
        """Which is why a calibration searches for it instead of assuming it."""
        scope = SimulatedPolScope(residual=_rotation(0, 5.0), read_noise=0.0)
        assert scope.intensity_at(0.2583, 0.4861) < scope.intensity_at(0.25, 0.50)

    def test_default_polarization_reproduces_the_rigs_extinction_ratio(self):
        """0.9925 is not a round number -- it is chosen so the simulator
        achieves the 267 measured on the bench after calibration. A test
        asserting 'ER above 200' then asserts something real hardware does."""
        scope = SimulatedPolScope(read_noise=0.0)
        swing = 0.03
        i_ext = scope.intensity_at(0.25, 0.50)
        i_ell = scope.intensity_at(0.25 - swing, 0.50)
        er = (1 / np.sin(np.pi * swing) ** 2) * (i_ell - i_ext) / (i_ext - scope.black_level) + 1
        assert 240 < er < 300, f"extinction ratio {er:.1f} is not bench-like"

    def test_perfect_polarization_gives_unbounded_extinction_ratio(self):
        """The knob does what it claims: depolarization is what limits it."""
        scope = SimulatedPolScope(polarization=1.0, read_noise=0.0)
        assert scope.intensity_at(0.25, 0.50) == pytest.approx(scope.black_level)

    def test_a_specimen_changes_what_is_measured(self):
        """Built from the package's own forward model rather than a hand-made
        matrix -- an arbitrary one can easily perturb only components the
        analyser does not read, and then prove nothing."""
        from polscope_library import mueller_from_stokes, stokes_after_adr

        sample = mueller_from_stokes(
            *stokes_after_adr(retardance=0.5, orientation=0.3, transmittance=1.0, depolarization=1.0),
            direction="forward",
        )
        clear = SimulatedPolScope(read_noise=0.0)
        seen = SimulatedPolScope(sample_mueller=sample, read_noise=0.0)
        # Compare across all five states: a birefringent specimen must change
        # at least the states that carry S1 and S2.
        differences = [
            abs(clear.intensity_at(a, b) - seen.intensity_at(a, b))
            for a, b in [(0.25, 0.50), (0.22, 0.50), (0.25, 0.53), (0.28, 0.50), (0.25, 0.47)]
        ]
        assert max(differences) > 1.0, f"specimen had no measurable effect: {differences}"


class TestMeasurement:
    def test_dark_frame_is_the_black_level(self):
        scope = SimulatedPolScope(black_level=102.0, read_noise=0.0)
        assert scope.measure_dark() == pytest.approx(102.0)

    def test_every_frame_carries_the_black_level(self):
        scope = SimulatedPolScope(read_noise=0.0)
        assert scope.intensity_at(0.25, 0.50) > scope.black_level - 1e-9

    def test_exposures_are_counted(self):
        """A calibration needing ten times the frames costs ten times the bench
        time, so the count is worth asserting on."""
        scope = SimulatedPolScope()
        scope.measure_intensity()
        scope.measure_dark()
        scope.intensity_at(0.3, 0.4)
        assert scope.exposures == 3

    def test_intensity_at_leaves_the_crystals_alone(self):
        scope = SimulatedPolScope()
        scope.set_retardance("LCA", 0.31)
        scope.set_retardance("LCB", 0.47)
        scope.intensity_at(0.1, 0.9)
        assert scope.get_retardance("LCA") == pytest.approx(0.31)
        assert scope.get_retardance("LCB") == pytest.approx(0.47)

    def test_noise_is_reproducible_from_the_seed(self):
        a = SimulatedPolScope(read_noise=5.0, rng=np.random.default_rng(7))
        b = SimulatedPolScope(read_noise=5.0, rng=np.random.default_rng(7))
        assert [a.measure_intensity() for _ in range(5)] == [b.measure_intensity() for _ in range(5)]
