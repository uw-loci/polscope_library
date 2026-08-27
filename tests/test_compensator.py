"""The universal-compensator forward model.

The point of this model is that it lets calibration be tested with no
microscope. That only works if the model is right, so it is pinned to the
system matrix already validated against real OpenPolScope output, and to a
palette read off the rig.
"""

import numpy as np
import pytest

from polscope_library import calculate_stokes_to_intensity_matrix
from polscope_library.calibration.compensator import (
    EXTINCTION_LCA_WAVES,
    EXTINCTION_LCB_WAVES,
    compensator_stokes,
    ideal_palette,
    retardances_for_stokes,
)

SCHEMES = ["4-State", "5-State"]


class TestAgainstTheVerifiedMatrix:
    """The primary correctness gate for everything calibration-related."""

    @pytest.mark.parametrize("scheme", SCHEMES)
    @pytest.mark.parametrize("swing", [0.01, 0.03, 0.05, 0.1])
    def test_ideal_palette_reproduces_the_system_matrix(self, scheme, swing):
        palette = ideal_palette(swing, scheme)
        produced = np.stack([compensator_stokes(a, b) for a, b in palette])
        assert np.allclose(produced, calculate_stokes_to_intensity_matrix(swing, scheme), atol=1e-12)

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_extinction_is_the_first_state(self, scheme):
        """Both schemes put extinction first; the rest of the package assumes it."""
        palette = ideal_palette(0.03, scheme)
        assert np.allclose(compensator_stokes(*palette[0]), [1, 0, 0, -1], atol=1e-12)


class TestPhysics:
    def test_nominal_extinction_is_circular(self):
        """A quarter wave on LC-A and a half wave on LC-B send linear to
        circular, which the circular analyser blocks."""
        s = compensator_stokes(EXTINCTION_LCA_WAVES, EXTINCTION_LCB_WAVES)
        assert np.allclose(s, [1, 0, 0, -1], atol=1e-12)

    @pytest.mark.parametrize("lca", [0.0, 0.13, 0.25, 0.4, 0.75])
    @pytest.mark.parametrize("lcb", [0.0, 0.22, 0.5, 0.9])
    def test_output_is_always_fully_polarized(self, lca, lcb):
        """The compensator is lossless -- it reorients, it does not depolarize.
        A degree of polarization below 1 would mean the model leaks."""
        _, s1, s2, s3 = compensator_stokes(lca, lcb)
        assert np.isclose(s1**2 + s2**2 + s3**2, 1.0, atol=1e-12)

    def test_s1_depends_only_on_lca(self):
        """This is what makes the palette structure what it is -- see the
        module docstring."""
        a = compensator_stokes(0.22, 0.10)
        b = compensator_stokes(0.22, 0.87)
        assert np.isclose(a[1], b[1])

    def test_periodic_in_whole_waves(self):
        assert np.allclose(compensator_stokes(0.22, 0.5), compensator_stokes(1.22, 2.5), atol=1e-12)


class TestInverse:
    @pytest.mark.parametrize("lca", [0.05, 0.22, 0.25, 0.28, 0.45])
    @pytest.mark.parametrize("lcb", [0.10, 0.47, 0.50, 0.53, 0.95])
    def test_round_trip(self, lca, lcb):
        got_a, got_b = retardances_for_stokes(compensator_stokes(lca, lcb))
        assert np.isclose(float(got_a), lca, atol=1e-9)
        assert np.isclose(float(got_b), lcb, atol=1e-9)

    def test_survives_the_sin_delta_a_zero_singularity(self):
        """At a half wave on LC-A, sin(delta_A) vanishes. Recovering delta_B by
        division would be 0/0; atan2 does not care."""
        got_a, got_b = retardances_for_stokes(compensator_stokes(0.5, 0.3))
        assert np.isclose(float(got_a), 0.5, atol=1e-9)
        assert np.isfinite(float(got_b))

    def test_accepts_a_stack(self):
        palette = ideal_palette(0.03, "5-State")
        stokes = np.stack([compensator_stokes(a, b) for a, b in palette])
        lca, lcb = retardances_for_stokes(stokes)
        assert lca.shape == (5,) and lcb.shape == (5,)
        assert np.allclose(np.stack([lca, lcb], axis=-1), palette, atol=1e-9)


class TestPaletteStructure:
    def test_five_state_matches_the_palette_read_off_the_rig(self):
        """The instrument's own August palette, at swing 0.03 -- LC-A/LC-B in
        waves, listed in OpenPolScope's acquisition order. Independent of any
        of our maths, so agreement is real evidence rather than a tautology."""
        measured = {(0.25, 0.50), (0.25, 0.47), (0.25, 0.53), (0.22, 0.50), (0.28, 0.50)}
        derived = {(round(float(a), 6), round(float(b), 6)) for a, b in ideal_palette(0.03, "5-State")}
        assert derived == measured

    def test_five_state_repeats_each_base_value_three_times(self):
        """The structural signature of a real 5-State calibration: the three
        states that swing LC-A all share the extinction LC-B, and vice versa.
        A palette lacking this did not come from a calibration."""
        palette = ideal_palette(0.03, "5-State")
        lca, lcb = np.round(palette[:, 0], 9), np.round(palette[:, 1], 9)
        assert np.sum(lca == lca[0]) == 3
        assert np.sum(lcb == lcb[0]) == 3

    def test_four_state_does_NOT_have_that_structure(self):
        """Guards against applying the 5-State rule as though it were general.
        The 4-State states are not axis-aligned, so the bases do not repeat."""
        palette = ideal_palette(0.03, "4-State")
        lca, lcb = np.round(palette[:, 0], 9), np.round(palette[:, 1], 9)
        assert np.sum(lca == lca[0]) == 1
        assert np.sum(lcb == lcb[0]) == 2

    def test_palette_is_referenced_to_the_measured_extinction(self):
        """Real extinction drifts from the nominal quarter/half wave -- the rig
        measured (0.248, 0.451) -- so the starting guess must follow it."""
        palette = ideal_palette(0.03, "5-State", lca_ext=0.248, lcb_ext=0.451)
        assert np.allclose(palette[0], [0.248, 0.451], atol=1e-12)
        # the swing offsets are preserved, only the origin moved
        nominal = ideal_palette(0.03, "5-State")
        assert np.allclose(palette - palette[0], nominal - nominal[0], atol=1e-12)
