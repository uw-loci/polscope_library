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
    def test_five_state_matches_the_rigs_august_palette(self):
        """The palette in the rig's 2026-08-11 file metadata, at swing 0.03.

        Read this as a consistency check, NOT as validation against a real
        calibration. Those five pairs are exactly the textbook nominal --
        a quarter wave and a half wave, plus and minus the swing on each
        crystal, to the digit -- which is what a Meadowlark controller holds
        before anything optimises it. A calibration that actually ran lands on
        untidy numbers: the later palette from the same rig puts extinction at
        (0.248, 0.451).

        So this pins the model to the convention, and the machine-precision
        agreement with calculate_stokes_to_intensity_matrix above is what
        actually establishes correctness. See test_a_real_calibration_is_not
        _symmetric_and_we_cannot_yet_explain_it.
        """
        nominal = {(0.25, 0.50), (0.25, 0.47), (0.25, 0.53), (0.22, 0.50), (0.28, 0.50)}
        derived = {(round(float(a), 6), round(float(b), 6)) for a, b in ideal_palette(0.03, "5-State")}
        assert derived == nominal

    def test_a_real_calibration_is_not_symmetric_and_we_cannot_yet_explain_it(self):
        """An open question recorded as a test so it is not quietly forgotten.

        The rig's later palette -- extinction (0.248, 0.451), which looks like
        a genuine optimisation -- swings LC-A by exactly +/-0.030 but LC-B by
        +0.048 / -0.014, i.e. centred 0.017 waves away from its own extinction
        value. This model cannot produce that. Adding residual instrument
        birefringence does not help: up to 5 degrees it moves the extinction
        POINT (0.250/0.500 -> 0.258/0.486) while leaving the swing states
        exactly symmetric about it.

        Three optical explanations have now been tested against real data and
        all three are eliminated:
          * Residual instrument birefringence (lossless SO(3) downstream). Up
            to 5 degrees it moves the extinction POINT (0.250/0.500 ->
            0.258/0.486) and leaves the swing states at exactly +/-0.030.
          * The A/B curve mismatch -- recOrder reads the LC-A columns for BOTH
            crystals (Calibration.py:1304, 1351-1353). Checked against the
            rig's own mmgr_dal_MeadowlarkLC.csv: the curves differ by 4.8%,
            which is near-multiplicative, and scaling preserves symmetry.
            Re-reading the reported values through either curve leaves them
            asymmetric.
          * An imperfect (elliptical rather than circular) analyser. Same
            outcome: the extinction point moves, the swing stays symmetric.

        There is a structural reason all three fail. Extinction is a smooth
        minimum of intensity, so the level set I == I_elliptical around it is
        approximately an ellipse CENTRED on that minimum, and two points on it
        along a line through the centre are symmetric to first order. The
        observed offset is 0.017 waves on a 0.031-wave swing -- 55% asymmetry
        -- which would need violent anharmonicity across 0.03 waves. No smooth
        optical model produces that.

        The remaining explanation is transcription: unlike the August values,
        which came from file metadata, these were copied by hand off a
        Micro-Manager screen. Confirm by reading a real palette out of
        calibration metadata or the device properties before treating the
        asymmetry as physical.

        Meanwhile SimulatedPolScope reproduces symmetric palettes, which the
        analysis above says is the physically correct behaviour rather than a
        simplification.
        """
        ext_a, ext_b = 0.248, 0.451
        lcb_states = (0.437, 0.499)
        offsets = tuple(round(b - ext_b, 4) for b in lcb_states)
        assert offsets == (-0.014, 0.048), "palette changed; revisit the hypotheses above"
        # LC-A, by contrast, is exactly symmetric -- so this is not general drift.
        assert round(0.218 - ext_a, 4) == -0.030
        assert round(0.278 - ext_a, 4) == 0.030

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
