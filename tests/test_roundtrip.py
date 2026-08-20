"""Round-trip tests: forward model -> inverse -> recover the inputs.

These are the tests that actually establish the port is correct. We build
synthetic polarization-state images from a *known* retardance and orientation
using the forward model, run the reconstruction, and check we get the known
values back. If any torch-to-numpy translation were wrong -- a transposed
matrix, a sign, a wrong wrap convention -- these would fail.
"""

import numpy as np
import pytest

from polscope_library import (
    Birefringence,
    calculate_intensity_to_stokes_matrix,
    calculate_stokes_to_intensity_matrix,
    mmul,
    radians_to_nanometers,
    reconstruct,
    stokes_after_adr,
)

SWING = 0.03
WAVELENGTH_NM = 549.0


def synth_states(retardance_rad, orientation_rad, transmittance=1000.0, depolarization=1.0, scheme="5-State"):
    """Generate polarization-state intensities from known sample parameters."""
    s = stokes_after_adr(
        retardance_rad,
        orientation_rad,
        np.broadcast_to(transmittance, np.shape(retardance_rad)).astype(float),
        np.broadcast_to(depolarization, np.shape(retardance_rad)).astype(float),
    )
    s2i = calculate_stokes_to_intensity_matrix(SWING, scheme=scheme)
    return mmul(s2i, np.stack(s))


class TestMatrices:
    def test_pinv_is_left_inverse(self):
        """I2S @ S2I must be the identity: the inverse really inverts."""
        for scheme in ("4-State", "5-State"):
            s2i = calculate_stokes_to_intensity_matrix(SWING, scheme=scheme)
            i2s = calculate_intensity_to_stokes_matrix(SWING, scheme=scheme)
            np.testing.assert_allclose(i2s @ s2i, np.eye(4), atol=1e-10)

    def test_shapes(self):
        assert calculate_stokes_to_intensity_matrix(SWING, "5-State").shape == (5, 4)
        assert calculate_stokes_to_intensity_matrix(SWING, "4-State").shape == (4, 4)
        assert calculate_intensity_to_stokes_matrix(SWING, "5-State").shape == (4, 5)

    def test_swing_is_periodic_on_integers(self):
        a = calculate_stokes_to_intensity_matrix(0.03)
        b = calculate_stokes_to_intensity_matrix(1.03)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match="not implemented"):
            calculate_stokes_to_intensity_matrix(SWING, scheme="3-State")

    def test_pair_sum_identity(self):
        """The +/- pairs sum equally -- the identity the scheme-check tool uses.

        I(+S1) + I(-S1) == I(+S2) + I(-S2), for any sample, because the sample
        terms cancel. This is what makes it possible to identify the state
        ordering empirically from acquired data.
        """
        rng = np.random.default_rng(0)
        ret = rng.uniform(0, 1.0, (16, 16))
        ori = rng.uniform(0, np.pi, (16, 16))
        states = synth_states(ret, ori)
        np.testing.assert_allclose(states[1] + states[3], states[2] + states[4], atol=1e-9)


class TestRoundTrip:
    @pytest.mark.parametrize("scheme", ["4-State", "5-State"])
    def test_recovers_known_retardance_and_orientation(self, scheme):
        # Sweep orientation across the full axial range and retardance up to
        # just under the quarter-wave ceiling the arcsine imposes.
        h = w = 64
        yy, xx = np.mgrid[0:h, 0:w]
        orientation = (xx / w) * np.pi * 0.999
        retardance = 0.05 + (yy / h) * 1.4  # radians, stays below pi/2

        states = synth_states(retardance, orientation, scheme=scheme)
        out = reconstruct(states, swing=SWING, wavelength_nm=WAVELENGTH_NM, scheme=scheme)

        np.testing.assert_allclose(
            out.retardance_nm,
            radians_to_nanometers(retardance, WAVELENGTH_NM),
            atol=1e-6,
            err_msg="retardance not recovered",
        )
        # Orientation is axial, so compare modulo pi via the doubled angle.
        np.testing.assert_allclose(np.sin(2 * out.orientation_rad), np.sin(2 * orientation), atol=1e-6)
        np.testing.assert_allclose(np.cos(2 * out.orientation_rad), np.cos(2 * orientation), atol=1e-6)

    def test_recovers_transmittance_and_depolarization(self):
        ret = np.full((8, 8), 0.4)
        ori = np.full((8, 8), 0.7)
        states = synth_states(ret, ori, transmittance=1234.0, depolarization=0.8)
        out = reconstruct(states, swing=SWING, wavelength_nm=WAVELENGTH_NM)
        np.testing.assert_allclose(out.transmittance, 1234.0, rtol=1e-9)
        np.testing.assert_allclose(out.depolarization, 0.8, rtol=1e-9)

    def test_state_order_matters(self):
        """Permuting states must change the answer -- this is why order is checked.

        Swapping the +S1/-S1 pair with the +S2/-S2 pair is exactly the
        waveorder-vs-Oldenbourg ordering difference. It raises no error; it
        silently rotates the orientation. This test documents that.
        """
        ret = np.full((8, 8), 0.5)
        ori = np.full((8, 8), 0.3)
        states = synth_states(ret, ori)
        correct = reconstruct(states, swing=SWING, wavelength_nm=WAVELENGTH_NM)
        permuted = reconstruct(states[[0, 2, 1, 4, 3]], swing=SWING, wavelength_nm=WAVELENGTH_NM)

        # Retardance survives the permutation; orientation does not.
        np.testing.assert_allclose(permuted.retardance_nm, correct.retardance_nm, atol=1e-6)
        assert not np.allclose(
            np.sin(2 * permuted.orientation_rad), np.sin(2 * correct.orientation_rad), atol=1e-3
        ), "a permuted state order should change orientation; if it does not, the test is not exercising order"

    def test_wrong_state_count_is_an_error_not_garbage(self):
        ret = np.full((4, 4), 0.5)
        states = synth_states(ret, np.full((4, 4), 0.3))
        with pytest.raises(ValueError, match="expects 4 states"):
            reconstruct(states, swing=SWING, wavelength_nm=WAVELENGTH_NM, scheme="4-State")


class TestBackgroundCorrection:
    def test_background_removes_instrument_birefringence(self):
        """A background carrying instrument birefringence must be divided out.

        Model the instrument as its own retarder in series: reconstructing
        without the background sees instrument + sample, and with it should
        recover the sample alone.
        """
        h = w = 32
        yy, xx = np.mgrid[0:h, 0:w]
        sample_ret = 0.05 + (yy / h) * 0.5
        sample_ori = (xx / w) * np.pi * 0.999

        # Instrument residual: small, uniform, at a fixed orientation.
        inst_ret, inst_ori = 0.06, 0.9

        bg_states = synth_states(np.full((h, w), inst_ret), np.full((h, w), inst_ori))
        # Sample data carries both. Composing two retarders exactly requires
        # Mueller algebra; for a small instrument retardance the Stokes-level
        # superposition below is close enough to show the correction works.
        both = synth_states(sample_ret + inst_ret * np.cos(2 * (sample_ori - inst_ori)), sample_ori)

        uncorrected = reconstruct(both, swing=SWING, wavelength_nm=WAVELENGTH_NM)
        corrected = reconstruct(
            both, swing=SWING, wavelength_nm=WAVELENGTH_NM, background_intensities=bg_states, denom_floor=1e-12
        )
        truth = radians_to_nanometers(sample_ret, WAVELENGTH_NM)

        err_uncorrected = np.mean(np.abs(uncorrected.retardance_nm - truth))
        err_corrected = np.mean(np.abs(corrected.retardance_nm - truth))
        assert (
            err_corrected < err_uncorrected
        ), f"background correction made it worse: {err_corrected:.3f} nm vs {err_uncorrected:.3f} nm"

    def test_denom_floor_prevents_nan_on_circular_background(self):
        """A perfectly circular background is the degenerate case: s1=s2=0.

        Without a floor the forward Mueller divides by zero. This is not
        hypothetical -- a well-calibrated background is close to circular by
        construction.
        """
        h = w = 8
        ret = np.full((h, w), 0.4)
        ori = np.full((h, w), 0.3)
        states = synth_states(ret, ori)
        perfect_bg = synth_states(np.zeros((h, w)), np.zeros((h, w)))  # zero retardance -> s1=s2=0

        floored = reconstruct(
            states, swing=SWING, wavelength_nm=WAVELENGTH_NM, background_intensities=perfect_bg, denom_floor=1e-12
        )
        assert np.all(np.isfinite(floored.retardance_nm)), "denom_floor did not prevent NaN"


class TestOrientationEncoding:
    def test_sin_cos_encoding_averages_correctly_across_the_wrap(self):
        """The reason orientation is exposed as sin/cos and not as an angle.

        Two nearly-horizontal fibres reading 179 and 1 degrees are 2 degrees
        apart. Averaging the raw angles gives 90 degrees -- perpendicular.
        Averaging the doubled-angle vector gives the right answer.
        """
        a, b = np.deg2rad(179.0), np.deg2rad(1.0)
        res = Birefringence(
            retardance_nm=np.array([0.0, 0.0]),
            orientation_rad=np.array([a, b]),
            transmittance=np.array([1.0, 1.0]),
            depolarization=np.array([1.0, 1.0]),
        )
        naive = np.rad2deg(np.mean([a, b]))
        assert 89 < naive < 91, "the naive average should be the wrong ~90 deg this guards against"

        vector_mean = np.rad2deg(0.5 * np.arctan2(np.mean(res.orientation_sin2), np.mean(res.orientation_cos2))) % 180
        assert vector_mean < 1.0 or vector_mean > 179.0, f"circular mean should be near 0/180, got {vector_mean}"
