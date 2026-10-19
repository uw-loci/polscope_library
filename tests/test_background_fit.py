"""The estimated background: a smooth residual fitted per tile and removed.

Synthetic tiles carry a known residual (a 1-3 nm retardance ramp, the kind the
measured background misses when the slide's own strain differs from the
reference's) plus, optionally, a specimen of known retardance. Residual and
specimen share an axis, so their retardances add exactly and the right answer
is known at every pixel.
"""

import numpy as np
import pytest

from polscope_library import (
    EstimatedBackground,
    calculate_stokes_to_intensity_matrix,
    estimate_background,
    mmul,
    reconstruct,
    stokes_after_adr,
)

SWING = 0.03
WAVELENGTH_NM = 549.0
SHAPE = (256, 320)
AXIS = 0.4  # radians, shared by residual and specimen


def nm_to_rad(nm):
    return np.asarray(nm, dtype=np.float64) * 2 * np.pi / WAVELENGTH_NM


def ramp_nm(lo=1.0, hi=3.0, shape=SHAPE, bend=0.0):
    """A smooth residual: linear across x from lo to hi nm, plus an optional bow."""
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    xs = x / (shape[1] - 1)
    ys = y / (shape[0] - 1)
    return lo + (hi - lo) * xs + bend * (ys - 0.5) ** 2


def states_for(retardance_nm, transmittance=2000.0):
    ret = nm_to_rad(retardance_nm)
    s = stokes_after_adr(ret, np.full(ret.shape, AXIS), np.full(ret.shape, transmittance), np.ones(ret.shape))
    return mmul(calculate_stokes_to_intensity_matrix(SWING), np.stack(s))


def specimen_patch(value_nm, shape=SHAPE, box=(slice(100, 140), slice(120, 170))):
    out = np.zeros(shape)
    out[box] = value_nm
    return out, box


METHODS = ("qlipp", "openpolscope")


class TestSurfaceFitPort:
    def test_recovers_a_plane_exactly(self):
        """The median of a plane over a square block is its value at the block
        centre, so a plane comes back exactly -- which also checks the block
        centres are where the fit thinks they are."""
        y, x = np.mgrid[0:200, 0:300].astype(float)
        image = 3.0 + 0.01 * x - 0.02 * y
        np.testing.assert_allclose(estimate_background(image, order=2, block_size=20), image, rtol=0, atol=1e-9)

    def test_recovers_a_quadratic_closely(self):
        """Curvature shifts a block's median off its centre value by a little,
        in upstream as here; the fit is close, not exact."""
        y, x = np.mgrid[0:200, 0:300].astype(float)
        image = 3.0 + 0.01 * x - 0.02 * y + 1e-4 * x * y + 2e-5 * x**2 - 3e-5 * y**2
        np.testing.assert_allclose(estimate_background(image, order=2, block_size=20), image, rtol=0, atol=5e-3)

    def test_block_median_ignores_a_small_bright_feature(self):
        image = np.full((128, 128), 5.0)
        image[40:44, 40:44] = 1000.0  # well under half of one 32 px block
        np.testing.assert_allclose(estimate_background(image, order=1, block_size=32), 5.0, atol=1e-9)

    def test_block_larger_than_image_is_refused(self):
        with pytest.raises(ValueError, match="larger than the image"):
            estimate_background(np.zeros((16, 16)), block_size=32)


@pytest.mark.parametrize("method", METHODS)
class TestResidualRemoval:
    def test_clear_tile_goes_to_zero(self, method):
        states = states_for(ramp_nm())
        before = reconstruct(states, SWING, WAVELENGTH_NM, method=method)
        after = reconstruct(states, SWING, WAVELENGTH_NM, method=method, estimated_background=EstimatedBackground())
        assert np.ptp(before.retardance_nm) > 1.9  # the ramp is there
        assert after.background_fit.applied
        # What is left is the noise-free fit error of a tan-space surface:
        # far below a hundredth of a nanometre.
        assert np.max(after.retardance_nm) < 0.01
        assert after.background_fit.peak_removed_nm == pytest.approx(3.0, abs=0.02)
        assert after.background_fit.mean_removed_nm == pytest.approx(2.0, abs=0.05)

    def test_specimen_keeps_its_retardance(self, method):
        patch, box = specimen_patch(20.0)
        states = states_for(ramp_nm() + patch)
        out = reconstruct(states, SWING, WAVELENGTH_NM, method=method, estimated_background=EstimatedBackground())
        assert out.background_fit.applied
        # Retarders on one axis add in retardance, but the fit subtracts in
        # tan(retardance): 20 nm on a 3 nm residual comes back ~0.15 nm high.
        np.testing.assert_allclose(out.retardance_nm[box], 20.0, atol=0.3)
        clear = np.ones(SHAPE, dtype=bool)
        clear[box] = False
        assert np.max(out.retardance_nm[clear]) < 0.05
        # The block holding the specimen was excluded, not averaged in.
        assert out.background_fit.blocks_used < out.background_fit.blocks_total

    def test_orientation_of_the_specimen_is_kept(self, method):
        patch, box = specimen_patch(20.0)
        out = reconstruct(
            states_for(ramp_nm() + patch),
            SWING,
            WAVELENGTH_NM,
            method=method,
            estimated_background=EstimatedBackground(),
        )
        reference = reconstruct(states_for(patch + 1e-9), SWING, WAVELENGTH_NM, method=method)
        np.testing.assert_allclose(out.orientation_rad[box], reference.orientation_rad[box], atol=1e-3)

    def test_not_requested_means_no_report_and_no_change(self, method):
        states = states_for(ramp_nm())
        a = reconstruct(states, SWING, WAVELENGTH_NM, method=method)
        assert a.background_fit is None
        b = reconstruct(states, SWING, WAVELENGTH_NM, method=method, estimated_background=None)
        np.testing.assert_array_equal(a.retardance_nm, b.retardance_nm)

    def test_tile_full_of_specimen_is_left_alone(self, method):
        """The guard for the documented caveat: tissue filling the field must
        not be fitted away."""
        states = states_for(ramp_nm() + 30.0)
        plain = reconstruct(states, SWING, WAVELENGTH_NM, method=method)
        out = reconstruct(states, SWING, WAVELENGTH_NM, method=method, estimated_background=EstimatedBackground())
        assert not out.background_fit.applied
        assert "mostly specimen" in out.background_fit.reason
        assert out.background_fit.blocks_used == 0
        np.testing.assert_array_equal(out.retardance_nm, plain.retardance_nm)
        np.testing.assert_array_equal(out.orientation_rad, plain.orientation_rad)

    def test_without_the_guard_the_fit_removes_specimen(self, method):
        """The caveat itself, pinned so nobody mistakes the guard for optional:
        with every block admitted, smooth specimen retardance is removed."""
        states = states_for(ramp_nm() + 30.0)
        out = reconstruct(
            states,
            SWING,
            WAVELENGTH_NM,
            method=method,
            estimated_background=EstimatedBackground(max_block_retardance_nm=None),
        )
        assert out.background_fit.applied
        assert np.median(out.retardance_nm) < 1.0  # 31-33 nm of "tissue" gone

    def test_two_tiles_agree_at_the_seam(self, method):
        """The reason the option exists: two neighbouring tiles whose residuals
        differ (here 1-3 nm and 3-5 nm with a bow) meet without a step."""
        left = reconstruct(
            states_for(ramp_nm(1.0, 3.0)),
            SWING,
            WAVELENGTH_NM,
            method=method,
            estimated_background=EstimatedBackground(),
        )
        right = reconstruct(
            states_for(ramp_nm(3.0, 5.0, bend=2.0)),
            SWING,
            WAVELENGTH_NM,
            method=method,
            estimated_background=EstimatedBackground(),
        )
        step = abs(np.mean(left.retardance_nm[:, -1]) - np.mean(right.retardance_nm[:, 0]))
        assert step < 0.02

    def test_measured_background_still_comes_first(self, method):
        """A residual the measured background already explains leaves the fit
        nothing to remove."""
        residual = ramp_nm()
        patch, _box = specimen_patch(20.0)
        out = reconstruct(
            states_for(residual + patch),
            SWING,
            WAVELENGTH_NM,
            background_intensities=states_for(residual),
            method=method,
            estimated_background=EstimatedBackground(),
        )
        assert out.background_fit.applied
        assert out.background_fit.peak_removed_nm < 0.05


class TestSettings:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"order": -1},
            {"order": 1.5},
            {"block_size": 1},
            {"max_block_retardance_nm": 0},
            {"min_background_fraction": 1.5},
        ],
    )
    def test_bad_values_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            EstimatedBackground(**kwargs)

    def test_needs_two_dimensional_states(self):
        states = states_for(ramp_nm())[:, None, :, :].repeat(2, axis=1)  # (5, z, y, x)
        with pytest.raises(ValueError, match="per 2D image"):
            reconstruct(states, SWING, WAVELENGTH_NM, estimated_background=EstimatedBackground())

    def test_image_smaller_than_a_block_is_skipped(self):
        out = reconstruct(
            states_for(ramp_nm(shape=(16, 16))),
            SWING,
            WAVELENGTH_NM,
            estimated_background=EstimatedBackground(block_size=32),
        )
        assert not out.background_fit.applied
        assert "smaller than one" in out.background_fit.reason
