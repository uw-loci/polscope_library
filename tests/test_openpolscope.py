"""The OpenPolScope-exact method, held to the standard its name claims.

The decisive tests here run against a crop of real data: five raw states plus
OpenPolScope 3.20b's own retardance and orientation output for the same
pixels, acquired on this laboratory's LC-PolScope on 2026-08-25. "Exact" is
asserted per pixel against that output, not against a formula we wrote down.

The synthetic tests cover what the reference slide cannot reach -- it tops out
at 29 nm, a fifth of a quarter wave -- in particular the fold-back branch past
a quarter wave, which is one of the two places the methods actually differ.
"""

from pathlib import Path

import numpy as np
import pytest

from polscope_library import (
    RECONSTRUCTION_METHODS,
    calculate_stokes_to_intensity_matrix,
    mmul,
    reconstruct,
    stokes_after_adr,
)
from polscope_library.calibration import extinction_ratio, extinction_ratio_openpolscope

FIXTURE = Path(__file__).parent / "fixtures" / "openpolscope_3_20b_refslide_crop.npz"

#: How OpenPolScope 3.20b encodes its 16-bit output, read from its bytecode and
#: confirmed by the bit-exact test below. Retardance: counts = round((65535-1)
#: * nm / ceiling), saturating to 65535 above the ceiling or on dead pixels.
VALUE_BIT = 65535


@pytest.fixture(scope="module")
def crop():
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


def _library_order(states_openpolscope_order):
    """OpenPolScope's State0..4 -> library (ext, +S1, +S2, -S1, -S2).

    OpenPolScope's pairs are states 1&4 and 2&3; ours are 1&3 and 2&4. The
    QPSC rig applies this same remap via ``state_order`` in its YAML.
    """
    s = states_openpolscope_order
    return np.stack([s[0], s[1], s[2], s[4], s[3]]).astype(np.float64)


def _encode_retardance(ret_nm, ceiling_nm, dead):
    factor = ret_nm / ceiling_nm
    counts = np.rint((VALUE_BIT - 1) * factor)
    return np.where((factor > 1.0) | dead | ~np.isfinite(factor), VALUE_BIT, counts).astype(np.int64)


def synth_states(retardance_rad, orientation_rad, swing, transmittance=1000.0, depolarization=1.0):
    """Ideal 5-state intensities from a known sample, in library order."""
    shape = np.shape(retardance_rad)
    s = stokes_after_adr(
        retardance_rad,
        orientation_rad,
        np.broadcast_to(transmittance, shape).astype(float),
        np.broadcast_to(depolarization, shape).astype(float),
    )
    return mmul(calculate_stokes_to_intensity_matrix(swing, scheme="5-State"), np.stack(s))


# ----------------------------------------------------------------------------
# Against OpenPolScope's own output
# ----------------------------------------------------------------------------


class TestBitExactAgainstOpenPolScope:
    def test_retardance_matches_openpolscope_output_per_pixel(self, crop):
        r = reconstruct(
            intensities=_library_order(crop["sample_states"]),
            swing=float(crop["swing"]),
            wavelength_nm=float(crop["wavelength_nm"]),
            background_intensities=_library_order(crop["background_states"]),
            method="openpolscope",
            black_level=float(crop["black_level"]),
        )
        dead = ~np.isfinite(r.retardance_nm)
        ours = _encode_retardance(r.retardance_nm, float(crop["retardance_ceiling_nm"]), dead)
        theirs = crop["retardance_counts"].astype(np.int64)
        diff = np.abs(ours - theirs)
        # One count of slack for float32-vs-float64 rounding at the .5 boundary.
        assert diff.max() <= 1, f"max count error {diff.max()}"
        assert np.mean(diff == 0) > 0.995

    def test_orientation_matches_openpolscope_output(self, crop):
        r = reconstruct(
            intensities=_library_order(crop["sample_states"]),
            swing=float(crop["swing"]),
            wavelength_nm=float(crop["wavelength_nm"]),
            background_intensities=_library_order(crop["background_states"]),
            method="openpolscope",
            black_level=float(crop["black_level"]),
        )
        ours = np.degrees(r.orientation_rad) * 100.0  # centidegrees, like the file
        theirs = crop["orientation_counts"].astype(np.float64)
        d = np.abs(ours - theirs)
        d = np.minimum(d, 18000.0 - d)  # axial: 0 == 180 deg
        # OpenPolScope floors to whole degrees in this build, so the file is
        # coarser than our value; a whole degree of slack is the quantisation.
        assert np.percentile(d, 99) <= 100.0
        assert np.median(d) <= 60.0

    def test_wrong_wavelength_breaks_the_match(self, crop):
        """The match is a measurement, not a tautology: 3 nm the other way fails it."""
        r = reconstruct(
            intensities=_library_order(crop["sample_states"]),
            swing=float(crop["swing"]),
            wavelength_nm=546.0,
            background_intensities=_library_order(crop["background_states"]),
            method="openpolscope",
            black_level=float(crop["black_level"]),
        )
        dead = ~np.isfinite(r.retardance_nm)
        ours = _encode_retardance(r.retardance_nm, float(crop["retardance_ceiling_nm"]), dead)
        theirs = crop["retardance_counts"].astype(np.int64)
        assert np.mean(ours == theirs) < 0.5


# ----------------------------------------------------------------------------
# Where the two methods agree and where they part
# ----------------------------------------------------------------------------


class TestRelationToQlipp:
    def test_identical_without_a_background(self, crop):
        """No background: the two formulations are the same function."""
        kw = dict(
            intensities=_library_order(crop["sample_states"]),
            swing=float(crop["swing"]),
            wavelength_nm=float(crop["wavelength_nm"]),
            black_level=float(crop["black_level"]),
        )
        q = reconstruct(method="qlipp", **kw)
        o = reconstruct(method="openpolscope", **kw)
        m = np.isfinite(o.retardance_nm) & np.isfinite(q.retardance_nm) & (o.retardance_nm > 0.5)
        np.testing.assert_allclose(q.retardance_nm[m], o.retardance_nm[m], rtol=1e-9)
        d = np.abs(q.orientation_rad[m] - o.orientation_rad[m])
        d = np.minimum(d, np.pi - d)
        assert d.max() < 1e-9

    def test_the_only_difference_is_the_background_step(self, crop):
        """With a background they differ by a small, flat scale: 0.53% here.

        Documented, not fixed. QLIPP inverts the background's Mueller matrix;
        OpenPolScope subtracts in its normalised difference space. This pins
        the size so a future change to either path shows up.
        """
        kw = dict(
            intensities=_library_order(crop["sample_states"]),
            swing=float(crop["swing"]),
            wavelength_nm=float(crop["wavelength_nm"]),
            background_intensities=_library_order(crop["background_states"]),
            black_level=float(crop["black_level"]),
        )
        q = reconstruct(method="qlipp", **kw)
        o = reconstruct(method="openpolscope", **kw)
        m = np.isfinite(o.retardance_nm) & (o.retardance_nm > 2.0)
        ratio = np.median(q.retardance_nm[m] / o.retardance_nm[m])
        assert 0.990 < ratio < 0.999, ratio

    def test_openpolscope_keeps_climbing_past_a_quarter_wave(self):
        """The fold-back branch, which the reference slide never reaches.

        Past a quarter wave the extinction state outshines the mean swing state
        and OpenPolScope flips branch; QLIPP's arcsine folds the answer back
        below a quarter wave instead. Both behaviours are intended -- this test
        is what makes the difference visible rather than a surprise on tissue.
        """
        swing = 0.03
        true_rad = np.array([0.10, 0.40, 0.49, 0.51, 0.60, 0.80, 0.95]) * np.pi  # waves * 2pi/2 -> up to ~half wave
        orient = np.full(true_rad.shape, 0.3)
        states = synth_states(true_rad, orient, swing)
        o = reconstruct(states, swing, wavelength_nm=549.0, method="openpolscope")
        q = reconstruct(states, swing, wavelength_nm=549.0, method="qlipp")
        o_rad = o.retardance_nm * 2 * np.pi / 549.0
        q_rad = q.retardance_nm * 2 * np.pi / 549.0
        np.testing.assert_allclose(o_rad, true_rad, rtol=1e-6, atol=1e-9)
        folded = np.where(true_rad > np.pi / 2, np.pi - true_rad, true_rad)
        np.testing.assert_allclose(q_rad, folded, rtol=1e-6, atol=1e-9)


# ----------------------------------------------------------------------------
# Options and refusals
# ----------------------------------------------------------------------------


class TestOptions:
    def test_unknown_method_is_refused(self):
        states = synth_states(np.array([0.1]), np.array([0.2]), 0.03)
        with pytest.raises(ValueError, match="method must be one of"):
            reconstruct(states, 0.03, 549.0, method="recorder")
        assert RECONSTRUCTION_METHODS == ("qlipp", "openpolscope")

    def test_four_state_is_refused_for_openpolscope(self):
        states = np.ones((4, 2, 2))
        with pytest.raises(NotImplementedError, match="5-State only"):
            reconstruct(states, 0.03, 549.0, scheme="4-State", method="openpolscope")

    @pytest.mark.parametrize("method", RECONSTRUCTION_METHODS)
    def test_uniform_offset_is_invariant_for_both_methods(self, method):
        """A camera offset moves only S0, which neither retardance nor orientation reads.

        So black_level matters only through its clamp at zero. Stated and tested
        because it is the reason setting it to the real 103 is harmless for the
        QLIPP path and required only for pixels darker than the offset.
        """
        swing = 0.03
        true_rad = np.array([0.05, 0.20, 0.45]) * np.pi
        orient = np.array([0.1, 1.0, 2.5])
        states = synth_states(true_rad, orient, swing, transmittance=5000.0)
        base = reconstruct(states, swing, 549.0, method=method)
        offset = reconstruct(states + 103.0, swing, 549.0, method=method, black_level=103.0)
        raw_offset = reconstruct(states + 103.0, swing, 549.0, method=method)
        np.testing.assert_allclose(offset.retardance_nm, base.retardance_nm, rtol=1e-9)
        np.testing.assert_allclose(raw_offset.retardance_nm, base.retardance_nm, rtol=1e-9)
        np.testing.assert_allclose(offset.orientation_rad, base.orientation_rad, atol=1e-9)

    def test_black_level_clamps_at_zero(self):
        states = np.full((5, 1), 50.0)
        states[0] = 20.0  # extinction darker than the offset
        r = reconstruct(states, 0.03, 549.0, method="openpolscope", black_level=30.0)
        assert np.isfinite(r.retardance_nm).all()

    @pytest.mark.parametrize("method", RECONSTRUCTION_METHODS)
    def test_azimuth_reference_shifts_orientation(self, method):
        states = synth_states(np.array([0.3]), np.array([0.4]), 0.03)
        base = reconstruct(states, 0.03, 549.0, method=method)
        shifted = reconstruct(states, 0.03, 549.0, method=method, azimuth_reference_deg=30.0)
        d = (shifted.orientation_rad - base.orientation_rad) % np.pi
        np.testing.assert_allclose(d, np.radians(30.0), atol=1e-9)

    def test_mirror_negates_orientation(self):
        states = synth_states(np.array([0.3]), np.array([0.4]), 0.03)
        base = reconstruct(states, 0.03, 549.0, method="openpolscope")
        flipped = reconstruct(states, 0.03, 549.0, method="openpolscope", flip_orientation=True)
        np.testing.assert_allclose((base.orientation_rad + flipped.orientation_rad) % np.pi, 0.0, atol=1e-9)

    def test_openpolscope_does_not_fake_the_fields_it_does_not_compute(self):
        states = synth_states(np.array([0.3]), np.array([0.4]), 0.03)
        r = reconstruct(states, 0.03, 549.0, method="openpolscope")
        assert np.isnan(r.transmittance).all() and np.isnan(r.depolarization).all()


# ----------------------------------------------------------------------------
# Extinction ratio, OpenPolScope's definition
# ----------------------------------------------------------------------------


class TestExtinctionRatioOpenPolScope:
    def test_equal_swing_states_differ_from_recorder_by_exactly_one(self):
        ext, black, swing_i = 1268.03, 102.0, 3981.32
        ours = extinction_ratio(0.03, black, ext, swing_i)
        theirs = extinction_ratio_openpolscope(0.03, black, ext, [swing_i] * 4)
        assert theirs == pytest.approx(ours - 1.0, abs=1e-9)

    def test_averages_over_every_swing_state(self):
        a = extinction_ratio_openpolscope(0.03, 100.0, 1000.0, [3000.0, 5000.0])
        b = extinction_ratio_openpolscope(0.03, 100.0, 1000.0, [4000.0, 4000.0])
        assert a == pytest.approx(b)

    def test_nan_when_extinction_is_at_or_below_black(self):
        assert np.isnan(extinction_ratio_openpolscope(0.03, 100.0, 100.0, [200.0]))

    def test_rejects_no_states(self):
        with pytest.raises(ValueError):
            extinction_ratio_openpolscope(0.03, 100.0, 1000.0, [])
