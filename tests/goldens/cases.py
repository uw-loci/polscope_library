"""Golden cases for polscope_library (README.md in this folder).

Every ``case_<name>()`` builds its input from SEED or from the two committed OpenPolScope
crops in ``tests/fixtures/`` (real five-state data with their backgrounds), calls one public
function and returns one numpy array. RULES says how tests/test_goldens.py compares it with
the recorded golden and, when the rule is not exact, why. LABELS names the elements of the
short vectors.
"""

import functools
from pathlib import Path

import numpy as np

import polscope_library
from polscope_library import (
    EstimatedBackground,
    apply_orientation_offset,
    average_orientation,
    calculate_intensity_to_stokes_matrix,
    calculate_stokes_to_intensity_matrix,
    estimate_adr_from_stokes,
    estimate_background,
    mmul,
    mueller_from_stokes,
    orientation_to_vector,
    radians_to_nanometers,
    reconstruct,
    reconstruct_openpolscope,
    scheme_check,
    stokes_after_adr,
    stokes_from_intensities,
    transform_orientation,
    vector_to_orientation,
)
from polscope_library.background_fit import fit_residual_background
from polscope_library.calibration import (
    CalibrationSettings,
    SimulatedPolScope,
    calibrate,
    compensator_stokes,
    curve_from_sweep,
    extinction_ratio,
    extinction_ratio_openpolscope,
    find_fringe_landmarks,
    ideal_palette,
    retardances_for_stokes,
)
from polscope_library.calibration.workflow import swing_state_distances

SEED = 20261126
LIBRARY_VERSION = polscope_library.__version__
RECORD_VERSIONS = ["numpy", "scipy"]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REFSLIDE = FIXTURES / "openpolscope_3_20b_refslide_crop.npz"
SESSION_2026_10_16 = FIXTURES / "openpolscope_3_20b_2026_10_16_crop.npz"
SWING = 0.03
WAVELENGTH_NM = 546.0

# --------------------------------------------------------------------------- inputs


def _rng(offset=0):
    return np.random.default_rng(SEED + offset)


@functools.lru_cache(maxsize=None)
def crop(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def library_order(states_openpolscope_order):
    """OpenPolScope's State0..4 -> library order (ext, +S1, +S2, -S1, -S2)."""
    s = states_openpolscope_order
    return np.stack([s[0], s[1], s[2], s[4], s[3]]).astype(np.float64)


def _fixture_args(path, background=True):
    c = crop(path)
    return {
        "intensities": library_order(c["sample_states"]),
        "swing_waves": float(c["swing"]),
        "wavelength_nm": float(c["wavelength_nm"]),
        "background_intensities": library_order(c["background_states"]) if background else None,
        "black_level": float(c["black_level"]),
    }


def sample_field(shape=(32, 40)):
    """A known specimen: retardance ramps to a quarter wave, orientation sweeps a half turn."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    retardance = (0.02 + 0.9 * xx / (shape[1] - 1)) * np.pi / 2
    orientation = (yy / shape[0] * np.pi + 0.3 * np.sin(xx / 5.0)) % np.pi
    transmittance = 1000.0 * (0.7 + 0.3 * _rng(1).random(shape))
    depolarization = 0.85 + 0.15 * _rng(2).random(shape)
    return retardance, orientation, transmittance, depolarization


def synthetic_states(swing=SWING):
    ret, ori, trans, depol = sample_field()
    stokes = stokes_after_adr(ret, ori, trans, depol)
    return mmul(calculate_stokes_to_intensity_matrix(swing, scheme="5-State"), np.stack(stokes))


def smooth_image(shape=(64, 64), offset=0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    surface = 3.0 + 0.02 * xx - 0.015 * yy + 0.0004 * xx * yy - 0.0003 * yy**2
    return surface + _rng(10 + offset).normal(0.0, 0.05, shape)


def residual_components():
    """u, v with a smooth residual and a small bright specimen in one corner."""
    u = 0.01 * smooth_image(offset=1) / 3.0
    v = -0.008 * smooth_image(offset=2) / 3.0
    u[8:20, 40:56] += 0.2
    v[8:20, 40:56] -= 0.15
    return u, v


def synthetic_sweep():
    """A fringe sweep of one crystal from a smooth, monotone voltage-to-retardance curve."""
    volts = np.linspace(0.0, 20.0, 201)
    waves = 0.06 + 1.5 * np.exp(-volts / 4.0)
    signal = 4000.0 * 0.5 * (1.0 - 0.9925 * np.sin(2 * np.pi * waves)) + 102.0
    return volts, signal


def rotation(axis, degrees):
    """3x3 rotation about one axis of the (S1, S2, S3) frame."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    m = np.eye(3)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


def simulated_scope():
    return SimulatedPolScope(read_noise=0.0, residual=rotation(0, 4.0) @ rotation(1, 2.5), hysteresis=0.02)


@functools.lru_cache(maxsize=None)
def calibration(strategy):
    settings = CalibrationSettings(swing_waves=SWING, black_level=102.0)
    return calibrate(simulated_scope(), settings, strategy=strategy)


# --------------------------------------------------------------------------- Stokes matrices


def case_stokes_to_intensity_5state():
    return calculate_stokes_to_intensity_matrix(SWING, scheme="5-State")


def case_stokes_to_intensity_4state():
    return calculate_stokes_to_intensity_matrix(SWING, scheme="4-State")


def case_intensity_to_stokes_5state():
    return calculate_intensity_to_stokes_matrix(SWING, scheme="5-State")


def case_stokes_from_intensities_refslide():
    return stokes_from_intensities(library_order(crop(REFSLIDE)["sample_states"]), float(crop(REFSLIDE)["swing"]))


# --------------------------------------------------------------------------- reconstruction on the real crops


def case_qlipp_retardance_nm_refslide():
    return reconstruct(**_fixture_args(REFSLIDE, background=False)).retardance_nm


def case_qlipp_orientation_refslide():
    return reconstruct(**_fixture_args(REFSLIDE, background=False)).orientation_rad


def case_qlipp_transmittance_refslide():
    return reconstruct(**_fixture_args(REFSLIDE, background=False)).transmittance


def case_qlipp_depolarization_refslide():
    return reconstruct(**_fixture_args(REFSLIDE, background=False)).depolarization


def case_qlipp_background_retardance_nm_refslide():
    return reconstruct(**_fixture_args(REFSLIDE)).retardance_nm


def case_qlipp_background_orientation_refslide():
    return reconstruct(**_fixture_args(REFSLIDE)).orientation_rad


def case_openpolscope_retardance_nm_refslide():
    return reconstruct(method="openpolscope", **_fixture_args(REFSLIDE)).retardance_nm


def case_openpolscope_orientation_refslide():
    return reconstruct(method="openpolscope", **_fixture_args(REFSLIDE)).orientation_rad


def case_openpolscope_retardance_nm_2026_10_16():
    return reconstruct(method="openpolscope", **_fixture_args(SESSION_2026_10_16)).retardance_nm


def case_openpolscope_orientation_2026_10_16():
    return reconstruct(method="openpolscope", **_fixture_args(SESSION_2026_10_16)).orientation_rad


def _openpolscope_raw():
    c = crop(SESSION_2026_10_16)
    return reconstruct_openpolscope(
        library_order(c["sample_states"]),
        float(c["swing"]),
        background_intensities=library_order(c["background_states"]),
        black_level=float(c["black_level"]),
        azimuth_reference_deg=10.0,
        mirror=True,
    )


def case_openpolscope_raw_retardance_rad():
    return _openpolscope_raw()[0]


def case_openpolscope_raw_orientation_mirrored():
    return _openpolscope_raw()[1]


def case_openpolscope_raw_dead():
    return _openpolscope_raw()[2]


def _estimated_background():
    settings = EstimatedBackground(order=1, block_size=8, max_block_retardance_nm=None)
    return reconstruct(method="openpolscope", estimated_background=settings, **_fixture_args(REFSLIDE))


def case_estimated_background_retardance_nm_refslide():
    return _estimated_background().retardance_nm


def case_estimated_background_orientation_refslide():
    return _estimated_background().orientation_rad


def case_estimated_background_fit_refslide():
    fit = _estimated_background().background_fit
    values = [float(fit.applied), fit.blocks_total, fit.blocks_used, fit.peak_removed_nm, fit.mean_removed_nm]
    return np.asarray(values, dtype=np.float64)


# --------------------------------------------------------------------------- forward model and inversion


def case_stokes_after_adr():
    return np.stack(stokes_after_adr(*sample_field()))


def case_synthetic_states_5state():
    return synthetic_states()


def _adr_roundtrip():
    stokes = stokes_from_intensities(synthetic_states(), SWING)
    return estimate_adr_from_stokes(*stokes)


def case_adr_roundtrip_retardance():
    return _adr_roundtrip()[0]


def case_adr_roundtrip_orientation():
    return _adr_roundtrip()[1]


def case_adr_roundtrip_transmittance():
    return _adr_roundtrip()[2]


def case_adr_roundtrip_depolarization():
    return _adr_roundtrip()[3]


def case_mueller_from_stokes_forward():
    stokes = stokes_after_adr(*sample_field())
    return np.asarray(mueller_from_stokes(*stokes, direction="forward"))


def case_mueller_from_stokes_inverse():
    stokes = stokes_after_adr(*sample_field())
    return np.asarray(mueller_from_stokes(*stokes, direction="inverse"))


def case_radians_to_nanometers():
    return radians_to_nanometers(sample_field()[0], WAVELENGTH_NM)


# --------------------------------------------------------------------------- orientation geometry


def case_orientation_offset_rotate_flip():
    return apply_orientation_offset(sample_field()[1], rotate=True, flip=True)


def case_transform_orientation_flip_x_rotate_30():
    return transform_orientation(sample_field()[1], flip_x=True, rotate_deg=30.0)


def case_average_orientation_weighted():
    ori = sample_field()[1]
    stack = np.stack([(ori + k * 0.05) % np.pi for k in range(5)])
    weights = np.stack([sample_field()[0] * (1 + k) for k in range(5)])
    return average_orientation(stack, weights=weights, axis=0)


def case_orientation_vector():
    return np.stack(orientation_to_vector(sample_field()[1]))


def case_orientation_vector_roundtrip():
    return vector_to_orientation(*orientation_to_vector(sample_field()[1]))


# --------------------------------------------------------------------------- estimated background


def case_estimate_background_order2():
    return estimate_background(smooth_image(), order=2, block_size=16)


def case_estimate_background_order1_masked():
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    mask[3, 3] = False
    return estimate_background(smooth_image(), order=1, block_size=16, block_mask=mask)


def _residual_fit():
    u, v = residual_components()
    settings = EstimatedBackground(order=2, block_size=16, max_block_retardance_nm=20.0)
    return fit_residual_background(u, v, np.ones_like(u, dtype=bool), 1.0, WAVELENGTH_NM, settings)


def case_fit_residual_background_u():
    return _residual_fit()[0]


def case_fit_residual_background_v():
    return _residual_fit()[1]


def case_fit_residual_background_fit():
    fit = _residual_fit()[2]
    values = [float(fit.applied), fit.blocks_total, fit.blocks_used, fit.peak_removed_nm, fit.mean_removed_nm]
    return np.asarray(values, dtype=np.float64)


# --------------------------------------------------------------------------- calibration


def case_compensator_stokes_grid():
    lca = np.linspace(0.0, 0.5, 11)[:, None]
    lcb = np.linspace(0.0, 0.5, 9)[None, :]
    return np.asarray(compensator_stokes(lca, lcb))


def case_retardances_for_stokes_roundtrip():
    lca = np.linspace(0.05, 0.45, 7)[:, None]
    lcb = np.linspace(0.3, 0.7, 5)[None, :]
    return np.asarray(retardances_for_stokes(compensator_stokes(lca, lcb)))


def _palette_array(palette):
    return np.asarray([palette[k] for k in palette], dtype=np.float64)


def case_ideal_palette_5state():
    return np.asarray(ideal_palette(SWING, scheme="5-State"), dtype=np.float64)


def case_ideal_palette_4state():
    return np.asarray(ideal_palette(SWING, scheme="4-State"), dtype=np.float64)


def case_swing_state_distances_5state():
    palette = {"state%d" % i: tuple(row) for i, row in enumerate(ideal_palette(SWING, scheme="5-State"))}
    d = swing_state_distances(palette)
    return np.asarray([d[k] for k in d], dtype=np.float64)


def case_extinction_ratios():
    values = [
        extinction_ratio(SWING, 102.0, 110.0, 900.0),
        extinction_ratio_openpolscope(SWING, 102.0, 110.0, [900.0, 910.0, 905.0, 895.0]),
        scheme_check.extinction_ratio(110.0, 900.0, SWING, black_level=102.0),
    ]
    return np.asarray(values, dtype=np.float64)


def case_calibrate_single_pass_palette():
    return _palette_array(calibration("single_pass").palette_waves)


def case_calibrate_single_pass_summary():
    r = calibration("single_pass")
    return np.asarray([r.extinction_ratio, r.black_level, r.exposures, len(r.warnings)], dtype=np.float64)


def case_calibrate_iterative_palette():
    return _palette_array(calibration("iterative").palette_waves)


def case_calibrate_iterative_summary():
    r = calibration("iterative")
    return np.asarray([r.extinction_ratio, r.black_level, r.exposures, len(r.warnings)], dtype=np.float64)


# --------------------------------------------------------------------------- fringe sweep


def case_sweep_landmarks():
    volts, signal = synthetic_sweep()
    landmarks = find_fringe_landmarks(volts, signal)
    return np.asarray([[lm.voltage, lm.retardance_waves] for lm in landmarks], dtype=np.float64)


def case_sweep_curve():
    volts, signal = synthetic_sweep()
    curve = curve_from_sweep(volts, signal)
    probe = np.linspace(curve.voltages.min(), curve.voltages.max(), 60)
    return np.asarray(curve.retardance_for(probe), dtype=np.float64)


def case_sweep_curve_voltage_roundtrip():
    volts, signal = synthetic_sweep()
    curve = curve_from_sweep(volts, signal, offset_waves=0.25)
    lo, hi = curve.span_waves
    return np.asarray(curve.voltage_for(np.linspace(lo, hi, 30)), dtype=np.float64)


# --------------------------------------------------------------------------- scheme check


def case_scheme_bin_image():
    return scheme_check.bin_image(crop(REFSLIDE)["sample_states"][1].astype(np.float64), 8)


def case_scheme_pair_residual():
    states = library_order(crop(REFSLIDE)["sample_states"])[1:]
    return np.asarray(scheme_check.pair_residual(states, (0, 2), (1, 3)), dtype=np.float64)


# --------------------------------------------------------------------------- rules

_TRIG = "arctan2, arccos, sin and cos: transcendental functions differ by a few ulps between libm builds"
_MATMUL = (
    "per-pixel matrix products (tensordot/einsum reach BLAS): summation order and FMA use differ "
    "between builds in the last bits"
)
_LSTSQ = (
    "least-squares polynomial fit through LAPACK: different LAPACK builds round the normal-equation "
    "solve differently in the last bits"
)
_SEARCH = (
    "scipy minimize_scalar(method='bounded') stops at xatol 1e-5 waves; the band is twice that, since a "
    "different scipy release may stop one step earlier or later"
)
_RATIO = (
    _SEARCH
    + "; the extinction ratio follows the extinction setting quadratically, "
    + "so 2e-5 waves moves it by well under 1e-3"
)
_RAD = {"kind": "angle", "period": np.pi, "atol": 1e-12, "reason": _TRIG}
_NM = {"kind": "tolerance", "atol": 1e-9, "rtol": 1e-12, "reason": _TRIG + "; " + _MATMUL}

RULES = {
    "stokes_to_intensity_5state": {"kind": "tolerance", "atol": 1e-15, "rtol": 1e-14, "reason": _TRIG},
    "stokes_to_intensity_4state": {"kind": "tolerance", "atol": 1e-15, "rtol": 1e-14, "reason": _TRIG},
    "intensity_to_stokes_5state": {
        "kind": "tolerance",
        "atol": 1e-13,
        "rtol": 1e-13,
        "reason": _TRIG + "; pseudo-inverse through LAPACK",
    },
    "stokes_from_intensities_refslide": {"kind": "tolerance", "atol": 1e-9, "rtol": 1e-12, "reason": _MATMUL},
    "qlipp_retardance_nm_refslide": _NM,
    "qlipp_orientation_refslide": _RAD,
    "qlipp_transmittance_refslide": _NM,
    "qlipp_depolarization_refslide": _NM,
    "qlipp_background_retardance_nm_refslide": _NM,
    "qlipp_background_orientation_refslide": _RAD,
    "openpolscope_retardance_nm_refslide": _NM,
    "openpolscope_orientation_refslide": _RAD,
    "openpolscope_retardance_nm_2026_10_16": _NM,
    "openpolscope_orientation_2026_10_16": _RAD,
    "openpolscope_raw_retardance_rad": {"kind": "tolerance", "atol": 1e-12, "rtol": 1e-12, "reason": _TRIG},
    "openpolscope_raw_orientation_mirrored": _RAD,
    "openpolscope_raw_dead": {"kind": "exact"},
    "estimated_background_retardance_nm_refslide": {
        "kind": "tolerance",
        "atol": 1e-9,
        "rtol": 1e-10,
        "reason": _TRIG + "; " + _LSTSQ,
    },
    "estimated_background_orientation_refslide": {
        "kind": "angle",
        "period": np.pi,
        "atol": 1e-10,
        "reason": _TRIG + "; " + _LSTSQ,
    },
    "estimated_background_fit_refslide": {"kind": "tolerance", "atol": 1e-9, "rtol": 1e-10, "reason": _LSTSQ},
    "stokes_after_adr": {"kind": "tolerance", "atol": 1e-12, "rtol": 1e-13, "reason": _TRIG},
    "synthetic_states_5state": {"kind": "tolerance", "atol": 1e-9, "rtol": 1e-12, "reason": _TRIG + "; " + _MATMUL},
    "adr_roundtrip_retardance": {"kind": "tolerance", "atol": 1e-12, "rtol": 1e-12, "reason": _TRIG + "; " + _MATMUL},
    "adr_roundtrip_orientation": _RAD,
    "adr_roundtrip_transmittance": {"kind": "tolerance", "atol": 1e-9, "rtol": 1e-12, "reason": _TRIG + "; " + _MATMUL},
    "adr_roundtrip_depolarization": {
        "kind": "tolerance",
        "atol": 1e-12,
        "rtol": 1e-12,
        "reason": _TRIG + "; " + _MATMUL,
    },
    "mueller_from_stokes_forward": {"kind": "tolerance", "atol": 1e-12, "rtol": 1e-13, "reason": _TRIG},
    "mueller_from_stokes_inverse": {"kind": "tolerance", "atol": 1e-12, "rtol": 1e-13, "reason": _TRIG},
    "radians_to_nanometers": {"kind": "exact"},
    "orientation_offset_rotate_flip": _RAD,
    "transform_orientation_flip_x_rotate_30": _RAD,
    "average_orientation_weighted": _RAD,
    "orientation_vector": {"kind": "tolerance", "atol": 1e-15, "rtol": 1e-14, "reason": _TRIG},
    "orientation_vector_roundtrip": _RAD,
    "estimate_background_order2": {"kind": "tolerance", "atol": 1e-10, "rtol": 1e-10, "reason": _LSTSQ},
    "estimate_background_order1_masked": {"kind": "tolerance", "atol": 1e-10, "rtol": 1e-10, "reason": _LSTSQ},
    "fit_residual_background_u": {"kind": "tolerance", "atol": 1e-12, "rtol": 1e-10, "reason": _LSTSQ},
    "fit_residual_background_v": {"kind": "tolerance", "atol": 1e-12, "rtol": 1e-10, "reason": _LSTSQ},
    "fit_residual_background_fit": {"kind": "tolerance", "atol": 1e-9, "rtol": 1e-10, "reason": _LSTSQ},
    "compensator_stokes_grid": {"kind": "tolerance", "atol": 1e-15, "rtol": 1e-14, "reason": _TRIG},
    "retardances_for_stokes_roundtrip": {"kind": "tolerance", "atol": 1e-13, "rtol": 1e-13, "reason": _TRIG},
    "ideal_palette_5state": {"kind": "tolerance", "atol": 1e-13, "rtol": 1e-13, "reason": _TRIG},
    "ideal_palette_4state": {"kind": "tolerance", "atol": 1e-13, "rtol": 1e-13, "reason": _TRIG},
    "swing_state_distances_5state": {"kind": "tolerance", "atol": 1e-13, "rtol": 1e-13, "reason": _TRIG},
    "extinction_ratios": {"kind": "tolerance", "atol": 1e-12, "rtol": 1e-13, "reason": _TRIG},
    "calibrate_single_pass_palette": {"kind": "tolerance", "atol": 2e-5, "rtol": 0.0, "reason": _SEARCH},
    "calibrate_single_pass_summary": {
        "kind": "tolerance",
        "atol": 0.0,
        "rtol": 1e-3,
        "reason": _RATIO,
    },
    "calibrate_iterative_palette": {"kind": "tolerance", "atol": 2e-5, "rtol": 0.0, "reason": _SEARCH},
    "calibrate_iterative_summary": {
        "kind": "tolerance",
        "atol": 0.0,
        "rtol": 1e-3,
        "reason": _RATIO,
    },
    "sweep_landmarks": {
        "kind": "tolerance",
        "atol": 1e-12,
        "rtol": 1e-12,
        "reason": _TRIG + "; arcsin of a normalized fringe",
    },
    "sweep_curve": {
        "kind": "tolerance",
        "atol": 1e-12,
        "rtol": 1e-12,
        "reason": _TRIG + "; arcsin of a normalized fringe",
    },
    "sweep_curve_voltage_roundtrip": {
        "kind": "tolerance",
        "atol": 1e-10,
        "rtol": 1e-12,
        "reason": _TRIG + "; arcsin of a normalized fringe, then np.interp",
    },
    "scheme_bin_image": {
        "kind": "tolerance",
        "atol": 1e-10,
        "rtol": 1e-13,
        "reason": "mean over 8x8 blocks: summation order in the last bits",
    },
    "scheme_pair_residual": {
        "kind": "tolerance",
        "atol": 1e-13,
        "rtol": 1e-12,
        "reason": "RMS over the crop: summation order in the last bits",
    },
}

LABELS = {
    "estimated_background_fit_refslide": [
        "applied",
        "blocks_total",
        "blocks_used",
        "peak_removed_nm",
        "mean_removed_nm",
    ],
    "fit_residual_background_fit": ["applied", "blocks_total", "blocks_used", "peak_removed_nm", "mean_removed_nm"],
    "extinction_ratios": ["recorder", "openpolscope", "scheme_check"],
    "calibrate_single_pass_summary": ["extinction_ratio", "black_level", "exposures", "n_warnings"],
    "calibrate_iterative_summary": ["extinction_ratio", "black_level", "exposures", "n_warnings"],
}


def all_case_names():
    return sorted(name[len("case_") :] for name in globals() if name.startswith("case_"))
