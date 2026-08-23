#!/usr/bin/env python3
"""Identify the LC-PolScope calibration scheme from acquired data.

Answers, from images alone, three questions we cannot answer from the
MicroManager config:

  1. Which state is extinction?
  2. How do the four swing states pair up into (+S1,-S1) and (+S2,-S2)?
  3. Does the data look like a valid 5-state calibration at all?

WHY THIS WORKS
--------------
waveorder's 5-state system matrix (waveorder/stokes.py,
calculate_stokes_to_intensity_matrix) is, with chi = 2*pi*swing:

    State0 (ext)  [1,        0,        0, -1       ]
    State1 (+S1)  [1,  sin(chi),       0, -cos(chi)]
    State2 (+S2)  [1,        0, sin(chi), -cos(chi)]
    State3 (-S1)  [1, -sin(chi),       0, -cos(chi)]
    State4 (-S2)  [1,        0,-sin(chi), -cos(chi)]

Add the two members of a +/- pair and the sample term cancels:

    I(+S1) + I(-S1) = 2*S0 - 2*cos(chi)*S3
    I(+S2) + I(-S2) = 2*S0 - 2*cos(chi)*S3

Those are EQUAL, pixel by pixel, for any specimen. That identity holds only
for the correct pairing, so comparing the three possible ways to split four
states into two pairs identifies the convention -- with a sample in the
field, needing no swing value, no background, and no calibration file.

The three pairings of states (a,b,c,d) in file order:
    (a,b | c,d)
    (a,c | b,d)   <- waveorder / recOrder order: ext, +S1, +S2, -S1, -S2
    (a,d | b,c)   <- Oldenbourg / LOCI courseware order:
                     ext, +LCA, +LCB, -LCB, -LCA

A SECOND, INDEPENDENT CHECK
---------------------------
On a specimen-free (background) field, S1 = S2 = 0, so all four swing states
collapse to the same value and extinction sits below them. Pass --blank to
assert that: the four swing means should agree closely, and a large spread
means the states are not equivalent swings about a common extinction.

USAGE
-----
    python3 tools/polscope_scheme_check.py <dir-with-5-images>
    python3 tools/polscope_scheme_check.py <multipage.tif>
    python3 tools/polscope_scheme_check.py s0.tif s1.tif s2.tif s3.tif s4.tif
    python3 tools/polscope_scheme_check.py <background-dir> --blank
    python3 tools/polscope_scheme_check.py --selftest

Files in a directory are ordered by a State<N> / _<N> token in the filename
when present, otherwise by sorted filename. Check the printed order.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _read(path):
    if tifffile is not None and path.lower().endswith((".tif", ".tiff")):
        return np.asarray(tifffile.imread(path))
    import imageio.v3 as iio

    return np.asarray(iio.imread(path))


def _state_key(path):
    """Sort key: prefer an explicit State<N>, else a trailing _<N>, else name."""
    name = os.path.basename(path)
    m = re.search(r"[Ss]tate[_\-]?(\d+)", name)
    if m:
        return (0, int(m.group(1)), name)
    m = re.search(r"[_\-](\d+)\.[A-Za-z]+$", name)
    if m:
        return (1, int(m.group(1)), name)
    return (2, 0, name)


def load_states(paths):
    """Return (list-of-2D-float-arrays, list-of-labels)."""
    if len(paths) == 1 and os.path.isdir(paths[0]):
        files = []
        for ext in ("tif", "tiff", "png", "TIF", "TIFF"):
            files.extend(glob.glob(os.path.join(paths[0], "*." + ext)))
        files = sorted(set(files), key=_state_key)
        if len(files) < 4:
            sys.exit("ERROR: found %d image(s) in %s; need 4 or 5." % (len(files), paths[0]))
        if len(files) > 5:
            print("NOTE: %d images found; using the first 5 after ordering." % len(files))
            files = files[:5]
        arrays = [_read(f) for f in files]
        labels = [os.path.basename(f) for f in files]
    elif len(paths) == 1:
        stack = _read(paths[0])
        if stack.ndim < 3:
            sys.exit("ERROR: %s is a single plane; need a 4- or 5-plane stack." % paths[0])
        stack = np.squeeze(stack)
        if stack.shape[0] not in (4, 5):
            sys.exit("ERROR: leading axis is %d, expected 4 or 5 states. Shape=%s" % (stack.shape[0], stack.shape))
        arrays = [stack[i] for i in range(stack.shape[0])]
        labels = ["plane %d" % i for i in range(stack.shape[0])]
    else:
        arrays = [_read(p) for p in paths]
        labels = [os.path.basename(p) for p in paths]

    out = []
    for a in arrays:
        a = np.squeeze(np.asarray(a))
        if a.ndim == 3:
            # Colour or multi-channel input: collapse to one plane.
            axis = int(np.argmin(a.shape))
            a = a.mean(axis=axis)
        out.append(a.astype(np.float64))
    shapes = {a.shape for a in out}
    if len(shapes) != 1:
        sys.exit("ERROR: state images differ in shape: %s" % shapes)
    return out, labels


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

PAIRINGS = [
    ((0, 1), (2, 3), "(1,2 | 3,4)", "neither known convention"),
    ((0, 2), (1, 3), "(1,3 | 2,4)", "waveorder / recOrder 5-state"),
    ((0, 3), (1, 2), "(1,4 | 2,3)", "Oldenbourg / LOCI courseware"),
]


def extinction_ratio(i_extinction, i_elliptical, swing, black_level=0.0):
    """recOrder's extinction ratio (Calibration.calculate_extinction).

        ER = (1 / sin(pi*swing)^2) * (I_ell - I_ext) / (I_ext - black) + 1

    "the ratio of the largest and smallest intensities that the imaging system
    can transmit above background". Computed on a SPECIMEN-FREE field, which is
    what recOrder does during calibration.

    recOrder's own quality bands: >= 100 good, 80-100 okay, < 80 bad (and it
    then suggests: polarizer/LC misoriented, circular analyzer of the wrong
    handedness, condenser not set up for Kohler, or a component such as an
    autofocus dichroic distorting the polarization state).
    """
    denom = i_extinction - black_level
    if denom <= 0:
        return float("nan")
    return (1.0 / np.sin(np.pi * swing) ** 2) * (i_elliptical - i_extinction) / denom + 1.0


def bin_image(a, factor):
    """Mean-bin by an integer factor. All the pairing math is linear, so
    binning changes nothing physically -- it only averages down per-pixel
    noise, which otherwise sets the floor on the residuals and makes a blank
    field look like it has signal."""
    if factor <= 1:
        return a
    h = (a.shape[0] // factor) * factor
    w = (a.shape[1] // factor) * factor
    if h == 0 or w == 0:
        return a
    return a[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def pair_residual(swing_states, pa, pb):
    """Normalized RMS difference between the two pair-sums."""
    sum_a = swing_states[pa[0]] + swing_states[pa[1]]
    sum_b = swing_states[pb[0]] + swing_states[pb[1]]
    scale = float(np.mean(0.5 * (sum_a + sum_b)))
    if scale <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((sum_a - sum_b) ** 2)) / scale)


def analyse(states, labels, blank=False, bin_factor=8, swing_waves=0.03, black_level=0.0):
    n = len(states)
    print("=" * 74)
    print("LC-PolScope calibration-scheme check")
    print("=" * 74)
    print("States loaded, in this order:")
    means = [float(np.mean(s)) for s in states]
    for i, (lab, m) in enumerate(zip(labels, means)):
        s = states[i]
        print(
            "  [%d] %-38s mean=%10.2f  p1=%8.1f  p99=%8.1f"
            % (i, lab[:38], m, np.percentile(s, 1), np.percentile(s, 99))
        )

    ext_idx = int(np.argmin(means))
    print("")
    print("Extinction candidate: index %d (%s) -- the darkest state." % (ext_idx, labels[ext_idx]))
    if ext_idx != 0:
        print("  WARNING: the darkest state is NOT first. waveorder expects extinction")
        print("           in row 0, so the stack would need reordering before use.")

    others = [i for i in range(n) if i != ext_idx]
    if len(others) != 4:
        print("")
        print("Only %d swing states present -- the pairing test needs 4 (5-state" % len(others))
        print("scheme). A 4-state calibration uses three states 120 deg apart and")
        print("has no +/- pairs, so this test does not apply to it.")
        return

    swing = [bin_image(states[i], bin_factor) for i in others]
    swing_means = [means[i] for i in others]
    spread = (max(swing_means) - min(swing_means)) / np.mean(swing_means) * 100.0

    print("")
    print("-" * 74)
    print("CHECK 1  Swing-state equivalence (decisive only on a BLANK field)")
    print("-" * 74)
    print("  Swing-state means: " + ", ".join("%.2f" % m for m in swing_means))
    print("  Spread: %.2f%% of mean" % spread)
    if blank:
        er = extinction_ratio(means[ext_idx], float(np.mean(swing_means)), swing_waves, black_level)
        print("")
        print("  Extinction ratio (recOrder definition, swing=%.4f, black level=%.1f):" % (swing_waves, black_level))
        if not np.isfinite(er):
            print("    n/a -- extinction mean is at or below the black level.")
        else:
            print("    ER = %.1f" % er)
            if er >= 140.0:
                band = "GOOD, and within the 140-200 this scope is expected to reach"
            elif er >= 100.0:
                band = "good by recOrder's band (>=100), but below this scope's usual 140-200"
            elif er >= 80.0:
                band = "OKAY by recOrder's band (80-100) -- worth recalibrating"
            else:
                band = (
                    "BAD (<80). recOrder suggests: polarizer/LC misoriented, circular "
                    "analyzer wrong handedness, condenser not Kohler, or a component "
                    "distorting the polarization state"
                )
            print("    %s" % band)
            print("    (Black level defaults to 0. Pass --black-level with the camera's")
            print("     real offset for a correct number -- ER is sensitive to it.)")
        print("")
        if spread < 5.0:
            print("  PASS: on a specimen-free field the four swing states agree, as the")
            print("        5-state model predicts (S1 = S2 = 0 makes them identical).")
        else:
            print("  FAIL: the four swing states differ by more than 5%% on a field that")
            print("        should have no birefringence. They are not equivalent swings")
            print("        about a common extinction -- suspect the calibration or the")
            print("        assumption that this field is blank.")
    else:
        print("  (Not asserted: pass --blank when the field is specimen-free. With a")
        print("   sample present the swing states are EXPECTED to differ.)")

    print("")
    print("-" * 74)
    print("CHECK 2  Pair identity  I(+S1)+I(-S1) == I(+S2)+I(-S2)   [the decisive one]")
    print("-" * 74)
    print("  Computed on %dx%d-binned data to keep per-pixel noise from setting the" % (bin_factor, bin_factor))
    print("  residual floor. Lower residual = better. Indices are 1-based positions among the")
    print("  four swing states, in the order listed above with extinction removed.")
    print("")
    results = []
    for pa, pb, name, meaning in PAIRINGS:
        r = pair_residual(swing, pa, pb)
        results.append((r, name, meaning))
        print("    %-14s residual = %8.5f    %s" % (name, r, meaning))

    results.sort()
    best_r, best_name, best_meaning = results[0]
    runner_r = results[1][0]

    # How much birefringent signal is actually present to discriminate on.
    # With no sample, all four swing states are identical and every pairing
    # scores the same -- that is "inconclusive", not "no pairing fits".
    swing_stack = np.stack(swing)
    modulation = float(np.mean(swing_stack.max(axis=0) - swing_stack.min(axis=0)) / max(np.mean(swing_stack), 1e-12))

    print("")
    print("  Swing-state modulation: %.3f%% of mean" % (100.0 * modulation))
    print("  (How much the four swing states differ across the field. This is the")
    print("   signal the pairing test relies on -- near zero means a blank field")
    print("   and no test is possible.)")
    print("")

    if not np.isfinite(best_r):
        print("  INCONCLUSIVE: could not compute residuals (all-zero or negative data?).")
        return

    ratio = runner_r / best_r if best_r > 0 else float("inf")

    if ratio > 2.0 and best_r < 0.05:
        print("  RESULT: pairing %s wins clearly (%.1fx better than next)." % (best_name, ratio))
        print("          Consistent with: %s" % best_meaning)
        if best_meaning.startswith("waveorder"):
            print("          -> The stack order matches what waveorder expects. Feed the")
            print("             states in file order with scheme='5-State'.")
        elif best_meaning.startswith("Oldenbourg"):
            print("          -> This is the courseware/Oldenbourg order, NOT waveorder's.")
            print("             Reorder to (ext, +S1, +S2, -S1, -S2) before calling")
            print("             waveorder, or the orientation will be wrong.")
        else:
            print("          -> Matches neither documented convention. Do not reconstruct")
            print("             until the calibration provenance is established.")
    elif modulation < 0.02:
        print("  INCONCLUSIVE: this field carries almost no birefringent signal")
        print("        (modulation %.3f%%), so all three pairings score alike and none" % (100.0 * modulation))
        print("        can be distinguished. This is the expected outcome on a blank or")
        print("        background field -- it is not a failure.")
        print("        -> Re-run on a field with visible birefringent structure")
        print("           (collagen works well). Use --blank on background fields, where")
        print("           CHECK 1 above is the meaningful test.")
    elif ratio <= 2.0:
        print("  INCONCLUSIVE: %s scores best but only %.2fx better than the next," % (best_name, ratio))
        print("        which is not a clear separation. There is some signal here")
        print("        (modulation %.2f%%), so try a field with stronger, more varied" % (100.0 * modulation))
        print("        birefringence -- a field whose fibres all share one orientation")
        print("        cannot separate the two axes.")
    else:
        print(
            "  NO PAIRING FITS (best residual %.4f is too large, despite %.2f%% modulation)."
            % (best_r, 100.0 * modulation)
        )
        print("        The four states do not form two matched +/- pairs. Either the")
        print("        calibration is not a standard 5-state one, the images are not")
        print("        all from the same field/focus, or one state is mislabelled.")
    print("")
    print("Reminder: this identifies the SCHEME and ORDER only. The swing value")
    print("still has to come from the calibration -- look for a recOrder")
    print("calibration metadata file saved alongside the acquisition.")


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def selftest():
    """Synthesize known data and confirm the test identifies the ordering."""
    rng = np.random.default_rng(0)
    chi = 2 * np.pi * 0.03
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w]
    theta = np.deg2rad((xx / w) * 180.0)  # orientation sweeps 0-180 deg
    ret = 0.35 * (yy / h)  # retardance ramp
    s0 = np.full((h, w), 1000.0)
    s3 = s0 * np.cos(ret)
    s1 = s0 * np.sin(ret) * np.sin(2 * theta)
    s2 = -s0 * np.sin(ret) * np.cos(2 * theta)

    def measured(c1, c2, c3):
        """One row of the system matrix applied to the sample Stokes vector.

        Written out here rather than imported from :mod:`polscope_library.stokes`
        on purpose: the self-test is only evidence if the synthetic data comes
        from an independent construction of the same physics.
        """
        return s0 + c1 * s1 + c2 * s2 + c3 * s3

    ext = measured(0, 0, -1)
    p1 = measured(np.sin(chi), 0, -np.cos(chi))  # +S1
    p2 = measured(0, np.sin(chi), -np.cos(chi))  # +S2
    m1 = measured(-np.sin(chi), 0, -np.cos(chi))  # -S1
    m2 = measured(0, -np.sin(chi), -np.cos(chi))  # -S2

    def noisy(a):
        return a + rng.normal(0, 0.5, a.shape)

    print("SELF-TEST 1: stack in waveorder order (ext, +S1, +S2, -S1, -S2)")
    analyse([noisy(x) for x in (ext, p1, p2, m1, m2)], ["ext", "+S1", "+S2", "-S1", "-S2"])

    print("")
    print("")
    print("SELF-TEST 2: same data in courseware order (ext, +S1, +S2, -S2, -S1)")
    analyse([noisy(x) for x in (ext, p1, p2, m2, m1)], ["ext", "+S1", "+S2", "-S2", "-S1"])

    print("")
    print("")
    print("SELF-TEST 3: blank field, waveorder order (expect CHECK 1 to pass)")
    b0 = np.full((h, w), 1000.0)
    b3 = b0.copy()
    bext = b0 - b3
    bsw = b0 - np.cos(chi) * b3
    analyse(
        [noisy(bext), noisy(bsw), noisy(bsw), noisy(bsw), noisy(bsw)], ["ext", "sw1", "sw2", "sw3", "sw4"], blank=True
    )


def main():
    ap = argparse.ArgumentParser(
        description="Identify the LC-PolScope calibration scheme from acquired data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("paths", nargs="*", help="directory, multi-plane stack, or 5 files")
    ap.add_argument(
        "--blank",
        action="store_true",
        help="the field is specimen-free; assert the four swing states agree",
    )
    ap.add_argument(
        "--bin",
        type=int,
        default=8,
        dest="bin_factor",
        help="spatial binning factor for the pairing test (default 8; use 1 to disable)",
    )
    ap.add_argument(
        "--swing",
        type=float,
        default=0.03,
        help="swing in waves, for the extinction ratio (default 0.03)",
    )
    ap.add_argument(
        "--black-level",
        type=float,
        default=0.0,
        dest="black_level",
        help="camera black level, for the extinction ratio (default 0)",
    )
    ap.add_argument("--selftest", action="store_true", help="run built-in verification")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.paths:
        ap.error("give a directory, a multi-plane stack, or 5 image files")
    states, labels = load_states(args.paths)
    analyse(
        states,
        labels,
        blank=args.blank,
        bin_factor=args.bin_factor,
        swing_waves=args.swing,
        black_level=args.black_level,
    )


if __name__ == "__main__":
    main()
