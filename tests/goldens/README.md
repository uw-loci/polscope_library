# Goldens: the recorded answers of a released version

`goldens.npz` holds the output of 54 public functions of polscope_library on fixed inputs,
recorded from v0.2.0 (commit ba04d16) with the verify-by-oracle skill's `oracle.py capture`;
`manifest.json` records the commit, the library versions, the seed, the sha256 of every
output and the comparison rule each one is judged by. `tests/test_goldens.py` replays the
current code against them on every test run, so a change that moves a number users rely
on fails there with the first differing element located, whether or not any other test
noticed.

## Files

| File | What it is |
|---|---|
| `cases.py` | the inputs (built from `SEED` or from the two committed OpenPolScope crops in `tests/fixtures/` (real five-state data with their backgrounds)) and one `case_<name>()` per output; `RULES` holds the comparison rule and its reason, `LABELS` names the elements of the short vectors |
| `compare.py` | the three comparison rules (exact, tolerance, angle on the circle); identical in ppm_library, polscope_library and microscope_imageprocessing, kept in sync by hand |
| `refresh.py` | the only way `goldens.npz` and `manifest.json` change; see below |
| `goldens.npz`, `manifest.json` | the recorded outputs and their record |

## Rules

Every output starts at `exact`. Integer, boolean and label outputs, masks, histogram counts
and I/O round trips stay exact: the same code on the same input gives the same bits, and a
difference is a change. A tolerance or an angle rule is a claim that the two versions may
differ by that much for a named cause, and `RULES` in `cases.py` carries the cause beside the
number. The causes used here, and how the bands were chosen:

| Functions | Rule | Cause |
|---|---|---|
| `reconstruct_openpolscope`'s dead-pixel mask, `radians_to_nanometers` | exact | boolean output; one multiplication and one division |
| every orientation in radians (`reconstruct` by both methods, with measured and estimated backgrounds, `reconstruct_openpolscope` mirrored with an azimuth reference, `estimate_adr_from_stokes`, `apply_orientation_offset`, `transform_orientation`, `average_orientation`, `vector_to_orientation`) | angle, `period pi, atol 1e-12` (1e-10 after the estimated-background fit) | arctan2 differs by a few ulps between libm builds; 0 and pi are the same slow axis |
| retardance, transmittance and depolarization of the real crops, `stokes_from_intensities`, the synthetic states, the ADR round trip | tolerance `atol 1e-9, rtol 1e-12` (retardance in radians and depolarization `1e-12, 1e-12`) | arccos/arctan and sqrt, and the per-pixel matrix product (`tensordot`/`einsum` reach BLAS, where summation order and FMA use differ between builds) |
| the system matrices, `stokes_after_adr`, `mueller_from_stokes`, `compensator_stokes`, `retardances_for_stokes`, `ideal_palette`, `swing_state_distances`, the extinction ratios, `orientation_to_vector` | tolerance `atol 1e-15 to 1e-13, rtol 1e-14 to 1e-13` | sin and cos of the swing and the LC retardances: 1 to 4 ulps between libm builds; the inverse matrix goes through LAPACK |
| `estimate_background`, `fit_residual_background` and the estimated-background reconstruction | tolerance `rtol 1e-10` (atol 1e-12 to 1e-9 by scale) | least-squares polynomial fit through LAPACK |
| `calibrate` on the simulated microscope (`single_pass` and `iterative`): palette | tolerance `atol 2e-5 waves` | `scipy.optimize.minimize_scalar(method="bounded")` stops at `xatol` 1e-5 waves; the band is twice the solver's own tolerance, for a scipy release that stops one step earlier or later |
| `calibrate` summary (extinction ratio, black level, exposures, warnings) | tolerance `rtol 1e-3` | the extinction ratio follows the extinction setting quadratically, so 2e-5 waves moves it by well under 1e-3; the exposure count is an integer and must not move at all inside that band |
| `find_fringe_landmarks`, `curve_from_sweep` and its voltage round trip | tolerance `atol 1e-12 (1e-10 for the round trip), rtol 1e-12` | arcsin of a normalized fringe, then `np.interp` |
| `scheme_check.bin_image`, `pair_residual` | tolerance `rtol 1e-13 / 1e-12` | means over 8x8 blocks and an RMS over the crop: summation order in the last bits |

Measured on 2026-11-26 (WSL2, x86_64, Python 3.12.3, numpy 2.3.5, scipy 1.17.0): a second
run and a run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` reproduced every golden to the
bit, so the bands above allow for other machines and library versions. GitHub Actions on Python
3.10 and 3.12 (numpy 2.2.6 and 2.5.5, scipy 1.15.3 and 1.18.1) passed every case inside its
band on the first run (2026-11-26), the LC search's exposure counts included. Run `pytest -s tests/test_goldens.py` to
print the largest difference every tolerance and angle case actually saw.

## When the replay fails

The current code gives a different answer than v0.2.0 (commit ba04d16) did. Decide which it is:

- **A bug.** Fix the code. The golden stays.
- **An intended change** (a new algorithm, a corrected formula, a changed default). Record it
  in the changelog or the issue that describes the new behavior, then refresh (below) with
  that entry as the reason.

Never widen a rule, drop a case or refresh the goldens to make the replay pass. A widened
tolerance needs a measured cause written into `RULES`; the skill's `weakened_asserts.py`
reports every widening in a push range, and the reason beside it is what makes it fine.

## Refreshing

```bash
python3 tests/goldens/refresh.py --reason "<version>: <what changed and where it is documented>"
```

`refresh.py` runs every case with `oracle.py capture` from
`~/.claude/skills/verify-by-oracle/scripts/oracle.py` (pass `--oracle` or set `ORACLE_PY`
for another location), packs the results and writes the reason, commit and versions into
`manifest.json`. It refuses a tree with uncommitted changes to tracked files, because a
golden has to name the commit it came from: commit the code change first, refresh, then
commit the goldens with the same reason in the commit message. A commit that touches
`goldens.npz` without saying why in its message is a finding, not a refresh.

`--rules-only` rewrites the rules in `manifest.json` from `cases.RULES` without touching
the arrays, for a change to a rule that is not a change to the code; it still takes a
`--reason`.

Adding a case: write `case_<name>()` and its rule, commit, run `refresh.py` with the reason
"new case <name>", and check in the diff that only the new array was added (the sha256 of
every other case in `manifest.json` is unchanged).
