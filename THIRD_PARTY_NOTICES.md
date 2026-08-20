# Third-party notices

> The root `LICENSE` is a **verbatim, unmodified** BSD-3-Clause with a single
> copyright line. That is deliberate: adding "Portions Copyright" lines to it
> makes GitHub's licensee detector return `NOASSERTION`, and the repository then
> displays as having *no licence at all* -- worse for everyone than the tidier
> notice. Upstream copyright is retained where it belongs instead: in the header
> of every ported module, in this file, and in `third_party_licenses/`.

This package is **substantially derived** from two upstream projects, both
BSD-3-Clause licensed and both from the Chan Zuckerberg Biohub. It is not a
wrapper around them: the polarization math was ported from PyTorch to numpy so
that birefringence reconstruction runs without PyTorch and without the phase
machinery. The upstream notices below apply to those derived portions.

That derivation is why this package is BSD-3-Clause rather than MIT like its
sibling `ppm_library`.

To be precise about the reasoning, since an earlier draft of this file
overstated it: BSD-3 is permissive, so relicensing a derivative under MIT would
also have been *legal*, provided the upstream copyright notice, conditions and
disclaimer were retained for the derived portions. BSD-3 is chosen because it is
the honest choice, not the only lawful one -- the non-endorsement clause travels
with the derived code instead of being dropped, and matching the upstream
licence keeps the provenance obvious to anyone reading the tree.

Reviewed with the `license-check` skill on 2026-08-20: resolved `BSD-3-Clause`
(root `LICENSE` plus the `pyproject.toml` declaration, consistent), no copyleft
anywhere in the dependency set, derivation stated in the README and in every
ported module, and no stripped headers. Verdict **COMPLIANT**, with three
attribution gaps found and fixed in the same pass: per-file upstream copyright
lines, the upstream licence texts placed in-tree under `licenses/`, and the
numpy notice below. Not legal advice; escalate to counsel if a binding opinion
is ever needed.

---

## waveorder

- Source: https://github.com/mehta-lab/waveorder
- Licence: BSD 3-Clause, Copyright (c) 2025, Chan Zuckerberg Biohub
  -- full text in [`third_party_licenses/waveorder-LICENSE.txt`](third_party_licenses/waveorder-LICENSE.txt)
- Ported from: `waveorder/stokes.py` and the birefringence half of
  `waveorder/models/inplane_oriented_thick_pol3d.py`

Ported into `polscope_library/stokes.py` and `polscope_library/birefringence.py`:

| Upstream | Here |
|---|---|
| `stokes.calculate_stokes_to_intensity_matrix` | `stokes.calculate_stokes_to_intensity_matrix` |
| `stokes.calculate_intensity_to_stokes_matrix` | `stokes.calculate_intensity_to_stokes_matrix` |
| `stokes.mmul` | `stokes.mmul` |
| `stokes.stokes_after_adr` | `stokes.stokes_after_adr` |
| `stokes.estimate_adr_from_stokes` | `stokes.estimate_adr_from_stokes` |
| `stokes._s12_to_orientation` | `stokes._s12_to_orientation` |
| `stokes.mueller_from_stokes` | `stokes.mueller_from_stokes` |
| `stokes.apply_orientation_offset` | `stokes.apply_orientation_offset` |
| `inplane_oriented_thick_pol3d.apply_inverse_transfer_function` | `birefringence.reconstruct` |

**Deliberately NOT ported:** every phase model (`phase_thick_3d`,
`isotropic_thin_3d`, `isotropic_fluorescent_thick_3d`), the forward/simulation
paths, the napari visualisation helpers, and `correction.estimate_background`
(the estimated-background path).

## recOrder

- Source: https://github.com/mehta-lab/recOrder
- Licence: BSD 3-Clause, Copyright (c) 2020, Chan Zuckerberg Biohub
  -- full text in [`third_party_licenses/recOrder-LICENSE.txt`](third_party_licenses/recOrder-LICENSE.txt)
- **Pinned to tag `v0.4.2rc1`, commit `9f0a37abd4d4c358976617e64855de5321566ac6`**
  (2024-07-16). Note that GitHub labels the older `0.4.1` as "Latest" because
  `v0.4.2rc1` is flagged a pre-release, and that `v0.4.2rc1` is the only tag
  carrying a `v` prefix.
- The repository was **archived read-only on 2026-03-25** and is no longer
  maintained. This is the reason for porting rather than depending on it.

Ported into `polscope_library/`:

| Upstream | Here |
|---|---|
| `cli.apply_inverse_models.radians_to_nanometers` | `stokes.radians_to_nanometers` |
| `cli.apply_inverse_models.birefringence` | folded into `birefringence.reconstruct` |

**Deliberately NOT ported:** `cli.apply_inverse_models.phase` and
`birefringence_and_phase`, the whole napari plugin, the Qt/napari worker
threads, and the click/pydantic CLI machinery.

Still to be extracted in a later change (LC calibration, not reconstruction):
`calib/Calibration.py`, `calib/Optimization.py`, `io/core_functions.py`,
`io/metadata_reader.py`.

## Runtime and development dependencies

Resolved from PyPI. None is copyleft; no AGPL anywhere in the set.

| Package | Licence | Role |
|---|---|---|
| numpy | BSD-3-Clause -- Copyright (c) 2005-2025, NumPy Developers. All rights reserved. | runtime, the only one |
| pytest | MIT | development only |
| black | MIT | development only |
| ruff | MIT | development only |

numpy is BSD-3-Clause, which asks that its copyright line and the licence
conditions be reproduced by anything redistributing it. This package does not
bundle numpy -- pip fetches it at install time -- so the obligation is light,
but the notice is recorded here regardless.

## Deliberate divergences from upstream

Recorded so that anyone diffing against upstream knows these are intentional:

1. **PyTorch to numpy.** Every operation used has an exact numpy equivalent
   (`linalg.pinv`, `linalg.inv`, `einsum`, `moveaxis`, `remainder`, elementwise
   trig), with the same broadcasting and the same sign conventions.
2. **`denom_floor` in `mueller_from_stokes`.** Upstream divides by
   `s1**2 + s2**2`, which goes to zero for perfectly circular light -- the
   condition a good background approaches by construction. The optional floor
   bounds that division. Default `0.0` reproduces upstream exactly.
3. **`radians_to_nanometers` takes nanometres**, where upstream takes
   micrometres and multiplies by 1e3. Same result, one less unit to confuse.
4. **Orientation exposed as `sin(2*theta)`/`cos(2*theta)` properties** on the
   result, because orientation is axial and must not be averaged as a scalar.

## Citation

The method is QLIPP: Guo et al., *eLife* 2020;9:e55502.
https://elifesciences.org/articles/55502
