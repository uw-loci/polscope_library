# polscope_library

Birefringence reconstruction for **LC-PolScope** microscopes: quantitative
polarized-light microscopy using a liquid-crystal universal compensator.
Acquire N calibrated polarization states, recover retardance and slow-axis
orientation.

> **Part of the [QPSC (QuPath Scope Control)](https://github.com/uw-loci/qupath-extension-qpsc) system.**
> Sibling of [`ppm_library`](https://github.com/uw-loci/ppm_library), which does
> the same job for our rotation-stage polarized-light scope.

## What this is, and what it deliberately is not

The reconstruction math is **ported from** [`waveorder`](https://github.com/mehta-lab/waveorder)
and [`recOrder`](https://github.com/mehta-lab/recOrder) (both BSD-3-Clause, Chan
Zuckerberg Biohub) rather than depended on. Two reasons:

- **recOrder is archived read-only** (March 2026) and unmaintained.
- **waveorder requires Python >= 3.12 and PyTorch**, which are there for the
  *phase* reconstruction's FFT machinery. We reconstruct birefringence only, and
  that is elementary linear algebra. Ported to numpy, it runs on Python 3.10
  with no heavyweight dependency -- which matters when the code has to sit on a
  microscope-control PC.

**There is no phase reconstruction here and there is not intended to be.** See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for exactly what was ported,
what was left behind, and the deliberate divergences.

## Install

```bash
pip install -e ".[dev]"          # core + test tooling
pip install -e ".[dev,io]"       # also the scheme-check diagnostic (adds tifffile)
```

If an import fails with `cannot import name ... from 'polscope_library'
(unknown location)`, Python assembled the package from a bare folder named
`polscope_library` on `sys.path` (typically this repo's root, when the process
starts in the folder that contains the checkouts) instead of the installed
package. Re-run `pip install -e .` from a current checkout: the install now
registers the project root as a path entry, which wins over that folder.

## Use

```python
from polscope_library import reconstruct

result = reconstruct(
    intensities=[s0, s1, s2, s3, s4],          # IN CALIBRATION ORDER
    swing_waves=0.03,                           # waves, from the calibration
    wavelength_nm=546,                          # illumination wavelength
    scheme="5-State",
    background_intensities=[b0, b1, b2, b3, b4],  # specimen-free, defocused
)

result.retardance_nm        # nanometres
result.orientation_rad      # radians, [0, pi) -- axial, see below
result.transmittance
result.depolarization
```

## Checking a calibration you inherited

`reconstruct` takes the state images **in calibration order**, and a wrong order
is the worst failure mode this package has: it does not raise, it silently
rotates or mirrors the orientation map. Qualitative contrast survives it, so
good-looking images are not evidence that the order is right.

If you did not run the calibration yourself, identify it from the data:

```bash
polscope-scheme-check /path/to/five/state/images       # a directory, a stack, or 5 files
polscope-scheme-check --blank /path/to/background      # specimen-free field
polscope-scheme-check --selftest                       # verify the tool itself
```

It answers three questions the MicroManager config cannot: which state is
extinction, how the four swing states pair into (+S1,-S1) and (+S2,-S2), and
whether the data looks like a valid 5-state calibration at all. The decisive
test is the pair identity `I(+S1)+I(-S1) == I(+S2)+I(-S2)`, which holds for any
specimen and so distinguishes the candidate orderings without a reference slide.

Two caveats worth knowing before you trust a run:

- **It needs birefringent structure.** On a blank or background field all
  orderings score alike, and the tool reports INCONCLUSIVE rather than guessing.
  Collagen works well. Use `--blank` on background fields, where the
  swing-state-equivalence check is the meaningful one instead.
- **It identifies scheme and order only.** The swing value still has to come
  from the calibration metadata.

`--blank` additionally reports the extinction ratio, graded against recOrder's
bands (>=100 good, 80-100 okay, <80 bad).

## Four things that will bite you

**1. All states must share one exposure.** The inversion treats the intensities
as samples of a single radiometric scale. A per-state exposure or gain
difference biases retardance and orientation, and the images look completely
normal. There is no way to detect it after the fact.

**2. State order is positional.** The system matrix rows correspond to specific
polarization states in a fixed order (for 5-state: extinction, +S1, +S2, -S1,
-S2). Supplying a permutation raises no error -- it silently rotates or mirrors
the orientation map. If you are not certain of the order your calibration
produced, you can determine it empirically: adding the two members of a +/- pair
cancels the sample term, so `I(+S1) + I(-S1) == I(+S2) + I(-S2)` pixel-by-pixel
for *any* specimen, and only for the correct pairing.

**3. Orientation is axial -- never average it as a scalar.** 0 and pi are the
same physical orientation. The mean of 179 and 1 degrees is 90 degrees:
perpendicular to the truth, and entirely plausible-looking. Anything that
averages pixels later -- pyramid downsampling, tile blending, binning -- must
use `result.orientation_sin2` / `result.orientation_cos2`, which average
correctly as a vector, and recover the angle afterwards. The magnitude of that
resultant is a free coherence measure.

**4. Retardance is capped at a quarter wave.** It is recovered through an
arcsine, so the result folds back above pi/2 rather than continuing to climb
(about 137 nm at 546 nm). A strongly birefringent sample reads as less retarding
than it is. This is upstream's algorithm, not an artifact of the port.

## Background correction

Capture the background **specimen-free and slightly defocused (10-20 um)**. Dust
on the slide or coverslip that lands in the reference reappears as artificial
negative spots in every corrected image.

Backgrounds drift, so re-acquire often and keep track of which one produced a
given result -- reconstructing the same data against a different background
gives a different answer.

### Estimated background (opt-in)

A measured background removes the instrument's residual as it was where and
when the background was taken. Strain elsewhere on the slide, a different focus
or slow drift leave a smooth ramp of a few nanometres that differs from tile to
tile and shows as a step at every mosaic seam. `estimated_background=` fits a
low-order 2D polynomial to that residual in each image and subtracts it, after
the measured background (recOrder's "estimated background"):

```python
from polscope_library import EstimatedBackground, reconstruct

result = reconstruct(states, swing_waves=0.03, wavelength_nm=549,
                     background_intensities=bg, method="openpolscope",
                     estimated_background=EstimatedBackground())
result.background_fit   # applied?, blocks used, peak/mean nm removed
```

- **It cannot tell a smooth residual from smooth specimen.** On a sparse
  sample the fit is the residual; on tissue filling the tile it would remove
  tissue. So blocks whose median retardance is above `max_block_retardance_nm`
  (default 5 nm) are left out of the fit, and when fewer than
  `min_background_fraction` (default 25%) of blocks remain, the image is left
  uncorrected and `background_fit.applied` is False. Weakly birefringent
  tissue under the threshold still looks like residual; that is why this is
  off by default.
- It fits the normalised components `S1/S3, S2/S3` (OpenPolScope's `a, b` on
  that method) and leaves `S0`, `S3` alone, as waveorder 1.0 did. Current
  waveorder subtracts from all four, which drives a clear field to a quarter
  wave; see `polscope_library/background_fit.py`.
- It is the one non-pointwise step: each image gets its own surface, so it is
  meant per tile, and adds about 1 s to a 2048 x 2448 tile.

## Reconstruction methods

`reconstruct()` offers two, chosen with `method=`:

| | `qlipp` (default) | `openpolscope` |
|---|---|---|
| Lineage | waveorder / recOrder Stokes inversion | OpenPolScope 3.20b, 5-frame |
| Background | inverse of the background's Mueller matrix | subtraction in its normalised difference space |
| Retardance range | 0 to a quarter wave (arcsine folds back above) | 0 to a half wave (`den < 0` selects the other branch) |
| Schemes | 4-State and 5-State | 5-State only |
| `transmittance`, `depolarization` | computed | not computed (NaN) |

**Without a background the two are the same function** -- identical to machine
precision over 2.5 million reference-slide pixels. With one they differ by a
flat 0.53% on that slide, and the background step is the entire difference.
Past a quarter wave they part for real.

`openpolscope` reproduces OpenPolScope's own output **bit-for-bit** on the
reference data this package tests against (a crop of it ships in
`tests/fixtures/`, with OpenPolScope's retardance and orientation for the same
pixels). Use it to match data that was, or will be, processed in OpenPolScope.
For an exact match set `black_level` to the camera offset OpenPolScope uses --
retardance and orientation are invariant to a uniform offset except through
its clamp at zero, so it is harmless to set for `qlipp` too -- and pass the
wavelength OpenPolScope was set to, which is a typed-in setting there, not a
measured one.

`qlipp` is the better model where the two differ and stays the default.

The extinction ratio has the same split: `calibration.extinction_ratio` is
recOrder's definition, `calibration.extinction_ratio_openpolscope` is the one
OpenPolScope displays (mean over all swing states, no `+1`). Compare a number
read off the OpenPolScope screen with the latter.

## Citation

QLIPP: Guo et al., *eLife* 2020;9:e55502. https://elifesciences.org/articles/55502

The OpenPolScope method: Shribak M, Oldenbourg R. Techniques for fast and
sensitive measurements of two-dimensional birefringence distributions.
*Applied Optics* 42(16):3009-3017 (2003). https://doi.org/10.1364/AO.42.003009
