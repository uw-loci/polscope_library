"""Estimated background: a smooth residual fitted per image and removed.

Portions of this module are a port of upstream code:

    Copyright (c) 2025, Chan Zuckerberg Biohub  (waveorder)
    Licensed under the BSD 3-Clause License. Full text in
    ``third_party_licenses/waveorder-LICENSE.txt``.

What it is for
--------------
The measured background (a specimen-free field, see
:func:`polscope_library.reconstruct`) removes the instrument's residual
birefringence as it was *when and where that field was taken*. What it cannot
remove is a residual that differs from the reference: strain in a different
part of the slide or coverslip, a slightly different focus, slow drift. That
shows up as a smooth retardance ramp across each tile, different from tile to
tile, and as a visible step at every seam of a mosaic.

This module fits a low-order 2D polynomial surface to that residual in each
image and subtracts it -- recOrder's "estimated background" option.

How it is fitted (the upstream algorithm)
-----------------------------------------
The image is cut into square blocks (``block_size``, default 32 px); each
block contributes its median, which ignores small specimen features inside it;
a polynomial of total degree ``order`` (default 2) is least-squares fitted to
those medians at the block centres and evaluated at every pixel. That is
``waveorder.correction.estimate_background`` (``_sample_block_medians``,
``_grid_coordinates``, ``_fit_2d_polynomial_surface``), ported from torch to
numpy.

What it is fitted TO -- a deliberate divergence from current upstream
---------------------------------------------------------------------
Current waveorder (``inplane_oriented_thick_pol3d.apply_inverse_transfer_function``,
``remove_estimated_background=True``) subtracts a fitted surface from **all
four** Stokes parameters. That cannot be right for S0 and S3: on a clear field
both are close to their medians, so subtracting the fit drives them to zero,
and retardance -- ``arcsin(sqrt(S1^2 + S2^2) / |S|)`` -- goes to a quarter wave.
Measured on a synthetic 1-3 nm residual: clear pixels read 134 nm afterwards.

waveorder 1.0.0 (the code of the QLIPP paper, ``Polscope_bg_correction`` with
``bg_option="local_fit"``) fitted only the two **normalised** components,
``S1/S3`` and ``S2/S3``, and left ``S0`` and ``S3`` alone. That is what we do.
It is also the physically meaningful choice: a small residual retarder adds,
to first order, a constant vector to ``(S1/S3, S2/S3)`` -- ``tan(ret)`` times
the orientation's unit vector -- so a smooth residual is a smooth additive
surface there. OpenPolScope's own background step works in the same space
(``a/den - bg_a``), so on ``method="openpolscope"`` the fit acts on its
``(a, b)``.

The caveat: it cannot tell specimen from residual
-------------------------------------------------
The fit removes **anything** smooth, including real low-spatial-frequency
retardance. On a sparse sample the block medians sit on clear glass and the
fit is the residual. On a tissue section that fills the tile, the medians sit
on tissue and the fit removes tissue.

Two guards, both on by default, keep that from happening silently:

- ``max_block_retardance_nm``: a block whose median retardance is above this
  is treated as specimen and left out of the fit.
- ``min_background_fraction``: if fewer than this fraction of blocks survive,
  the tile is **not** corrected at all, and the returned
  :class:`BackgroundFit` says so. A tile full of tissue keeps its
  measured-background correction only.

Neither makes the fit safe on tissue whose own retardance is below the
threshold -- weakly birefringent tissue looks exactly like a residual. That is
why the option is off unless asked for.

It is a neighbourhood operation
-------------------------------
Everything else in this package is pointwise, which is what makes per-tile and
whole-mosaic reconstruction identical. This is not: each tile gets its own
surface, so reconstructing an assembled mosaic with it would give a different
(single, mosaic-wide) answer. Per tile is the intended use -- the residual it
targets is the per-tile one.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "BackgroundFit",
    "EstimatedBackground",
    "estimate_background",
    "fit_residual_background",
]


@dataclass(frozen=True)
class EstimatedBackground:
    """Settings for the estimated-background fit. Pass one to
    :func:`polscope_library.reconstruct` to turn the fit on.

    Attributes
    ----------
    order : int
        Total degree of the polynomial surface. 2 (upstream's default) bends
        once in each direction; 1 is a plane; 0 removes a constant.
    block_size : int
        Side of the square blocks whose medians are fitted, in pixels.
        Upstream's default is 32. A block must be bigger than the specimen
        features its median should ignore.
    max_block_retardance_nm : float or None
        Blocks whose median retardance exceeds this are treated as specimen
        and excluded from the fit. ``None`` keeps every block (upstream
        behaviour). Set it above the residual you expect to remove and below
        the specimen.
    min_background_fraction : float
        The fit is skipped, and the tile left as it was, when fewer than this
        fraction of blocks survive the retardance test.
    """

    order: int = 2
    block_size: int = 32
    max_block_retardance_nm: Optional[float] = 5.0
    min_background_fraction: float = 0.25

    def __post_init__(self):
        if int(self.order) != self.order or self.order < 0:
            raise ValueError(f"order must be a non-negative integer, got {self.order!r}")
        if int(self.block_size) != self.block_size or self.block_size < 2:
            raise ValueError(f"block_size must be an integer of at least 2, got {self.block_size!r}")
        if self.max_block_retardance_nm is not None and not self.max_block_retardance_nm > 0:
            raise ValueError(f"max_block_retardance_nm must be positive or None, got {self.max_block_retardance_nm!r}")
        if not 0.0 <= self.min_background_fraction <= 1.0:
            raise ValueError(f"min_background_fraction must be within [0, 1], got {self.min_background_fraction!r}")

    @property
    def n_coefficients(self) -> int:
        """Number of polynomial terms the fit solves for."""
        return (self.order + 1) * (self.order + 2) // 2


@dataclass(frozen=True)
class BackgroundFit:
    """What the estimated-background step did to one image.

    Attributes
    ----------
    applied : bool
        False when the fit was skipped; the image is then unchanged.
    reason : str
        One line, for logs and tile metadata.
    blocks_total, blocks_used : int
        Blocks in the image, and blocks that entered the fit.
    peak_removed_nm, mean_removed_nm : float
        Retardance of the removed surface itself, at its largest and on
        average. This is the size of the residual the measured background
        missed, and the number to compare between tiles. NaN when skipped.
    """

    applied: bool
    reason: str
    blocks_total: int
    blocks_used: int
    peak_removed_nm: float = float("nan")
    mean_removed_nm: float = float("nan")


def _block_medians(image: np.ndarray, block_size: int) -> np.ndarray:
    """Median of every whole ``block_size`` square, shape ``(rows, cols)``.

    Overhanging partial blocks at the right and bottom edges are ignored, as
    upstream does. Non-finite pixels are ignored too; a block with none left
    is NaN.
    """
    rows, cols = image.shape[0] // block_size, image.shape[1] // block_size
    blocks = image[: rows * block_size, : cols * block_size].reshape(rows, block_size, cols, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(rows, cols, block_size * block_size)
    # nanmedian warns on an all-NaN block, which is an expected case here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.where(np.isfinite(blocks), blocks, np.nan), axis=-1)


def _block_centres(shape: Tuple[int, int], block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates of each block's centre, as ``(rows, cols)`` grids.

    The centre of pixels ``0 .. block_size - 1`` is ``(block_size - 1) / 2``.
    Current upstream uses ``block_size / 2``, half a pixel off; waveorder 1.0
    used the value here.
    """
    rows, cols = shape[0] // block_size, shape[1] // block_size
    y = np.arange(rows) * block_size + (block_size - 1) / 2.0
    x = np.arange(cols) * block_size + (block_size - 1) / 2.0
    return np.meshgrid(y, x, indexing="ij")


def _terms(y: np.ndarray, x: np.ndarray, order: int) -> np.ndarray:
    """Polynomial terms ``y**i * x**j`` with ``i + j <= order``, last axis."""
    return np.stack([y**i * x**j for i in range(order + 1) for j in range(order + 1 - i)], axis=-1)


def _fit_surface(y: np.ndarray, x: np.ndarray, values: np.ndarray, order: int, shape: Tuple[int, int]) -> np.ndarray:
    """Least-squares polynomial through ``values`` at ``(y, x)``, evaluated on
    the full ``shape`` grid.

    Coordinates are scaled to ``[-1, 1]`` before fitting. That does not change
    the fitted surface (the polynomial space is the same) but keeps the
    normal equations well conditioned; upstream fits raw pixel coordinates,
    whose fourth powers on a 2048 px tile reach 1.8e13.
    """
    sy = (shape[0] - 1) / 2.0 or 1.0
    sx = (shape[1] - 1) / 2.0 or 1.0
    coeffs, *_ = np.linalg.lstsq(_terms(y / sy - 1.0, x / sx - 1.0, order), values, rcond=None)
    gy = np.arange(shape[0], dtype=np.float64) / sy - 1.0
    gx = np.arange(shape[1], dtype=np.float64) / sx - 1.0
    # Evaluate separably: sum_k c_k * y^i_k * x^j_k as an outer product per
    # term, which avoids building a (H, W, n_terms) array.
    surface = np.zeros(shape, dtype=np.float64)
    k = 0
    for i in range(order + 1):
        yi = gy**i
        for j in range(order + 1 - i):
            surface += coeffs[k] * np.outer(yi, gx**j)
            k += 1
    return surface


def estimate_background(
    image: np.ndarray,
    order: int = 2,
    block_size: int = 32,
    block_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Smooth polynomial background of a 2D image, evaluated at every pixel.

    The numpy port of ``waveorder.correction.estimate_background``: block
    medians, fitted with a polynomial of total degree ``order``.

    Parameters
    ----------
    image : 2D array
    order : int
    block_size : int
    block_mask : bool array of shape ``(rows, cols)``, optional
        Which blocks to fit. Not in upstream; ``None`` fits every block with a
        finite median, which is upstream's behaviour.

    Returns
    -------
    numpy.ndarray
        Same shape as ``image``.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if block_size > image.shape[0] or block_size > image.shape[1]:
        raise ValueError(f"block_size {block_size} is larger than the image {image.shape}")
    medians = _block_medians(image, block_size)
    y, x = _block_centres(image.shape, block_size)
    keep = np.isfinite(medians)
    if block_mask is not None:
        keep &= np.asarray(block_mask, dtype=bool)
    n_coeffs = (order + 1) * (order + 2) // 2
    if keep.sum() <= n_coeffs:
        raise ValueError(f"cannot fit a degree-{order} 2D polynomial to {int(keep.sum())} blocks")
    return _fit_surface(y[keep], x[keep], medians[keep], order, image.shape)


def fit_residual_background(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    retardance_scale: float,
    wavelength_nm: float,
    settings: EstimatedBackground,
) -> Tuple[np.ndarray, np.ndarray, BackgroundFit]:
    """Fit and subtract a smooth residual from the two normalised components.

    Parameters
    ----------
    u, v : 2D arrays
        The components a residual adds to: ``S1/S3, S2/S3`` (QLIPP) or
        OpenPolScope's ``a, b``.
    valid : 2D bool array
        Pixels that may inform the fit. Pixels past a quarter wave (``S3 <= 0``,
        or OpenPolScope's ``den <= 0``) must be excluded: there ``u, v`` are not
        small and do not add.
    retardance_scale : float
        ``ret = arctan(retardance_scale * hypot(u, v))``: 1 for QLIPP's
        ``S1/S3``, ``tan(pi * swing)`` for OpenPolScope's ``a, b``. Used to
        judge which blocks are specimen and to report what was removed.
    wavelength_nm : float
    settings : EstimatedBackground

    Returns
    -------
    (u_corrected, v_corrected, BackgroundFit)
        When the fit is skipped the inputs come back unchanged.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.ndim != 2 or u.shape != v.shape:
        raise ValueError(
            f"the estimated background is fitted per 2D image; got components of shape {u.shape} and {v.shape}"
        )
    bs = int(settings.block_size)
    if bs > u.shape[0] or bs > u.shape[1]:
        return u, v, BackgroundFit(False, f"skipped: image {u.shape} is smaller than one {bs} px block", 0, 0)

    valid = np.asarray(valid, dtype=bool) & np.isfinite(u) & np.isfinite(v)
    u_in = np.where(valid, u, np.nan)
    v_in = np.where(valid, v, np.nan)

    def to_nm(a, b):
        return np.arctan(retardance_scale * np.hypot(a, b)) * wavelength_nm / (2.0 * np.pi)

    mu = _block_medians(u_in, bs)
    mv = _block_medians(v_in, bs)
    # At least half the block must be usable, or its median speaks for too
    # few pixels to anchor a surface.
    usable = _block_medians(valid.astype(np.float64), bs) >= 0.5
    keep = usable & np.isfinite(mu) & np.isfinite(mv)
    if settings.max_block_retardance_nm is not None:
        block_ret = _block_medians(to_nm(u_in, v_in), bs)
        keep &= block_ret <= float(settings.max_block_retardance_nm)

    total = int(mu.size)
    used = int(keep.sum())
    needed = max(settings.n_coefficients + 1, int(np.ceil(settings.min_background_fraction * total)))
    if used < needed:
        threshold = (
            f" at or below {settings.max_block_retardance_nm:g} nm"
            if settings.max_block_retardance_nm is not None
            else ""
        )
        return (
            u,
            v,
            BackgroundFit(
                False,
                f"skipped: {used} of {total} blocks look like background{threshold}; "
                f"need {needed} ({settings.min_background_fraction:.0%}). The tile is probably mostly specimen.",
                total,
                used,
            ),
        )

    y, x = _block_centres(u.shape, bs)
    fu = _fit_surface(y[keep], x[keep], mu[keep], settings.order, u.shape)
    fv = _fit_surface(y[keep], x[keep], mv[keep], settings.order, u.shape)
    removed = to_nm(fu, fv)
    return (
        u - fu,
        v - fv,
        BackgroundFit(
            True,
            f"applied: degree-{settings.order} surface from {used} of {total} blocks",
            total,
            used,
            float(np.max(removed)),
            float(np.mean(removed)),
        ),
    )
