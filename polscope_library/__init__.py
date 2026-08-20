"""Birefringence reconstruction for LC-PolScope microscopes.

Quantitative polarized-light microscopy with a liquid-crystal universal
compensator: acquire N calibrated polarization states, recover retardance and
slow-axis orientation. Numpy only -- no PyTorch, no phase reconstruction.

Quick start::

    from polscope_library import reconstruct

    result = reconstruct(
        intensities=[s0, s1, s2, s3, s4],   # in calibration order
        swing=0.03,
        wavelength_nm=549,
        background_intensities=[b0, b1, b2, b3, b4],
    )
    result.retardance_nm
    result.orientation_rad     # axial: see the note on averaging

See ``THIRD_PARTY_NOTICES.md`` for the upstream code this is derived from.
"""

from .birefringence import Birefringence, reconstruct, stokes_from_intensities
from .stokes import (
    SCHEMES,
    apply_orientation_offset,
    calculate_intensity_to_stokes_matrix,
    calculate_stokes_to_intensity_matrix,
    estimate_adr_from_stokes,
    mmul,
    mueller_from_stokes,
    radians_to_nanometers,
    stokes_after_adr,
)

__version__ = "0.1.0"

__all__ = [
    "SCHEMES",
    "apply_orientation_offset",
    "Birefringence",
    "calculate_intensity_to_stokes_matrix",
    "calculate_stokes_to_intensity_matrix",
    "estimate_adr_from_stokes",
    "mmul",
    "mueller_from_stokes",
    "radians_to_nanometers",
    "reconstruct",
    "stokes_after_adr",
    "stokes_from_intensities",
]
