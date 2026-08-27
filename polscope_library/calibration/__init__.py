"""LC voltage calibration: finding the crystal settings that make the states.

The reconstruction half of this package begins after calibration. This half is
how the five calibrated states are found in the first place -- and, unusually
for calibration code, it can be exercised without a microscope, because
:mod:`~polscope_library.calibration.compensator` models the optics and
:class:`~polscope_library.calibration.simulator.SimulatedPolScope` stands in
for the instrument.
"""

from .compensator import (
    EXTINCTION_LCA_WAVES,
    EXTINCTION_LCB_WAVES,
    compensator_stokes,
    ideal_palette,
    retardances_for_stokes,
)
from .instrument import LCAxis, LiquidCrystalInstrument, RetardanceLimits, other_axis
from .simulator import SimulatedPolScope

__all__ = [
    "EXTINCTION_LCA_WAVES",
    "EXTINCTION_LCB_WAVES",
    "LCAxis",
    "LiquidCrystalInstrument",
    "RetardanceLimits",
    "SimulatedPolScope",
    "compensator_stokes",
    "ideal_palette",
    "other_axis",
    "retardances_for_stokes",
]
