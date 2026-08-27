"""The boundary between the calibration algorithm and a microscope.

Everything above this line is arithmetic and can be tested on a laptop;
everything below it drives hardware. recOrder draws no such line -- its
calibration object holds the Micro-Manager core, the snap manager, a device
property table and the optimizer all at once, which is why none of its
calibration logic can be exercised without a microscope, and why the defects
this package exists to fix went unnoticed for so long.

Four methods, plus one that may be absent. Units are **waves everywhere at this
boundary**, in every control mode. Volts exist only inside the voltage-mode
implementation and in whatever writes the final presets; the algorithm never
sees them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

__all__ = ["LCAxis", "LiquidCrystalInstrument", "RetardanceLimits", "other_axis"]

LCAxis = Literal["LCA", "LCB"]


def other_axis(axis: LCAxis) -> LCAxis:
    """The crystal that is not this one."""
    if axis == "LCA":
        return "LCB"
    if axis == "LCB":
        return "LCA"
    raise ValueError(f"unknown LC axis {axis!r}; expected 'LCA' or 'LCB'")


@dataclass(frozen=True)
class RetardanceLimits:
    """Travel of one liquid crystal, in waves.

    Defaults match the range recOrder enforces (``core_functions.py:106-110``).
    """

    min_waves: float = 0.001
    max_waves: float = 1.600

    def clamp(self, waves: float) -> float:
        """Bring a commanded retardance inside the achievable range."""
        return min(max(float(waves), self.min_waves), self.max_waves)

    def contains(self, waves: float) -> bool:
        return self.min_waves <= float(waves) <= self.max_waves


@runtime_checkable
class LiquidCrystalInstrument(Protocol):
    """What a calibration needs from a polarizing microscope."""

    @property
    def limits(self) -> RetardanceLimits:
        """Travel of the crystals, used to bound the search."""

    def set_retardance(self, axis: LCAxis, waves: float) -> None:
        """Command one crystal, in waves, and return once it has settled.

        **Must clamp to** :attr:`limits` **rather than raise.** recOrder raises
        here, and a ratio-constrained 4-State move computes its partner
        retardance from the free one -- so a search that wanders near a rail
        aborts the whole calibration part-way through, losing the states
        already solved. Clamping degrades one measurement instead; the search
        sees a poor cost there and moves away on its own.
        """

    def get_retardance(self, axis: LCAxis) -> float:
        """Read back what the crystal is ACTUALLY at, in waves.

        Not the commanded value. In voltage mode the command round-trips
        through a retardance-to-voltage curve and back, and the hardware
        quantises; the palette we record has to be what the instrument will
        reproduce tomorrow, not what we asked for today.
        """

    def measure_intensity(self) -> float:
        """Snap one frame and return its mean, black level NOT subtracted."""

    def measure_dark(self) -> float:
        """Mean of a frame with the illumination off, for the black level.

        Implementations that cannot darken the field raise
        :class:`NotImplementedError`, and the caller falls through to the next
        source. Never blocks for a human: recOrder asks the operator to close a
        shutter by hand and waits on stdin, which on a rig with no shutter is
        an unanswerable question asked from a worker thread.
        """
