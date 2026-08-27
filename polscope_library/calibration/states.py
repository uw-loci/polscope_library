"""The calibration schemes, as data rather than as branching.

Every state-specific defect in recOrder's optimizers lives in a duplicated
``if state == "45" or state == "135"`` chain -- two classes, six branches each,
kept in step by hand. They were not: one passes ``"LCB"`` where its sibling
passes ``"LCA"``, another uses LC-A bounds for an LC-B scan, another omits an
argument. Each is individually small and individually invisible.

Describing the schemes once, as a table both searches read, does not fix those
bugs so much as make them unrepresentable. There is one place a state can say
which crystal it varies, so two searches cannot disagree about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .instrument import LCAxis

__all__ = ["SCHEMES", "SCHEME_MOVES", "StateMove", "moves_for"]


@dataclass(frozen=True)
class StateMove:
    """One state of a calibration scheme, and how to find it.

    Attributes
    ----------
    name
        Physical name, e.g. ``"I45"``. ``"ext"`` is extinction.
    channel
        Micro-Manager channel preset this state becomes, ``State0``..``State4``.
    kind
        ``extinction`` -- search both crystals for minimum transmission.
        ``open_loop`` -- no search; set the seed and measure. Its intensity
        becomes the reference every later state is matched against.
        ``scan`` -- vary one crystal to match the reference.
        ``constrained`` -- vary LC-A, with LC-B following on a fixed ratio.
    axis
        Crystal the search varies. ``None`` for extinction and open-loop.
    constraint_sign
        For constrained moves, the sign of LC-B's excursion. ``None`` otherwise.
    """

    name: str
    channel: str
    kind: str
    axis: Optional[LCAxis]
    constraint_sign: Optional[int] = None


# Extinction and I0 are shared by both schemes. I0 is deliberately open-loop:
# recOrder does not search for it, it steps to the seed and measures. That
# measurement is the reference every later state is matched to, so the swing is
# defined by this move rather than discovered.
#
# Note there are no seed values here. Each state starts from the corresponding
# row of the ideal palette, which is derived by inverting the system matrix --
# see compensator.ideal_palette. Transcribing the seeds instead, as recOrder
# does, quietly decides the answer: the cost is symmetric about extinction, so
# both +swing and -swing on a crystal give the same intensity and the search
# stays wherever it started. A seed with the wrong sign therefore converges
# perfectly onto the wrong state, and the orientation map comes out mirrored.
_EXT = StateMove("ext", "State0", "extinction", None)
_I0 = StateMove("I0", "State1", "open_loop", None)

# 5-State: four unconstrained scans, alternating crystals.
_I45 = StateMove("I45", "State2", "scan", "LCB")
_I90 = StateMove("I90", "State3", "scan", "LCA")
_I135 = StateMove("I135", "State4", "scan", "LCB")

# 4-State: two joint moves along a fixed ratio instead of the 45/90/135 trio.
# Both vary LC-A -- the fact recOrder's default optimizer gets wrong for I120,
# where it passes LC-B and then overwrites it, leaving LC-A untouched for the
# whole optimisation while recording the result as an LC-A value.
_I60 = StateMove("I60", "State2", "constrained", "LCA", constraint_sign=+1)
_I120 = StateMove("I120", "State3", "constrained", "LCA", constraint_sign=-1)

SCHEME_MOVES = {
    "4-State": (_EXT, _I0, _I60, _I120),
    "5-State": (_EXT, _I0, _I45, _I90, _I135),
}

SCHEMES = tuple(SCHEME_MOVES)


def moves_for(scheme: str):
    """States of a scheme, extinction first.

    Raises
    ------
    ValueError
        If the scheme is unknown. recOrder falls through its ``if`` chains and
        raises ``UnboundLocalError`` from deep inside the optimizer instead,
        which says nothing about the cause.
    """
    try:
        return SCHEME_MOVES[scheme]
    except KeyError:
        raise ValueError(f"unknown calibration scheme {scheme!r}; expected one of {list(SCHEME_MOVES)}") from None
