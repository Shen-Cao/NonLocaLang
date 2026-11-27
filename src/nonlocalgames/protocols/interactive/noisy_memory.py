"""Fixed noisy-memory channels and readouts for the trusted split-Bell-label example."""
from dataclasses import replace
from fractions import Fraction
from typing import Literal

from ...quantum.exact import RationalMatrix
from ...quantum.exact.algebra import add, identity, kron, scale, subtract
from .examples import split_bell_label
from .model import Continuation, FirstMove, KrausTerm, LocalStrategy, TwoRoundProtocol


_Channel = Literal["damping", "pauli_twirl", "phase_plus", "phase_minus"]
_Readout = Literal["bell", "memory_z", "xy_parity"]
_X = RationalMatrix(((0, 1), (1, 0)))
_Y = RationalMatrix(((0, 0), (0, 0)), ((0, -1), (1, 0)))
_Z = RationalMatrix(((1, 0), (0, -1)))


def _first_move(channel: _Channel) -> FirstMove:
    if channel == "pauli_twirl":
        # Pauli twirl of the real damping channel, not of the phased channels.
        terms = tuple(KrausTerm(Fraction(n, 169), p)
                      for n, p in ((81, identity(2)), (36, _X), (16, _Z), (36, _Y)))
    else:
        if channel == "damping":
            re, im = 5, 0
        elif channel == "phase_plus":
            re, im = 3, 4
        elif channel == "phase_minus":
            re, im = 3, -4
        else:
            raise ValueError("channel must be damping, pauli_twirl, phase_plus, or phase_minus")
        stay = RationalMatrix(((1, 0), (0, Fraction(re, 13))), ((0, 0), (0, Fraction(im, 13))))
        jump = RationalMatrix(((0, Fraction(12, 13)), (0, 0)))
        terms = (KrausTerm(1, stay), KrausTerm(1, jump))
    return FirstMove(0, (terms,))  # one acknowledgement, no observed Kraus index


def noisy_split_bell_label(
    channel: _Channel = "damping", *, readout: _Readout = "bell",
) -> tuple[TwoRoundProtocol, LocalStrategy]:
    """Return a trusted split-Bell preparation and a fixed strategy for exact replay.

    A single noisy operation stores Q1 as M. Both classical questions and the
    acknowledgement are 0. 'bell' guesses the private label. 'memory_z' reads
    Z on M and accepts the + sign. 'xy_parity' reads X on M times Y on Q2 and
    accepts the sign (+,-,-,+) for the private label (Φ+,Φ−,Ψ+,Ψ−).
    The private label selects neither the channel nor the final POVM.
    """
    protocol, ideal = split_bell_label()
    strategy = replace(ideal, first=(_first_move(channel),))
    if readout == "bell":
        return protocol, strategy
    if readout == "memory_z":
        observable, wanted = kron(identity(2), _Z), (0, 0, 0, 0)
    elif readout == "xy_parity":
        # Tensor order at the final move is Q2 ⊗ M, not arrival order.
        observable, wanted = kron(_Y, _X), (0, 1, 1, 0)
    else:
        raise ValueError("readout must be bell, memory_z, or xy_parity")

    eye = identity(4)
    effects = tuple(scale(op, Fraction(1, 2))
                    for op in (add(eye, observable), subtract(eye, observable)))
    yes_no = (identity(1), RationalMatrix(((0,),)))
    cases = tuple(replace(c, after_first=(Continuation(0, yes_no if bit == 0 else yes_no[::-1]),))
                  for c, bit in zip(protocol.cases, wanted, strict=True))
    final, = strategy.final  # a single shared local history
    strategy = replace(strategy, final=(replace(final, effects=effects),))
    return replace(protocol, cases=cases), strategy
