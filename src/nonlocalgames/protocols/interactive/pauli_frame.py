"""Entanglement swapping with a classical follow-up and checked Pauli-frame parity."""
from fractions import Fraction
from typing import Literal

from ...quantum.exact import RationalMatrix
from ...quantum.exact.algebra import identity, kron, scale
from .model import (
    Continuation, FinalMove, FirstMove, KrausTerm, LocalHistory, LocalStrategy,
    PrivateCase, TwoRoundProtocol,
)


_BIT_PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))
_Scoring = Literal["frame", "zero", "swapped"]


def _bell_projector(p: int, q: int) -> RationalMatrix:
    # (I ⊗ X^p Z^q)|Φ+>, before the common 1/sqrt(2) normalization.
    sign = 1 - 2*q
    ket = (1, 0, 0, sign) if p == 0 else (0, 1, sign, 0)
    return RationalMatrix(tuple(tuple(Fraction(a*b, 2) for b in ket) for a in ket))


def _verifier(scoring: _Scoring) -> TwoRoundProtocol:
    phi = _bell_projector(0, 0)
    state = kron(phi, phi)  # A,L form the first pair; R,B the second.
    tests = (RationalMatrix(((0,),)), RationalMatrix(((1,),)))
    cases = []
    for c in range(2):
        follow = []
        for p, q in _BIT_PAIRS:
            b = c ^ p
            parity = (p, q)[b]
            if scoring == "zero":
                parity = 0
            elif scoring == "swapped":
                parity = (q, p)[b]
            follow.append(Continuation(b, tuple(tests[int((a ^ d) == parity)]
                                                for a, d in _BIT_PAIRS)))
        cases.append(PrivateCase(Fraction(1, 2), 0, state, tuple(follow)))
    return TwoRoundProtocol(1, 16, tuple(cases))


def _strategy(stale_right: bool) -> LocalStrategy:
    i2, i4 = identity(2), identity(4)
    outcomes = tuple((KrausTerm(1, kron(kron(i2, _bell_projector(p, q)), i2)),)
                     for p, q in _BIT_PAIRS)
    z = (RationalMatrix(((1, 0), (0, 0))), RationalMatrix(((0, 0), (0, 1))))
    x = tuple(scale(RationalMatrix(((1, s), (s, 1))), Fraction(1, 2)) for s in (1, -1))
    bases = (z, x)

    readouts = {}
    final = []
    for p, q in _BIT_PAIRS:
        for b in range(2):
            # Only the history is available here: b XOR p recovers c *after* y.
            right = b ^ p if stale_right else b
            if (b, right) not in readouts:
                readouts[b, right] = tuple(kron(kron(bases[b][a], i4), bases[right][d])
                                           for a, d in _BIT_PAIRS)
            final.append(FinalMove(LocalHistory(0, 2*p + q, b), readouts[b, right]))
    return LocalStrategy((FirstMove(0, outcomes),), tuple(final))


def adaptive_pauli_frame(
    *, stale_right: bool = False, scoring: _Scoring = "frame",
) -> tuple[TwoRoundProtocol, LocalStrategy]:
    """Build one four-qubit verifier and a supplied strategy for exact replay.

    In A,L,R,B tensor order, the prover receives Φ+ on AL and RB. It measures
    LR in the Bell basis and reports y=2p+q (p=ZZ parity, q=XX parity). The
    verifier samples a private bit c before y and sends b=c XOR p afterwards.
    The prover reads A and B in basis b (0=Z, 1=X) and reports 2a+d.

    stale_right changes only B's readout basis to b XOR p. scoring changes
    only the acceptance test: "frame" checks a XOR d against (p,q)[b],
    "zero" against zero, and "swapped" against (q,p)[b]. The two endpoints
    are registers of one prover, not independently restricted parties.
    """
    if type(stale_right) is not bool:
        raise TypeError("stale_right must be a boolean")
    if scoring not in ("frame", "zero", "swapped"):
        raise ValueError("scoring must be frame, zero, or swapped")
    return _verifier(scoring), _strategy(stale_right)
