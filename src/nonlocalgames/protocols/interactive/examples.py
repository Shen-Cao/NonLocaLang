"""Small exact verifier/strategy pairs, not calls to an optimizer."""
from fractions import Fraction

from ...quantum.exact import RationalMatrix
from ...quantum.exact.algebra import identity, scale
from .model import (
    Continuation, FinalMove, FirstMove, KrausTerm, LocalHistory, LocalStrategy,
    PrivateCase, TwoRoundProtocol,
)


def _x_effects():
    return tuple(scale(RationalMatrix(((1, s), (s, 1))), Fraction(1, 2)) for s in (1, -1))


def _output_bit(bit, dimension):
    ident = identity(dimension)
    return tuple(scale(ident, int(z == bit)) for z in range(2))


def sign_discrimination(*, reveal=False, dephase=False):
    """A hidden equiprobable sign in a qubit; reveal and Z-dephase are physical controls."""
    if type(reveal) is not bool or type(dephase) is not bool:
        raise TypeError("reveal and dephase must be booleans")
    cases, first, final = [], [], []
    for bit, s in enumerate((1, -1)):
        off = 0 if dephase else 2*s
        state = scale(RationalMatrix(((4, off), (off, 1))), Fraction(1, 5))
        x = bit if reveal else 0  # revealing *changes the verifier*
        cases.append(PrivateCase(Fraction(1, 2), x, state, (Continuation(0, _output_bit(bit, 1)),)))
    for x in range(2 if reveal else 1):
        first.append(FirstMove(x, ((KrausTerm(1, identity(2)),),)))
        effects = _output_bit(x, 2) if reveal else _x_effects()
        final.append(FinalMove(LocalHistory(x, 0, 0), effects))
    return TwoRoundProtocol(1, 2, tuple(cases)), LocalStrategy(tuple(first), tuple(final))


def correlated_memory(mode="matched"):
    """Bell reference with identical averaged channels and first-bit laws in all controls."""
    i = identity(2)
    z = RationalMatrix(((1, 0), (0, -1)))
    half = Fraction(1, 2)
    if mode == "matched" or mode == "swapped":
        pair = (i, z) if mode == "matched" else (z, i)
        outcomes = tuple((KrausTerm(half, op),) for op in pair)
    elif mode == "independent":
        outcomes = ((KrausTerm(half/2, i), KrausTerm(half/2, z)),)*2
    else:
        raise ValueError("mode must be matched, swapped, or independent")
    phi = scale(RationalMatrix(((1, 0, 0, 1), (0, 0, 0, 0),
                                (0, 0, 0, 0), (1, 0, 0, 1))), half)
    x = _x_effects()
    # Send y after commitment, then compare X outcomes: reference bit = z XOR y.
    after = tuple(Continuation(y, tuple(x[answer ^ y] for answer in range(2))) for y in range(2))
    protocol = TwoRoundProtocol(2, 2, (PrivateCase(1, 0, phi, after),))
    strategy = LocalStrategy((FirstMove(0, outcomes),),
                             tuple(FinalMove(LocalHistory(0, y, y), x) for y in range(2)))
    return protocol, strategy


def _basis_state(basis, bit):
    sign = 1 if bit == 0 else -1
    x = 5*sign*(1 if basis == 0 else -1)
    z = 12*sign
    return scale(RationalMatrix(((13+z, x), (x, 13-z))), Fraction(1, 26))


def delayed_basis(*, keep_qubit=True):
    """Reveal a hidden basis after a first move: retain the qubit or measure Z and reset."""
    if type(keep_qubit) is not bool:
        raise TypeError("keep_qubit must be a boolean")
    cases = []
    for basis in range(2):
        for bit in range(2):
            cont = Continuation(basis, _output_bit(bit, 1))
            cases.append(PrivateCase(Fraction(1, 4), 0, _basis_state(basis, bit), (cont, cont)))
    if keep_qubit:
        outcomes = ((KrausTerm(1, identity(2)),), ())  # explicit zero CP map for answer 1
    else:
        outcomes = tuple((KrausTerm(1, RationalMatrix(((1-y, y), (0, 0)))),) for y in range(2))
    final = tuple(FinalMove(LocalHistory(0, y, basis),
                            tuple(_basis_state(basis, z) for z in range(2))
                            if keep_qubit else _output_bit(y, 2))
                  for y in range(2) for basis in range(2))
    return (TwoRoundProtocol(1, 2, tuple(cases)), LocalStrategy((FirstMove(0, outcomes),), final))
