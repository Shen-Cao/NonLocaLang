"""Exact strategy and bound certificates for the three uniform 3×3 XOR orbits."""
from fractions import Fraction

from ..analysis import classical_value, certify_quantum_strategy, verify_xor_dual
from ..claims import make_evidence
from ..core.resources import ResourceSpec
from ..core.scalar_wire import scalar_string
from ..quantum.exact import ExactQuantumStrategy, RationalMatrix
from ..quantum.exact.algebra import add, identity, kron, scale, subtract
from .xor3 import xor3_game


def _bell_strategy(alice, bob):
    """Measure the normalized sum of |j,j> with the supplied ±1 observables."""
    dimension = alice[0].dimension
    size = dimension**2
    occupied = range(0, size, dimension + 1)
    density = RationalMatrix(tuple(tuple(Fraction(1, dimension) if i in occupied and j in occupied
                                         else 0 for j in range(size)) for i in range(size)))
    unit = identity(dimension)
    half = Fraction(1, 2)

    def measurements(observables):
        return tuple((scale(add(unit, op), half), scale(subtract(unit, op), half))
                     for op in observables)

    return ExactQuantumStrategy(density, (measurements(alice), measurements(bob)))


def _qubit_strategy(mask):
    """A Bloch vector (c,s) specifies the real observable cZ+sX."""
    z = (1, 0)
    tilted = (Fraction(3, 5), Fraction(4, 5))
    if mask == 1:
        alice = (z, z, tilted)
        bob = ((Fraction(24, 25), Fraction(7, 25)),) * 2 + ((Fraction(4, 5), Fraction(-3, 5)),)
    else:
        alice = bob = (z, tilted, (tilted[0], -tilted[1]))

    def observables(vectors):
        return tuple(RationalMatrix(((c, s), (s, -c))) for c, s in vectors)

    return _bell_strategy(observables(alice), observables(bob))


def _optimal_strategy_10():
    """Four anticommuting Pauli strings realize the optimal rational Gram matrix."""
    x = RationalMatrix(((0, 1), (1, 0)))
    y = RationalMatrix(((0, 0), (0, 0)), ((0, -1), (1, 0)))
    z = RationalMatrix(((1, 0), (0, -1)))
    first = kron(x, identity(2))
    other_three = add(kron(y, identity(2)), add(kron(z, x), kron(z, y)))
    half = Fraction(1, 2)
    alice = (first, scale(add(first, other_three), half),
             scale(subtract(first, other_three), half))
    # On the Bell state, <A ⊗ B> = tr(A Bᵀ)/d.
    bob = tuple(RationalMatrix(zip(*op.real), zip(*op.imag)) for op in alice)
    return _bell_strategy(alice, bob)


def _dual_artifact(game, diagonal):
    bound = verify_xor_dual(game, (diagonal,) * 6)
    return make_evidence(game, ResourceSpec('quantum_commuting'), 'xor_dual', bound.witness)


def representative_evidence(mask: int, *, tight: bool = False):
    """Return raw certificates for enumerated local value and finite quantum gap.

    The default evidence for masks 1 and 10 uses two-qubit strategies. For mask
    10, tight=True includes a four-dimensional strategy and a matching upper
    certificate. Mask 0 already has matching bounds. Mask 1 supports only a
    strict-advantage certificate.
    """
    game = xor3_game(mask)
    if mask not in (0, 1, 10):
        raise ValueError('Evidence is defined only for XOR representatives 0, 1, and 10')
    if not isinstance(tight, bool):
        raise TypeError('tight must be a Boolean')
    if tight and mask == 1:
        raise NotImplementedError('Mask 1 has no matching optimum certificate')
    local = classical_value(game)
    local_artifact = make_evidence(game, ResourceSpec('local'), 'local_enumeration',
                                   {'strategy': local.witness, 'value': scalar_string(local.value)})
    if mask == 0:
        return (local_artifact, _dual_artifact(game, Fraction(1, 12)))
    strategy = _optimal_strategy_10() if tight else _qubit_strategy(mask)
    quantum = certify_quantum_strategy(game, strategy)
    point = make_evidence(game, ResourceSpec('quantum_finite'), 'quantum_strategy', quantum.witness)
    if tight:
        return (local_artifact, point, _dual_artifact(game, Fraction(1, 18)))
    return (local_artifact, point)
