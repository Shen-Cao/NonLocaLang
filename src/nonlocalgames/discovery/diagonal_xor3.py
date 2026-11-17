"""A three-question XOR family with tunable weight on matching questions."""
from fractions import Fraction

from ..analysis import classical_value
from ..certificates.xor import verify_xor_dual
from ..certificates.xor_primal import verify_xor_gram
from ..claims import make_evidence
from ..core.game import Game, Scenario
from ..core.resources import ResourceSpec
from ..core.scalar_wire import parse_rational, scalar_string

_SCENARIO = Scenario((3, 3), (2, 2))


def diagonal_xor3_game(p):
    """Ask equal questions with total probability p. The winning parity is one iff x=y."""
    p = parse_rational(p)
    if not 0 <= p <= 1:
        raise ValueError('Diagonal question probability must lie in [0,1]')
    distribution = {x: p/3 if x[0] == x[1] else (1-p)/6 for x in _SCENARIO.inputs}
    return Game.from_rule(_SCENARIO, lambda x, a: (a[0] ^ a[1]) == (x[0] == x[1]),
                          distribution=distribution, name=f'Diagonal 3×3 XOR ({scalar_string(p)})')


def _gram(p):
    if p <= Fraction(1, 5):
        return ((1,)*6,)*6
    # Three unit vectors at 120° have pairwise inner product −1/2.
    # Alice uses their negatives, making every diagonal correlation −1.
    trine = tuple(tuple(1 if i == j else Fraction(-1, 2) for j in range(3)) for i in range(3))
    return tuple(tuple(trine[i%3][j%3] * (1 if i//3 == j//3 else -1) for j in range(6))
                 for i in range(6))


def diagonal_xor3_evidence(p):
    """Construct and verify exact certificates for the classical and quantum optima.

    The signed question-weight matrix has eigenvalues (1−2p)/3 on the constant
    vector and −(1+p)/6 on its orthogonal complement. One quarter of its spectral
    norm supplies a uniform dual diagonal in the winning-score convention.
    The two candidate Gram matrices attain the corresponding objective branches.
    """
    game = diagonal_xor3_game(p)
    p = 3*game.probability((0, 0))
    local = classical_value(game)
    point = verify_xor_gram(game, _gram(p))
    diagonal = max(abs(1-2*p), (1+p)/2) / 12
    upper = verify_xor_dual(game, (diagonal,)*6)
    if point.value != upper.value:
        raise RuntimeError('Diagonal XOR certificates have different objective values')
    return (
        make_evidence(game, ResourceSpec('local'), 'local_enumeration',
                      {'strategy': local.witness, 'value': scalar_string(local.value)}),
        make_evidence(game, ResourceSpec('quantum_finite'), 'xor_primal', point.witness),
        make_evidence(game, ResourceSpec('quantum_commuting'), 'xor_dual', upper.witness),
    )
