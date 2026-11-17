"""Rational evidence for CHSH under a one-dimensional family of question weights."""
from fractions import Fraction
from numbers import Integral

from ..analysis import classical_value, certify_quantum_strategy, verify_xor_dual
from ..claims import VerificationBudget, make_evidence, verify_artifact
from ..claims.types import EvidenceError, ExecutionStatus
from ..claims.verify import _quantum_policy
from ..core.identity import JSONBudgetExceeded
from ..core.resources import ResourceSpec
from ..core.scalar_wire import ScalarBudgetExceeded, parse_rational, scalar_string
from ..core.wire_validation import validate_game_numbers
from ..games import biased_chsh
from ..quantum.exact import ExactBudgetExceeded
from ._bell import planar_bell_strategy


WEIGHTED_CHSH_METHOD = 'nonlocalgames.weighted_chsh_rational_angles.v1'
MAX_GRID_DENOMINATOR = 1_000_000


def _cosine(index, denominator):
    return Fraction(denominator**2 - index**2, denominator**2 + index**2)


def _grid_point(p, q, denominator):
    """Maximize the bias 3(q+p)c−4pc³ over the rational half-angle grid."""
    left, right = 0, denominator
    while right - left > 1:
        middle = (left + right) // 2
        # c decreases with the index. The derivative vanishes at 4pc²=q+p.
        if 4*p*_cosine(middle, denominator)**2 > q+p:
            left = middle
        else:
            right = middle

    def bias(index):
        c = _cosine(index, denominator)
        return 3*(q+p)*c - 4*p*c**3

    index = max(sorted({0, left, right, denominator}), key=bias)
    return (_cosine(index, denominator),
            Fraction(2*index*denominator, denominator**2 + index**2))


def _tangent(p, q):
    """Dual diagonal in the checker's winning-score convention, when feasible."""
    if p <= Fraction(1, 10):
        return (q/2, (q-p)/4)*2
    if p == 1:
        return (Fraction(0), Fraction(1, 4))*2
    return None


def weighted_chsh_member(p, *, grid_denominator, verification_budget):
    """Generate raw local, finite quantum and applicable XOR upper certificates.

    The 11 question pair has probability p, and the other three pairs each have
    probability q=(1-p)/3. The two-qubit candidate measures Alice's observables
    at angles 0 and 2θ and Bob's at θ and −θ. Setting tan(θ/2)=j/D gives exact
    rational matrices. Binary search brackets the continuous stationary point,
    then exact comparison selects a grid point, choosing the smaller j on ties.
    The exported lower certificate is checked by ordinary finite Born scoring.
    """
    if not isinstance(verification_budget, VerificationBudget):
        raise TypeError('Expected VerificationBudget')
    if (isinstance(grid_denominator, bool) or not isinstance(grid_denominator, Integral)
            or not 1 <= grid_denominator <= MAX_GRID_DENOMINATOR):
        raise ValueError(f'grid_denominator must be an integer in [1,{MAX_GRID_DENOMINATOR}]')
    budget = verification_budget
    try:
        p = parse_rational(p, max_bits=budget.max_wire_scalar_bits, max_chars=budget.max_wire_scalar_chars)
        game = biased_chsh(p)
        q = (1-p)/3
        tangent = _tangent(p, q)
        if budget.max_artifacts < 2 + (tangent is not None):
            raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'Weighted CHSH evidence exceeds max_artifacts')
        if budget.max_strategies < 4:
            raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'Classical replay exceeds max_strategies')
        validate_game_numbers(game, max_scalar_bits=budget.max_wire_scalar_bits,
                              max_scalar_chars=budget.max_wire_scalar_chars)
        policy = _quantum_policy(game, budget, None)
        local = classical_value(game, max_strategies=budget.max_strategies)
        c, s = _grid_point(p, q, int(grid_denominator))
        strategy = planar_bell_strategy(((1, 0), (2*c*c-1, 2*c*s)), ((c, s), (c, -s)), budget=policy)
        point = certify_quantum_strategy(game, strategy, budget=policy, max_bytes=budget.max_artifact_bytes)
        def artifact(resource, kind, payload):
            return make_evidence(game, ResourceSpec(resource), kind, payload, max_bytes=budget.max_artifact_bytes)
        artifacts = [artifact('local', 'local_enumeration',
                              {'strategy': local.witness, 'value': scalar_string(local.value)}),
                     artifact('quantum_finite', 'quantum_strategy', point.witness)]
        if tangent is not None:
            upper = verify_xor_dual(game, tangent, max_matrix_size=budget.max_matrix_size,
                                    max_scalar_bits=budget.max_wire_scalar_bits, max_scalar_chars=budget.max_wire_scalar_chars)
            artifacts.append(artifact('quantum_commuting', 'xor_dual', upper.witness))
        for item in artifacts:
            verify_artifact(game, item, budget=budget)
        return tuple(artifacts)
    except (ExactBudgetExceeded, JSONBudgetExceeded, ScalarBudgetExceeded) as error:
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, str(error)) from error
