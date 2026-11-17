from fractions import Fraction

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import (ValueClaim, VerificationBudget, assess_value, make_evidence,
                                 verify_artifact)
from nonlocalgames.discovery.diagonal_xor3 import diagonal_xor3_evidence, diagonal_xor3_game


def test_gram_realization_certifies_both_capped_thresholds():
    p = Fraction(1, 3)
    game, artifacts = diagonal_xor3_game(p), diagonal_xor3_evidence(p)
    checked = verify_artifact(game, artifacts[1])
    assert checked.actual_dimensions == (2, 2) and checked.attained_score == Fraction(5, 6)
    for relation in ('at_least', 'at_most'):
        query = ValueClaim(ResourceSpec('quantum_finite', (2, 2)), relation, '5/6')
        assert assess_value(game, query, artifacts).truth == 'proved'
    query = ValueClaim(ResourceSpec('quantum_finite', (1, 1)), 'at_least', '5/6')
    assert assess_value(game, query, artifacts).truth == 'unknown'


@pytest.mark.parametrize('source', [ResourceSpec('nonsignalling'), ResourceSpec('quantum_commuting'),
                                   ResourceSpec('quantum_finite', (3, 3))])
def test_wrong_or_restricted_gram_source_is_refused(source):
    game, raw = diagonal_xor3_game(0), diagonal_xor3_evidence(0)[1]
    rebound = make_evidence(game, source, 'xor_primal', raw.payload)
    result = assess_value(game, ValueClaim(ResourceSpec('local'), 'at_most', 1), [rebound])
    assert result.execution == 'unsupported' and result.truth == 'unknown'


def test_gram_checks_matrix_budget_and_leaves_no_classical_lower_bound():
    p = Fraction(1, 4)
    game, raw = diagonal_xor3_game(p), diagonal_xor3_evidence(p)[1]
    query = ValueClaim(ResourceSpec('local'), 'at_least', 0)
    assert assess_value(game, query, [raw]).intervals[0].lower is None
    limited = assess_value(game, query, [raw], budget=VerificationBudget(max_matrix_size=5))
    assert limited.execution == 'budget_exceeded' and limited.truth == 'unknown'
