"""Scalar parsing failures stay operational outcomes, with no proof fallback."""
from fractions import Fraction

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import (ValueClaim, VerificationBudget, assess_value,
                                 claim_from_dict, claim_to_dict, make_evidence)
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded
from nonlocalgames.games import chsh


@pytest.mark.parametrize('text, status', [('1e1000000000', 'budget_exceeded'),
                                         ('1/0', 'invalid_evidence')])
def test_local_value_is_preflighted_before_enumeration(monkeypatch, text, status):
    import nonlocalgames.analysis.classical as classical
    monkeypatch.setattr(classical, 'classical_value', lambda *a, **k: pytest.fail('must parse first'))
    game, local = chsh(), ResourceSpec('local')
    artifact = make_evidence(game, local, 'local_enumeration',
                             {'strategy': [[0, 0], [0, 0]], 'value': text})
    result = assess_value(game, ValueClaim(local, 'at_most', 1), [artifact])
    assert result.execution == status and result.truth == 'unknown'


def test_claim_threshold_decoder_bounds_exponents_and_preserves_legacy_spelling():
    game = chsh()
    claim = ValueClaim(ResourceSpec('local'), 'at_least', Fraction(3, 4))
    data = claim_to_dict(game, claim)
    data['threshold'] = '75e-2'
    assert claim_from_dict(game, data) == claim
    data['threshold'] = '1e1000000000'
    with pytest.raises(ScalarBudgetExceeded):
        claim_from_dict(game, data)
    data['threshold'] = '1/0'
    with pytest.raises(ValueError, match='denominator'):
        claim_from_dict(game, data)


def test_metadata_is_not_recursively_interpreted_as_rational_data():
    game, resource = chsh(), ResourceSpec('quantum_finite')
    artifact = make_evidence(game, resource, 'numerical_hint',
                             {'note': '1e1000000000', 'value': '1/0'})
    result = assess_value(game, ValueClaim(resource, 'at_least', 1), [artifact])
    assert result.execution == 'completed' and result.truth == 'unknown'
    assert result.numerical_hints == (artifact.artifact_id,)


def test_wire_budget_is_configurable_and_counted_as_operational_failure():
    game, local = chsh(), ResourceSpec('local')
    artifact = make_evidence(game, local, 'local_enumeration',
                             {'strategy': [[0, 0], [0, 0]], 'value': '3/4'})
    result = assess_value(game, ValueClaim(local, 'at_most', 1), [artifact],
                          budget=VerificationBudget(max_wire_scalar_bits=2))
    assert result.execution == 'budget_exceeded' and result.truth == 'unknown'
    assert assess_value(game, ValueClaim(local, 'at_most', 1), [artifact]).truth == 'proved'


def test_added_scalar_limits_preserve_existing_positional_budget_arguments():
    budget = VerificationBudget(1_000_000, 100_000, 128, 8_000_000, 1)
    assert budget.max_artifacts == 1
    assert budget.max_wire_scalar_bits == 4096
    game, local = chsh(), ResourceSpec('local')
    artifact = make_evidence(game, local, 'local_strategy', {'strategy': [[0, 0], [0, 0]]})
    result = assess_value(game, ValueClaim(local, 'at_least', 0), [artifact, artifact], budget=budget)
    assert result.execution == 'budget_exceeded'
    assert 'max_artifacts' in result.issues[0].message
