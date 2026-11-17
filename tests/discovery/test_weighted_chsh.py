"""Strict advantage assessed from independently replayed rational evidence."""
from dataclasses import replace
from fractions import Fraction

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import GapClaim, VerificationBudget, assess_gap
from nonlocalgames.claims.types import EvidenceError
from nonlocalgames.discovery.weighted_chsh import MAX_GRID_DENOMINATOR, weighted_chsh_member
from nonlocalgames.games import biased_chsh


@pytest.mark.parametrize('p,denominator,local,truth', [
    (0, 16, 1, 'disproved'),
    (Fraction(1, 10), 16, Fraction(9, 10), 'disproved'),
    (Fraction(1, 4), 16, Fraction(3, 4), 'proved'),
    (Fraction(1001, 10000), 8, Fraction(8999, 10000), 'unknown'),
    (Fraction(1001, 10000), 64, Fraction(8999, 10000), 'proved'),
    (1, 16, 1, 'disproved'),
])
def test_bell_witness_and_tangent_upper_resolve_weighted_family(p, denominator, local, truth):
    budget = VerificationBudget()
    artifacts = weighted_chsh_member(p, grid_denominator=denominator, verification_budget=budget)
    claim = GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local'))
    result = assess_gap(biased_chsh(p), claim, artifacts, budget=budget)
    assert (result.truth, result.execution) == (truth, 'completed')
    quantum_interval, local_interval = result.intervals
    assert local_interval.lower == local_interval.upper == local
    if truth == 'disproved':
        assert quantum_interval.upper == local
    if truth == 'proved':
        assert quantum_interval.lower > local
    if p == Fraction(1, 10):
        assert artifacts[-1].payload['diagonal'] == ['3/20', '1/20', '3/20', '1/20']


@pytest.mark.parametrize('options', [
    {'max_strategies': 3}, {'max_matrix_size': 7}, {'max_artifacts': 2}, {'max_artifact_bytes': 100},
    {'max_wire_scalar_bits': 3}, {'max_quantum_scalar_bits': 2}, {'max_wire_scalar_chars': 2},
])
def test_member_generation_reports_verification_budget_exhaustion(options):
    with pytest.raises(EvidenceError) as caught:
        weighted_chsh_member(Fraction(1, 10), grid_denominator=8,
                             verification_budget=replace(VerificationBudget(), **options))
    assert caught.value.issue.code == 'budget_exceeded'


@pytest.mark.parametrize('p,denominator', [(True, 4), (0.25, 4), ('1/0', 4), ('6/5', 4), (0, True),
                                          (0, 0), (0, MAX_GRID_DENOMINATOR+1)])
def test_invalid_grid_and_probability_are_distinct_from_exhaustion(p, denominator):
    with pytest.raises((TypeError, ValueError)) as caught:
        weighted_chsh_member(p, grid_denominator=denominator, verification_budget=VerificationBudget())
    assert not isinstance(caught.value, EvidenceError)
