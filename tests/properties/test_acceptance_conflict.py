"""Impossible acceptance evidence is quarantined, without rejecting loose bounds."""
from fractions import Fraction

import pytest

from nonlocalgames import Game, ResourceSpec
from nonlocalgames.claims.admission import EvidenceBundle, bound_intervals
from nonlocalgames.claims.types import EvidenceCheck, ExecutionStatus, TruthStatus
from nonlocalgames.core.identity import verifier_id
from nonlocalgames.core.resources import Resource
from nonlocalgames.games import chsh
from nonlocalgames.properties import PropertyClaim, assess_property


def _mock_admission(monkeypatch, checks, *, intervals=None, game=None):
    import nonlocalgames.claims.admission as admission
    game = chsh() if game is None else game
    computed, rules = bound_intervals(checks)
    bundle = EvidenceBundle(verifier_id(game), ExecutionStatus.COMPLETED,
        computed if intervals is None else intervals, tuple(checks),
        tuple(c.artifact_id for c in checks if c.is_proof),
        tuple(c.artifact_id for c in checks if not c.is_proof), rules)
    monkeypatch.setattr(admission, 'verify_evidence_bundle', lambda *args, **kwargs: bundle)
    return game


@pytest.mark.parametrize('fields', [
    {'lower': Fraction(2), 'upper': None},
    {'lower': None, 'upper': Fraction(-1)},
    {'lower': None, 'upper': None, 'attained_score': Fraction(-1)},
    {'lower': None, 'upper': None, 'attained_score': Fraction(2)},
])
@pytest.mark.parametrize('kind', ['perfect_strategy', 'pseudotelepathy'])
def test_impossible_checked_acceptance_facts_are_never_mathematical_disproof(monkeypatch, fields, kind):
    check = EvidenceCheck('bad_checker_output', Resource.QUANTUM_FINITE,
                           checker='synthetic_fault.v1', **fields)
    # Deliberately model a future checker/aggregation defect, not admitted user evidence.
    game = _mock_admission(monkeypatch, (check,), intervals=())
    claim = PropertyClaim(kind, ResourceSpec('quantum_finite') if kind == 'perfect_strategy' else None)
    result = assess_property(game, claim)
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.CONFLICT
    assert result.issues[0].evidence_id == 'bad_checker_output'
    assert '[0,1]' in result.issues[0].message


def test_loose_lower_and_upper_remain_valid_but_uninformative(monkeypatch):
    check = EvidenceCheck('loose', Resource.QUANTUM_FINITE, Fraction(-2), Fraction(3), 'loose_bound.v1')
    game = _mock_admission(monkeypatch, (check,))
    result = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')))
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.COMPLETED


def test_loose_upper_does_not_invalidate_checked_attainment(monkeypatch):
    checks = (EvidenceCheck('point', Resource.LOCAL, Fraction(1), None, 'point.v1', attained_score=Fraction(1)),
              EvidenceCheck('loose', Resource.NONSIGNALLING, None, Fraction(3), 'loose_bound.v1'))
    constant = Game.from_rule(chsh().scenario, lambda x, a: 1)
    game = _mock_admission(monkeypatch, checks, game=constant)
    result = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')))
    assert result.truth == TruthStatus.PROVED and result.execution == ExecutionStatus.COMPLETED


def test_out_of_range_untrusted_hints_do_not_become_a_conflict():
    from nonlocalgames.claims import EvidenceKind, make_evidence
    game = chsh()
    hint = make_evidence(game, ResourceSpec('quantum_finite'), EvidenceKind.NUMERICAL_HINT,
                         {'attained_score': '-999', 'lower': '2', 'upper': '-1', 'guarantee': 'exact'})
    result = assess_property(game, PropertyClaim('pseudotelepathy'), [hint])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.COMPLETED


def test_zero_upper_and_zero_attained_score_are_in_range(monkeypatch):
    game = Game.from_rule(chsh().scenario, lambda x, a: 0)
    upper = EvidenceCheck('zero_upper', Resource.NONSIGNALLING, None, Fraction(0), 'zero_bound.v1')
    _mock_admission(monkeypatch, (upper,), game=game)
    result = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')))
    assert result.truth == TruthStatus.DISPROVED and result.execution == ExecutionStatus.COMPLETED
    point = EvidenceCheck('zero_point', Resource.QUANTUM_FINITE, None, None, 'point.v1', attained_score=Fraction(0))
    _mock_admission(monkeypatch, (point,), game=game)
    result = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')))
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.COMPLETED
