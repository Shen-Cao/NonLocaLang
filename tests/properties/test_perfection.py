"""Perfection is an attainment statement, with a separate classical exclusion."""
from fractions import Fraction

import pytest

from nonlocalgames import Game, ResourceSpec, Scenario
from nonlocalgames.analysis import classical_value
from nonlocalgames.certificates.quantum import certify_quantum_strategy
from nonlocalgames.certificates.xor import verify_xor_dual
from nonlocalgames.claims import (EvidenceKind, ExecutionStatus, TruthStatus,
                                  VerificationBudget, make_evidence)
from nonlocalgames.games import chsh, ghz
from nonlocalgames.properties import PropertyClaim, assess_property
from nonlocalgames.quantum.exact import exact_chsh_strategy, exact_ghz_strategy


LOCAL, QUANTUM = ResourceSpec('local'), ResourceSpec('quantum_finite')


def _local(game):
    result = classical_value(game)
    return make_evidence(game, LOCAL, EvidenceKind.LOCAL_ENUMERATION,
                          {'strategy': result.witness, 'value': str(result.value)})


def _quantum(game, strategy):
    return make_evidence(game, QUANTUM, EvidenceKind.QUANTUM_STRATEGY,
                          certify_quantum_strategy(game, strategy).witness)


def test_ghz_has_verified_finite_attainment_and_pseudotelepathy():
    game = ghz()
    artifacts = [_local(game), _quantum(game, exact_ghz_strategy())]
    q = assess_property(game, PropertyClaim('perfect_strategy', QUANTUM), artifacts)
    assert q.truth == TruthStatus.PROVED
    assert q.checks[1].attained_score == 1 and q.checks[1].actual_dimensions == (2, 2, 2)
    assert assess_property(game, PropertyClaim('perfect_strategy', LOCAL), artifacts).truth == TruthStatus.DISPROVED
    assert assess_property(game, PropertyClaim('pseudotelepathy'), artifacts).truth == TruthStatus.PROVED
    assert assess_property(game, PropertyClaim('pseudotelepathy'), artifacts[1:]).truth == TruthStatus.UNKNOWN


def test_suboptimal_quantum_point_is_neither_perfection_nor_a_perfection_disproof():
    game = chsh()
    artifacts = [_local(game), _quantum(game, exact_chsh_strategy())]
    assert assess_property(game, PropertyClaim('perfect_strategy', QUANTUM), artifacts).truth == TruthStatus.UNKNOWN
    assert assess_property(game, PropertyClaim('pseudotelepathy'), artifacts).truth == TruthStatus.UNKNOWN
    upper = make_evidence(game, ResourceSpec('quantum_commuting'), EvidenceKind.XOR_DUAL,
                           verify_xor_dual(game, [Fraction(9, 100)]*4).witness)
    for kind in ('perfect_strategy', 'pseudotelepathy'):
        claim = PropertyClaim(kind, QUANTUM if kind == 'perfect_strategy' else None)
        assert assess_property(game, claim, artifacts+[upper]).truth == TruthStatus.DISPROVED


def test_local_perfection_embeds_in_finite_quantum_and_excludes_pseudotelepathy():
    game = Game.from_rule(Scenario((2, 1), (2, 2)),
        lambda x, a: int(a[0] == a[1]) if x[0] == 0 else -7,
        distribution={(0, 0): 1})
    assert not game.is_acceptance_game
    artifacts = [_local(game)]
    for resource in (LOCAL, QUANTUM):
        result = assess_property(game, PropertyClaim('perfect_strategy', resource), artifacts)
        assert result.truth == TruthStatus.PROVED and result.checked_entries == 4
    assert assess_property(game, PropertyClaim('pseudotelepathy'), artifacts).truth == TruthStatus.DISPROVED


def test_nonacceptance_sampled_scores_do_not_define_perfect_play():
    game = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: 2)
    result = assess_property(game, PropertyClaim('perfect_strategy', LOCAL), [_local(game)])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.UNSUPPORTED
    assert result.counterexample.reason == 'reward_outside_acceptance_interval'


def test_perfection_reuses_admission_failures_and_budgets():
    game = ghz()
    artifact = _quantum(game, exact_ghz_strategy())
    result = assess_property(game, PropertyClaim('perfect_strategy', QUANTUM), [artifact],
                             budget=VerificationBudget(max_quantum_joint_dimension=4))
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.BUDGET_EXCEEDED
    for resource in (ResourceSpec('quantum_closure'), ResourceSpec('quantum_commuting'),
                     ResourceSpec('quantum_finite', (2, 2, 2))):
        result = assess_property(game, PropertyClaim('perfect_strategy', resource), [artifact])
        assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.UNSUPPORTED


def test_no_unsolicited_strategy_search_or_solver_is_started(monkeypatch):
    import nonlocalgames.analysis.classical as classical
    monkeypatch.setattr(classical, 'classical_value', lambda *args, **kwargs: pytest.fail('no evidence requested enumeration'))
    assert assess_property(chsh(), PropertyClaim('perfect_strategy', QUANTUM)).truth == TruthStatus.UNKNOWN
    assert assess_property(chsh(), PropertyClaim('pseudotelepathy')).truth == TruthStatus.UNKNOWN
