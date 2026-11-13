"""Structural syntax properties keep sampled and stored-rule scopes distinct."""
from fractions import Fraction

import pytest

from nonlocalgames import Game, ResourceSpec, Scenario
from nonlocalgames.claims import ExecutionStatus, TruthStatus, VerificationBudget
from nonlocalgames.games import chsh, ghz
from nonlocalgames.properties import OffSupportPolicy, PropertyClaim, assess_property


@pytest.mark.parametrize('constructor', [chsh, ghz])
@pytest.mark.parametrize('kind', ['acceptance', 'boolean_predicate', 'xor_predicate', 'xor_score'])
def test_binary_parity_games_are_recognized_for_multiple_party_counts(constructor, kind):
    game = constructor()
    result = assess_property(game, PropertyClaim(kind))
    assert result.truth == TruthStatus.PROVED and result.execution == ExecutionStatus.COMPLETED
    assert result.checked_entries == game.scenario.entries


@pytest.mark.parametrize('reward', [Fraction(0), Fraction(1), Fraction(1, 2), Fraction(-3, 2)])
def test_parity_score_is_not_the_same_as_an_xor_winning_predicate(reward):
    game = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: reward)
    assert assess_property(game, PropertyClaim('xor_score')).truth == TruthStatus.PROVED
    assert assess_property(game, PropertyClaim('xor_predicate')).truth == TruthStatus.DISPROVED
    acceptance = assess_property(game, PropertyClaim('acceptance'))
    assert acceptance.truth == (TruthStatus.PROVED if 0 <= reward <= 1 else TruthStatus.DISPROVED)
    boolean = assess_property(game, PropertyClaim('boolean_predicate'))
    assert boolean.truth == (TruthStatus.PROVED if reward in (0, 1) else TruthStatus.DISPROVED)


def _promised_game():
    return Game.from_rule(Scenario((2, 1), (2, 2)),
        lambda x, a: int(a[0] == a[1]) if x[0] == 0 else 2+2*a[0]+a[1],
        distribution={(0, 0): 1})


@pytest.mark.parametrize('kind', ['acceptance', 'boolean_predicate', 'xor_predicate', 'xor_score'])
def test_off_support_policy_is_explicit_and_part_of_the_property(kind):
    game = _promised_game()
    full = assess_property(game, PropertyClaim(kind))
    sampled = assess_property(game, PropertyClaim(kind, off_support='sampled'))
    assert full.truth == TruthStatus.DISPROVED
    assert full.counterexample.input_profile == (1, 0)
    assert sampled.truth == TruthStatus.PROVED and sampled.checked_entries == 4
    assert full.property_id != sampled.property_id
    assert PropertyClaim(kind).off_support == OffSupportPolicy.ALL


def test_nonbinary_outputs_are_an_exact_structural_counterexample():
    game = Game.from_rule(Scenario((1, 1), (2, 3)), lambda x, a: 1)
    result = assess_property(game, PropertyClaim('xor_score'))
    assert result.truth == TruthStatus.DISPROVED
    assert result.counterexample.party == 1 and result.checked_entries == 0


def test_structural_queries_do_not_accept_or_consume_irrelevant_evidence():
    def artifacts():
        yield object()
        pytest.fail('must not consume beyond the first unwanted artifact')
    result = assess_property(chsh(), PropertyClaim('acceptance'), artifacts())
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.UNSUPPORTED
    result = assess_property(chsh(), PropertyClaim('acceptance'), None)
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.INVALID_EVIDENCE
    result = assess_property(chsh(), PropertyClaim('acceptance'), budget=VerificationBudget(max_property_entries=15))
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.BUDGET_EXCEEDED


def test_property_definitions_reject_ambiguous_options_and_unsupported_names():
    with pytest.raises(ValueError):
        PropertyClaim('acceptance', ResourceSpec('local'))
    with pytest.raises(TypeError):
        PropertyClaim('perfect_strategy')
    with pytest.raises(ValueError):
        PropertyClaim('perfect_strategy', ResourceSpec('local'), off_support='all')
    with pytest.raises(ValueError):
        PropertyClaim('pseudotelepathy', ResourceSpec('quantum_finite'))
    with pytest.raises(ValueError):
        PropertyClaim('self_testing')
