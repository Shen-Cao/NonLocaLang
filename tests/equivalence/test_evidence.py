from dataclasses import replace
from fractions import Fraction
import json

import pytest

from nonlocalgames import Game, ResourceSpec, Scenario
from nonlocalgames.analysis import certify_quantum_strategy, classical_value
from nonlocalgames.certificates.xor import verify_xor_dual
from nonlocalgames.claims import (GapClaim, ValueClaim, VerificationBudget, assess_gap,
                                 assess_value, make_evidence, verify_artifact)
from nonlocalgames.claims.types import EvidenceError
from nonlocalgames.core.identity import verifier_id
from nonlocalgames.equivalence.evidence import relabel_evidence, transport_receipt_to_dict
from nonlocalgames.equivalence.finite import relabel_game
from nonlocalgames.equivalence.maps import Relabelling
from nonlocalgames.games import chsh, ghz
from nonlocalgames.properties import PropertyClaim, assess_property
from nonlocalgames.quantum.exact import exact_chsh_strategy, exact_ghz_strategy


def _mapping(game):
    parties = game.scenario.parties
    permutation = tuple(range(1, parties)) + (0,)
    return Relabelling(game.scenario, permutation,
                       tuple((1, 0) for _ in range(parties)),
                       tuple(((1, 0), (0, 1)) for _ in range(parties)))


def _local(game, kind='local_strategy'):
    result = classical_value(game)
    payload = {'strategy': result.witness}
    if kind == 'local_enumeration':
        payload['value'] = str(result.value)
    return make_evidence(game, ResourceSpec('local'), kind, payload)


def test_local_transport_rechecks_both_sides_and_receipt_is_not_proof(monkeypatch):
    import nonlocalgames.analysis.classical as classical
    game = chsh()
    artifact, mapping = _local(game), _mapping(game)
    monkeypatch.setattr(classical, 'classical_value', lambda *a, **k: pytest.fail('no search for strategy transport'))
    transported = relabel_evidence(game, artifact, mapping)
    target = relabel_game(game, mapping)
    assert verify_artifact(target, transported.artifact).attained_score == Fraction(3, 4)
    record = json.loads(json.dumps(transport_receipt_to_dict(transported.receipt)))
    assert record['source_evidence_id'] == artifact.artifact_id
    assert record['target_game_id'] == verifier_id(target)
    assert record['source_resource_id'] == record['target_resource_id']
    claim = ValueClaim(ResourceSpec('local'), 'at_least', Fraction(3, 4))
    assert assess_value(target, claim, [transported.artifact]).truth == 'proved'
    assert assess_value(target, claim, [transported.receipt]).execution == 'invalid_evidence'
    restored = relabel_evidence(target, transported.artifact, mapping.inverse()).artifact
    assert restored.artifact_id == artifact.artifact_id


def test_local_enumeration_replays_explicitly_and_retains_both_bounds(monkeypatch):
    import nonlocalgames.analysis.classical as classical
    game = chsh()
    artifact, mapping = _local(game, 'local_enumeration'), _mapping(game)
    original, calls = classical.classical_value, []
    def replay(game, **kwargs):
        calls.append(verifier_id(game))
        return original(game, **kwargs)
    monkeypatch.setattr(classical, 'classical_value', replay)
    transported = relabel_evidence(game, artifact, mapping)
    target = relabel_game(game, mapping)
    assert calls == [verifier_id(game), verifier_id(target)]
    check = verify_artifact(target, transported.artifact)
    assert check.lower == check.upper == Fraction(3, 4)


def test_exact_chsh_gap_transports_without_any_numerical_optimizer(monkeypatch):
    import scipy.optimize
    import cvxpy as cp
    game, mapping = chsh(), _mapping(chsh())
    lower = certify_quantum_strategy(game, exact_chsh_strategy())
    upper = verify_xor_dual(game, [Fraction(9, 100)]*4)
    artifacts = [_local(game, 'local_enumeration'),
                 make_evidence(game, ResourceSpec('quantum_finite'), 'quantum_strategy', lower.witness),
                 make_evidence(game, ResourceSpec('quantum_commuting'), 'xor_dual', upper.witness)]
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *a, **k: pytest.fail('no optimizer'))
    monkeypatch.setattr(cp.Problem, 'solve', lambda *a, **k: pytest.fail('no optimizer'))
    transported = [relabel_evidence(game, artifact, mapping).artifact for artifact in artifacts]
    target = relabel_game(game, mapping)
    result = assess_gap(target, GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local')), transported)
    assert result.truth == 'proved' and result.execution == 'completed'
    assert verify_artifact(target, transported[2]).upper == upper.value


def test_ghz_perfect_strategy_survives_party_and_input_dependent_output_maps():
    game, mapping = ghz(), _mapping(ghz())
    original = certify_quantum_strategy(game, exact_ghz_strategy())
    quantum = make_evidence(game, ResourceSpec('quantum_finite'), 'quantum_strategy', original.witness)
    transported = [relabel_evidence(game, artifact, mapping).artifact
                   for artifact in (_local(game, 'local_enumeration'), quantum)]
    result = assess_property(relabel_game(game, mapping), PropertyClaim('pseudotelepathy'), transported)
    assert result.truth == 'proved' and result.execution == 'completed'


def test_xor_diagonal_follows_party_question_nodes_on_unequal_alphabets():
    game = Game.from_rule(Scenario((2, 3), (2, 2)),
                          lambda x, a: (a[0] ^ a[1]) == ((x[0]+x[1]) % 2))
    mapping = Relabelling(game.scenario, (1, 0), ((2, 0, 1), (1, 0)),
                          (((1, 0), (0, 1), (1, 0)), ((0, 1), (1, 0))))
    source = verify_xor_dual(game, [1, 2, 3, 4, 5])
    artifact = make_evidence(game, ResourceSpec('quantum_commuting'), 'xor_dual', source.witness)
    target = relabel_evidence(game, artifact, mapping).artifact
    assert target.payload['diagonal'] == ['5', '3', '4', '2', '1']
    assert verify_artifact(relabel_game(game, mapping), target).upper == source.value


def test_damaged_or_wrong_source_evidence_cannot_be_rebound():
    game, mapping = chsh(), _mapping(chsh())
    damaged = make_evidence(game, ResourceSpec('local'), 'local_enumeration',
                            {'strategy': [[0, 0], [0, 0]], 'value': '1'})
    with pytest.raises(EvidenceError):
        relabel_evidence(game, damaged, mapping)
    different = Game(game.scenario, (1, 0, 0, 0), game.payoff)
    with pytest.raises(EvidenceError, match='different verifier'):
        relabel_evidence(game, _local(different), mapping)
    artifact = _local(game)
    with pytest.raises(EvidenceError, match='max_artifact_bytes'):
        relabel_evidence(game, artifact, mapping,
                         budget=VerificationBudget(max_artifact_bytes=artifact.nbytes-1))


@pytest.mark.parametrize('kind, resource', [('numerical_hint', 'quantum_finite'),
                                          ('nonsignalling_lp', 'nonsignalling')])
def test_unsupported_evidence_is_not_transported_by_rewriting_ids(kind, resource):
    game = chsh()
    artifact = make_evidence(game, ResourceSpec(resource), kind, {})
    with pytest.raises(NotImplementedError, match='unsupported'):
        relabel_evidence(game, artifact, _mapping(game))


def test_restricted_source_contract_is_never_erased_during_transport():
    game = chsh()
    source = _local(game)
    artifact = make_evidence(game, ResourceSpec('local', shared_randomness='forbidden'),
                             source.kind, source.payload)
    with pytest.raises(EvidenceError, match='standard unrestricted'):
        relabel_evidence(game, artifact, _mapping(game))


def test_invalid_source_is_rejected_before_target_table_allocation(monkeypatch):
    import nonlocalgames.equivalence.evidence as module
    game = chsh()
    artifact = _local(game)
    monkeypatch.setattr(module, 'relabel_game', lambda *a, **k: pytest.fail('check source first'))
    with pytest.raises(EvidenceError, match='max_artifact_bytes'):
        relabel_evidence(game, artifact, _mapping(game),
                         budget=VerificationBudget(max_artifact_bytes=artifact.nbytes-1))
    invalid = make_evidence(game, ResourceSpec('local'), 'local_enumeration',
                            {'strategy': [[0, 0], [0, 0]], 'value': '1'})
    with pytest.raises(EvidenceError, match='exact replay'):
        relabel_evidence(game, invalid, _mapping(game))


def test_transport_checks_new_score_instead_of_trusting_a_builder(monkeypatch):
    import nonlocalgames.equivalence.evidence as module
    game, source = chsh(), _local(chsh())
    original, calls = module.verify_artifact, []
    def checked(game, artifact, **kwargs):
        result = original(game, artifact, **kwargs)
        calls.append(result)
        return replace(result, lower=Fraction(0)) if len(calls) == 2 else result
    monkeypatch.setattr(module, 'verify_artifact', checked)
    with pytest.raises(RuntimeError, match='changed'):
        relabel_evidence(game, source, _mapping(game))
