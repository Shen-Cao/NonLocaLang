"""Independent adversarial tests of claim admission and bound-direction inference."""
from dataclasses import replace
from fractions import Fraction

import pytest

from nonlocalgames import Game, Resource, ResourceSpec, Scenario
from nonlocalgames.analysis import AnalysisResult, certify_nonsignalling, classical_value
from nonlocalgames.certificates.xor import verify_xor_dual
from nonlocalgames.claims import (EvidenceKind, ExecutionStatus, GapClaim, TruthStatus,
                                  ValueClaim, VerificationBudget, assess_gap, assess_value,
                                  evidence_from_dict, evidence_to_dict, make_evidence)
from nonlocalgames.games import chsh, gyni


LOCAL = ResourceSpec('local')
QUANTUM = ResourceSpec('quantum_finite')
COMMUTING = ResourceSpec('quantum_commuting')
NS = ResourceSpec('nonsignalling')


def _local(game, *, enumeration=True):
    result = classical_value(game)
    payload = {'strategy': result.witness}
    if enumeration:
        payload['value'] = str(result.value)
    kind = EvidenceKind.LOCAL_ENUMERATION if enumeration else EvidenceKind.LOCAL_STRATEGY
    return make_evidence(game, LOCAL, kind, payload)


def _ns(game, *, primal=None):
    certificate = certify_nonsignalling(game)
    primal = certificate.witness['primal'] if primal is None else primal
    return make_evidence(game, NS, EvidenceKind.NONSIGNALLING_LP,
                          {'primal': list(map(str, primal)),
                           'dual': list(map(str, certificate.witness['dual']))})


def test_local_witness_is_a_lower_bound_never_an_optimality_certificate():
    game, artifact = chsh(), _local(chsh(), enumeration=False)
    lower = assess_value(game, ValueClaim(QUANTUM, 'at_least', '3/4'), [artifact])
    assert lower.truth == TruthStatus.PROVED
    upper = assess_value(game, ValueClaim(LOCAL, 'at_most', '3/4'), [artifact])
    assert upper.truth == TruthStatus.UNKNOWN and upper.execution == ExecutionStatus.COMPLETED
    assert upper.intervals[0].upper is None


@pytest.mark.parametrize('constructor', [chsh, gyni])
def test_exact_ns_gap_does_not_prove_quantum_advantage(constructor):
    game = constructor()
    assessment = assess_gap(game, GapClaim(QUANTUM, LOCAL), [_local(game), _ns(game)])
    assert assessment.truth == TruthStatus.UNKNOWN
    assert assessment.execution == ExecutionStatus.COMPLETED
    assert assessment.intervals[0].lower == classical_value(game).value


def test_ns_primal_lower_is_not_transported_down_to_quantum():
    game = gyni()
    artifact = _ns(game)
    quantum = assess_value(game, ValueClaim(QUANTUM, 'at_least', '1/3'), [artifact])
    assert quantum.truth == TruthStatus.UNKNOWN
    assert quantum.intervals[0].lower is None
    assert quantum.intervals[0].upper == Fraction(1, 3)
    ns = assess_value(game, ValueClaim(NS, 'at_least', '1/3'), [artifact])
    assert ns.truth == TruthStatus.PROVED


def test_commuting_upper_does_not_become_a_nonsignalling_upper():
    game = chsh()
    payload = verify_xor_dual(game, [Fraction(9, 100)]*4).witness
    xor = make_evidence(game, COMMUTING, EvidenceKind.XOR_DUAL, payload)
    artifacts = [xor, _ns(game), _local(game)]
    ns = assess_value(game, ValueClaim(NS, 'at_most', '9/10'), artifacts)
    assert ns.truth == TruthStatus.DISPROVED and ns.execution == ExecutionStatus.COMPLETED
    quantum = assess_value(game, ValueClaim(QUANTUM, 'at_most', '43/50'), artifacts)
    assert quantum.truth == TruthStatus.PROVED
    assert assess_gap(game, GapClaim(QUANTUM, LOCAL), artifacts).truth == TruthStatus.UNKNOWN


@pytest.mark.parametrize('constant', [Fraction(1), Fraction(-7, 3)])
def test_closed_value_relations_and_signed_score_boundaries(constant):
    game = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: constant)
    artifacts = [_local(game), _ns(game)]
    for relation in ('at_least', 'at_most'):
        assert assess_value(game, ValueClaim(QUANTUM, relation, constant), artifacts).truth == TruthStatus.PROVED
    assert assess_value(game, ValueClaim(QUANTUM, 'at_least', constant+1), artifacts).truth == TruthStatus.DISPROVED
    assert assess_value(game, ValueClaim(QUANTUM, 'at_most', constant-1), artifacts).truth == TruthStatus.DISPROVED
    assert assess_gap(game, GapClaim(QUANTUM, LOCAL, 0), artifacts).truth == TruthStatus.DISPROVED


def test_strict_gap_uses_strict_lower_and_nonstrict_upper_comparisons():
    game = gyni()
    exact = [_local(game), _ns(game)]
    assert assess_gap(game, GapClaim(NS, LOCAL, Fraction(1, 13)), exact).truth == TruthStatus.PROVED
    assert assess_gap(game, GapClaim(NS, LOCAL, Fraction(1, 12)), exact).truth == TruthStatus.DISPROVED
    game = chsh()
    deterministic = tuple(int(a == (0, 0)) for x in game.inputs for a in game.outputs)
    interval = [_local(game), _ns(game, primal=deterministic)]
    result = assess_gap(game, GapClaim(NS, LOCAL, 0), interval)
    assert result.truth == TruthStatus.UNKNOWN
    assert result.intervals[0].lower == result.intervals[1].upper
    assert assess_gap(game, GapClaim(NS, LOCAL, Fraction(1, 4)), interval).truth == TruthStatus.DISPROVED


def test_fake_exact_result_and_numerical_hint_never_acquire_proof_status():
    game = chsh()
    forged = AnalysisResult(Fraction(1), 'quantum_finite', 'fabricated', 'exact', 'optimal',
                             {'certified': True}, ((0, 0), (0, 0)))
    rejected = assess_value(game, ValueClaim(QUANTUM, 'at_least', 1), [forged])
    assert rejected.truth == TruthStatus.UNKNOWN and rejected.execution == ExecutionStatus.INVALID_EVIDENCE
    hint = make_evidence(game, QUANTUM, EvidenceKind.NUMERICAL_HINT,
                         {'value': '1', 'guarantee': 'certified_lower_bound', 'certified': True, 'status': 'optimal'})
    result = assess_gap(game, GapClaim(QUANTUM, LOCAL), [_local(game), hint])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.COMPLETED
    assert result.numerical_hints == (hint.artifact_id,)
    assert hint.artifact_id not in result.verified_evidence


def test_verified_bound_records_are_not_public_admission_tokens():
    from nonlocalgames.claims.verify import EvidenceCheck
    forged = EvidenceCheck('forged', Resource.QUANTUM_FINITE, Fraction(1), Fraction(1), 'pretend_checker')
    result = assess_value(chsh(), ValueClaim(QUANTUM, 'at_least', 1), [forged])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.INVALID_EVIDENCE


def test_local_optimality_payload_is_replayed_and_mismatch_rejected():
    game = chsh()
    forged = make_evidence(game, LOCAL, EvidenceKind.LOCAL_ENUMERATION,
                           {'strategy': ((0, 0), (0, 0)), 'value': '1'})
    result = assess_value(game, ValueClaim(LOCAL, 'at_least', 1), [forged])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.INVALID_EVIDENCE
    good = _local(game)
    limited = assess_value(game, ValueClaim(LOCAL, 'at_least', '1/2'), [good],
                            budget=VerificationBudget(max_strategies=1))
    assert limited.truth == TruthStatus.UNKNOWN and limited.execution == ExecutionStatus.BUDGET_EXCEEDED


def test_saved_certificate_replay_does_not_call_lp_optimization(monkeypatch):
    import scipy.optimize
    game = chsh()
    artifact = evidence_from_dict(evidence_to_dict(_ns(game)))
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *args, **kwargs: pytest.fail('must recheck, not solve'))
    result = assess_value(game, ValueClaim(QUANTUM, 'at_most', 1), [artifact])
    assert result.truth == TruthStatus.PROVED and result.execution == ExecutionStatus.COMPLETED


def test_wrong_source_kind_and_restricted_claims_stay_unsupported():
    game = chsh()
    payload = _ns(game).payload
    wrong_source = make_evidence(game, QUANTUM, EvidenceKind.NONSIGNALLING_LP, payload)
    result = assess_value(game, ValueClaim(QUANTUM, 'at_most', 1), [wrong_source])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.UNSUPPORTED
    for resource in (ResourceSpec('quantum_finite', (2, 2)),
                     ResourceSpec('quantum_finite', measurement_class='projective'),
                     ResourceSpec('local', shared_randomness='forbidden')):
        result = assess_value(game, ValueClaim(resource, 'at_least', '3/4'), [_local(game)])
        assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.UNSUPPORTED


def test_evidence_binding_preserves_sampler_and_ignores_display_name_only():
    game = chsh()
    artifact = _local(game)
    renamed = assess_value(replace(game, name='same verifier'), ValueClaim(LOCAL, 'at_most', '3/4'), [artifact])
    assert renamed.truth == TruthStatus.PROVED
    changed = Game(game.scenario, (1, 0, 0, 0), game.payoff)
    rejected = assess_value(changed, ValueClaim(LOCAL, 'at_most', '3/4'), [artifact])
    assert rejected.truth == TruthStatus.UNKNOWN and rejected.execution == ExecutionStatus.INVALID_EVIDENCE


def test_conflicting_checked_bounds_are_quarantined_after_transport(monkeypatch):
    from nonlocalgames.claims.verify import EvidenceCheck
    import nonlocalgames.claims.inference as inference
    checks = iter((EvidenceCheck('lower', Resource.LOCAL, Fraction(3, 4), None, 'test_checker'),
                   EvidenceCheck('upper', Resource.NONSIGNALLING, None, Fraction(1, 2), 'test_checker')))
    monkeypatch.setattr(inference, 'verify_artifact', lambda *args, **kwargs: next(checks))
    result = assess_value(chsh(), ValueClaim(LOCAL, 'at_least', '1/4'), [object(), object()])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.CONFLICT
    assert any(i.lower is not None and i.upper is not None and i.lower > i.upper for i in result.intervals)
    assert result.verified_evidence == ('lower', 'upper')


def test_artifact_count_budget_stops_unbounded_evidence_iterators():
    artifact = _local(chsh(), enumeration=False)
    def stream():
        yield artifact
        yield artifact
        pytest.fail('must not consume beyond max_artifacts+1')
    result = assess_value(chsh(), ValueClaim(LOCAL, 'at_least', 0), stream(),
                           budget=VerificationBudget(max_artifacts=1))
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.BUDGET_EXCEEDED


def test_artifact_byte_budget_counts_the_complete_envelope():
    artifact = _local(chsh(), enumeration=False)
    result = assess_value(chsh(), ValueClaim(LOCAL, 'at_least', 0), [artifact],
                           budget=VerificationBudget(max_artifact_bytes=artifact.nbytes-1))
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.BUDGET_EXCEEDED
