"""Operational failures and serialized receipts remain separate from mathematical proofs."""
from fractions import Fraction

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.analysis import classical_value
from nonlocalgames.claims import (EvidenceKind, ExecutionStatus, TruthStatus,
                                  ValueClaim, assess_value, evidence_to_dict, make_evidence)
from nonlocalgames.games import chsh


LOCAL = ResourceSpec('local')
QUANTUM = ResourceSpec('quantum_finite')


def _local(game):
    result = classical_value(game)
    return make_evidence(game, LOCAL, EvidenceKind.LOCAL_ENUMERATION,
                          {'strategy': result.witness, 'value': str(result.value)})


@pytest.mark.parametrize('failure, expected', [
    (RuntimeError('simulated checker invariant failure'), ExecutionStatus.CHECKER_ERROR),
    (ImportError('simulated unavailable exact checker dependency'), ExecutionStatus.DEPENDENCY_UNAVAILABLE),
])
def test_checker_failures_remain_operational_unknown(monkeypatch, failure, expected):
    import nonlocalgames.analysis.classical as classical
    game = chsh()
    artifact = _local(game)
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(classical, 'classical_value', fail)
    result = assess_value(game, ValueClaim(LOCAL, 'at_most', '3/4'), [artifact])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == expected
    assert result.issues[0].evidence_id == artifact.artifact_id
    assert str(failure) in result.issues[0].message


def test_restricted_numerical_hint_retains_context_without_gaining_bounds():
    from nonlocalgames.claims import assessment_to_dict
    game = chsh()
    restricted = ResourceSpec('quantum_finite', (2, 2), state_constraint='ppt',
                               ppt_cuts=((0,),), measurement_class='projective')
    hint = make_evidence(game, restricted, EvidenceKind.NUMERICAL_HINT,
                         {'value': 1.0, 'guarantee': 'exact', 'method': 'untrusted proposal'})
    archived = evidence_to_dict(hint)
    assert archived['resource']['local_dimensions'] == [2, 2]
    assert archived['resource']['state_constraint'] == 'ppt'
    result = assess_value(game, ValueClaim(QUANTUM, 'at_least', '9/10'), [hint])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.COMPLETED
    assert result.intervals[0].lower is None and result.intervals[0].upper is None
    assert result.checks[0].artifact_id == hint.artifact_id and not result.checks[0].is_proof
    receipt = assessment_to_dict(result)
    assert receipt['checks'][0]['checker'] == 'untrusted_observation.v1'
    assert receipt['numerical_hints'] == [hint.artifact_id]


def test_claim_definitions_reject_changed_semantics_and_receipts_are_not_proofs():
    from copy import deepcopy
    from nonlocalgames.claims import assessment_to_dict, claim_from_dict, claim_to_dict
    game = chsh()
    claim = ValueClaim(LOCAL, 'at_most', Fraction(3, 4))
    encoded = claim_to_dict(game, claim)
    assert claim_from_dict(game, encoded) == claim
    for alter in (
        lambda data: data.update(threshold='1'),
        lambda data: data.update(relation='at_least'),
        lambda data: data['resource'].update(kind='quantum_finite'),
        lambda data: data.update(unexpected=True),
    ):
        changed = deepcopy(encoded)
        alter(changed)
        with pytest.raises(ValueError):
            claim_from_dict(game, changed)
    equivalent = deepcopy(encoded)
    equivalent['threshold'] = '6/8'
    assert claim_from_dict(game, equivalent) == claim
    result = assess_value(game, claim, [_local(game)])
    receipt = assessment_to_dict(result)
    assert receipt['truth'] == 'proved'
    assert receipt['checks'][0]['upper'] == '3/4'
    fake_admission = assess_value(game, claim, [receipt])
    assert fake_admission.truth == TruthStatus.UNKNOWN
    assert fake_admission.execution == ExecutionStatus.INVALID_EVIDENCE


def test_operational_failure_cannot_be_serialized_as_mathematical_disproof():
    from nonlocalgames.claims import ClaimAssessment
    for execution in ExecutionStatus:
        if execution == ExecutionStatus.COMPLETED:
            continue
        with pytest.raises(ValueError, match='operational failure'):
            ClaimAssessment('claim', 'game', TruthStatus.DISPROVED, execution)
