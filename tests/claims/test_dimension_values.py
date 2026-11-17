from fractions import Fraction
from hashlib import sha256

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.analysis import certify_quantum_strategy, classical_value, verify_xor_dual
from nonlocalgames.claims import (GapClaim, ValueClaim, VerificationBudget, assess_gap, assess_value,
                                 assessment_to_dict, make_evidence)
from nonlocalgames.core.identity import canonical_json, resource_id
from nonlocalgames.games import chsh
from nonlocalgames.quantum.exact import exact_chsh_strategy


@pytest.fixture(scope='module')
def evidence():
    game = chsh()
    local = classical_value(game)
    optimum = make_evidence(game, ResourceSpec('local'), 'local_enumeration',
                            {'strategy': local.witness, 'value': '3/4'})
    quantum = certify_quantum_strategy(game, exact_chsh_strategy())
    point = make_evidence(game, ResourceSpec('quantum_finite'), 'quantum_strategy', quantum.witness)
    dual = verify_xor_dual(game, (Fraction(9, 100),)*4)
    upper = make_evidence(game, ResourceSpec('quantum_commuting'), 'xor_dual', dual.witness)
    return optimum, point, upper


def test_cap_is_componentwise_and_preserves_witness_binding(evidence):
    local, point, _ = evidence
    cap = ResourceSpec('quantum_finite', (2, 2))
    claim = ValueClaim(cap, 'at_least', Fraction(4, 5))
    result = assess_value(chsh(), claim, [local, point])
    assert result.truth == 'proved' and result.execution == 'completed'
    record = assessment_to_dict(result)
    assert record['schema'] == 'nonlocalgames.dimension_claim_assessment.v1'
    assert record['target_resource_id'] == record['intervals'][0]['target_resource_id'] == resource_id(cap)
    assert record['intervals'][0]['local_dimensions'] == [2, 2]
    assert result.intervals[0].lower_refs == (point.artifact_id,)
    for limits in ((4, 1), (1, 4)):
        resource = ResourceSpec('quantum_finite', limits)
        result = assess_value(chsh(), ValueClaim(resource, 'at_least', Fraction(4, 5)), [local, point])
        assert result.execution == 'completed' and result.truth == 'unknown'
        assert result.intervals[0].lower == Fraction(3, 4)
        assert result.intervals[0].lower_refs == (local.artifact_id,)


def test_upper_bound_follows_set_inclusion_but_local_upper_does_not(evidence):
    local, _, upper = evidence
    cap = ResourceSpec('quantum_finite', (3, 2))
    threshold = Fraction(43, 50)
    claim = ValueClaim(cap, 'at_most', threshold)
    assert assess_value(chsh(), claim, [local]).truth == 'unknown'
    result = assess_value(chsh(), claim, [upper])
    assert result.truth == 'proved'
    assert result.intervals[0].lower is None
    assert result.intervals[0].upper_refs == (upper.artifact_id,)


def test_bad_proofs_and_other_contracts_do_not_enter_dimension_rule(evidence):
    cap = ResourceSpec('quantum_finite', (2, 2))
    claim = ValueClaim(cap, 'at_least', 0)
    limited = assess_value(chsh(), claim, evidence, budget=VerificationBudget(max_artifacts=1))
    assert limited.execution == 'budget_exceeded' and limited.truth == 'unknown'
    wrong = make_evidence(chsh(), ResourceSpec('local'), 'local_enumeration',
                          {'strategy': [[0, 0], [0, 0]], 'value': '0'})
    assert assess_value(chsh(), claim, [wrong]).execution == 'invalid_evidence'
    for res in (ResourceSpec('quantum_finite', (2,)), ResourceSpec('quantum_finite', (2, 2),
                              measurement_class='projective')):
        assert assess_value(chsh(), ValueClaim(res, 'at_least', 0), evidence).execution == 'unsupported'
    gap = GapClaim(cap, ResourceSpec('quantum_finite', (1, 1)))
    assert assess_gap(chsh(), gap, evidence).execution == 'unsupported'


def test_standard_receipt_bytes_match_pre_dimension_rule(evidence):
    local, _, _ = evidence
    game, res = chsh(), ResourceSpec('local')
    for result, digest in (
        (assess_value(game, ValueClaim(res, 'at_least', '3/4'), [local]),
         '4de4d8ea087d072e6dd9f283dad252e167c9ca205e87be3607f754fae6d7098f'),
        (assess_gap(game, GapClaim(ResourceSpec('quantum_finite'), res), [local]),
         '338625df2e7ef1f735a4b59b326ade6a16c335d43c8aa9d1514bf56fac9602e9'),
    ):
        encoded = canonical_json(assessment_to_dict(result)).encode()
        assert sha256(encoded).hexdigest() == digest


def test_global_conflict_is_reported_even_if_the_point_does_not_fit(monkeypatch):
    from nonlocalgames import Resource
    from nonlocalgames.claims import admission
    from nonlocalgames.claims.types import EvidenceCheck
    checks = iter((EvidenceCheck('point', Resource.QUANTUM_FINITE, Fraction(1), None,
                                 'rational_quantum_strategy.v1', attained_score=1, actual_dimensions=(2, 2)),
                   EvidenceCheck('upper', Resource.NONSIGNALLING, None, Fraction(1, 2),
                                 'rational_ns_lp.v1')))
    monkeypatch.setattr(admission, 'verify_artifact', lambda *args, **kwargs: next(checks))
    query = ValueClaim(ResourceSpec('quantum_finite', (1, 1)), 'at_most', 1)
    result = assess_value(chsh(), query, [object(), object()])
    assert result.truth == 'unknown' and result.execution == 'conflict'
    assert result.verified_evidence == ('point', 'upper') and not result.intervals
