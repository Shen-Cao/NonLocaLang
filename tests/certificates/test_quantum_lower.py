import copy
import json
from fractions import Fraction

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.analysis import classical_value, certify_xor_upper_bound
from nonlocalgames.certificates.quantum import (certify_quantum_lower_bound,
    certify_quantum_strategy, verify_quantum_lower_certificate)
from nonlocalgames.claims import EvidenceKind, GapClaim, ValueClaim, assess_gap, assess_value, make_evidence
from nonlocalgames.games import chsh, ghz, magic_square
from nonlocalgames.quantum import chsh_strategy, magic_square_strategy
from nonlocalgames.quantum.exact import ExactBudget, ExactBudgetExceeded, exact_chsh_strategy, exact_ghz_strategy


def local_artifact(game):
    result = classical_value(game)
    return make_evidence(game, ResourceSpec('local'), EvidenceKind.LOCAL_ENUMERATION,
                         {'value': str(result.value), 'strategy': result.witness})


def test_exact_chsh_lower_and_xor_upper_form_a_certified_interval():
    game = chsh()
    lower = certify_quantum_strategy(game, exact_chsh_strategy())
    upper = certify_xor_upper_bound(game, solver='SCS', eps=1e-8)
    assert lower.value == Fraction(3880899, 4546756)
    assert lower.value > Fraction(3, 4) and lower.value <= upper.value
    assert upper.value - lower.value < Fraction(1, 1_000_000)
    artifacts = [local_artifact(game),
        make_evidence(game, ResourceSpec('quantum_finite'), EvidenceKind.QUANTUM_STRATEGY, lower.witness),
        make_evidence(game, ResourceSpec('quantum_commuting'), EvidenceKind.XOR_DUAL, upper.witness)]
    result = assess_gap(game, GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local')), artifacts)
    assert result.truth == 'proved' and result.execution == 'completed'
    assert result.intervals[0].lower == lower.value and result.intervals[0].upper == upper.value


def test_exact_ghz_and_rationalized_magic_square_attain_one():
    cases = [(ghz(), certify_quantum_strategy(ghz(), exact_ghz_strategy())),
             (magic_square(), certify_quantum_lower_bound(magic_square(), magic_square_strategy()))]
    for game, result in cases:
        assert result.value == 1 and result.guarantee == 'certified_lower_bound'
        recovered = verify_quantum_lower_certificate(game, json.loads(json.dumps(result.witness)))
        assert recovered.value == 1
        artifact = make_evidence(game, ResourceSpec('quantum_finite'), EvidenceKind.QUANTUM_STRATEGY, result.witness)
        assert assess_gap(game, GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local')),
                           [local_artifact(game), artifact]).truth == 'proved'


def test_numerical_chsh_repair_yields_a_checked_positive_gap():
    result = certify_quantum_lower_bound(chsh(), chsh_strategy(), repair_steps=20)
    assert result.value > Fraction(3, 4)
    assert result.guarantee == 'certified_lower_bound'
    assert 'rationalization' in result.diagnostics
    assert verify_quantum_lower_certificate(chsh(), result.witness).value == result.value


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(lower_bound='1'),
    lambda p: p.update(resource='quantum_commuting'),
    lambda p: p.update(game_fingerprint='0'*64),
    lambda p: p.update(lower_bound='1e1000000000'),
    lambda p: p.update(extra=True),
])
def test_certificate_rejects_wrong_score_scope_binding_and_syntax(mutation):
    payload = copy.deepcopy(certify_quantum_strategy(chsh(), exact_chsh_strategy()).witness)
    mutation(payload)
    with pytest.raises(ValueError):
        verify_quantum_lower_certificate(chsh(), payload)


def test_exact_budget_exhaustion_is_not_a_negative_mathematical_verdict():
    from nonlocalgames.claims import VerificationBudget
    game = chsh()
    result = certify_quantum_strategy(game, exact_chsh_strategy())
    with pytest.raises(ExactBudgetExceeded):
        verify_quantum_lower_certificate(game, result.witness, budget=ExactBudget(max_joint_dimension=3))
    artifact = make_evidence(game, ResourceSpec('quantum_finite'), EvidenceKind.QUANTUM_STRATEGY, result.witness)
    limited = assess_value(game, ValueClaim(ResourceSpec('quantum_finite'), 'at_least', '4/5'), [artifact],
                           budget=VerificationBudget(max_quantum_joint_dimension=3))
    assert limited.truth == 'unknown' and limited.execution == 'budget_exceeded'


def test_lower_certificate_cannot_override_restricted_resource_assumptions():
    game = chsh()
    result = certify_quantum_strategy(game, exact_chsh_strategy())
    for resource in (ResourceSpec('quantum_finite', (1, 1)),
                     ResourceSpec('quantum_finite', measurement_class='projective'),
                     ResourceSpec('quantum_commuting')):
        artifact = make_evidence(game, resource, EvidenceKind.QUANTUM_STRATEGY, result.witness)
        assessed = assess_value(game, ValueClaim(ResourceSpec('quantum_finite'), 'at_least', '4/5'), [artifact])
        assert assessed.truth == 'unknown' and assessed.execution == 'unsupported'
