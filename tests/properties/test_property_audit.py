"""Property verdicts require the right witnessed model, attainment fact, and support scope."""
from fractions import Fraction

import pytest

from nonlocalgames import AnalysisTask, Game, ResourceSpec
from nonlocalgames.analysis import AnalysisResult, classical_value, verify_xor_dual
from nonlocalgames.certificates.quantum import certify_quantum_strategy
from nonlocalgames.claims import EvidenceKind, VerificationBudget, make_evidence
from nonlocalgames.games import chsh, ghz
from nonlocalgames.properties import PropertyClaim, assess_property
from nonlocalgames.quantum.exact import exact_chsh_strategy, exact_ghz_strategy


def _local_artifact(game, *, complete=False):
    result = classical_value(game)
    payload = {'strategy': result.witness}
    if complete:
        payload['value'] = str(result.value)
    return make_evidence(game, ResourceSpec('local'),
                         EvidenceKind.LOCAL_ENUMERATION if complete else EvidenceKind.LOCAL_STRATEGY, payload)


def _quantum_artifact(game, strategy):
    certificate = certify_quantum_strategy(game, strategy)
    return make_evidence(game, ResourceSpec('quantum_finite'), EvidenceKind.QUANTUM_STRATEGY,
                         certificate.witness)


def _ns_perfect_chsh():
    game = chsh()
    primal = tuple('1/2' if (a[0] ^ a[1]) == x[0]*x[1] else '0'
                   for x in game.inputs for a in game.outputs)
    dual = ('-1/4',)*4 + ('0',)*8
    return make_evidence(game, ResourceSpec('nonsignalling'), EvidenceKind.NONSIGNALLING_LP,
                         {'primal': primal, 'dual': dual})


def test_perfect_nonsignalling_point_does_not_prove_quantum_attainment(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *a, **k: pytest.fail('property inference must only replay exact evidence'))
    game, evidence = chsh(), (_ns_perfect_chsh(),)
    perfect = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')), evidence)
    pseudo = assess_property(game, PropertyClaim('pseudotelepathy'), evidence)
    assert perfect.truth == pseudo.truth == 'unknown'
    assert perfect.execution == pseudo.execution == 'completed'


def test_exact_nonperfect_quantum_point_and_classical_gap_are_not_pseudotelepathy():
    game = chsh()
    evidence = (_quantum_artifact(game, exact_chsh_strategy()), _local_artifact(game, complete=True))
    assert assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')),
                           evidence).truth == 'unknown'
    assert assess_property(game, PropertyClaim('pseudotelepathy'), evidence).truth == 'unknown'


def test_explicit_quantum_attainment_and_classical_upper_bound_prove_pseudotelepathy():
    game = ghz()
    point = _quantum_artifact(game, exact_ghz_strategy())
    classical = _local_artifact(game, complete=True)
    assert assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')),
                           (point,)).truth == 'proved'
    assert assess_property(game, PropertyClaim('pseudotelepathy'), (point,)).truth == 'unknown'
    result = assess_property(game, PropertyClaim('pseudotelepathy'), (point, classical))
    assert result.truth == 'proved' and result.execution == 'completed'


def test_classical_perfect_point_refutes_pseudotelepathy_and_embeds_in_finite_quantum():
    game = Game.from_rule(chsh().scenario, lambda x, a: 1)
    point = _local_artifact(game)
    assert assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')),
                           (point,)).truth == 'proved'
    assert assess_property(game, PropertyClaim('pseudotelepathy'), (point,)).truth == 'disproved'


def test_certified_quantum_upper_below_one_refutes_attainment():
    game = chsh()
    bound = verify_xor_dual(game, (Fraction(9, 100),)*4)
    artifact = make_evidence(game, ResourceSpec('quantum_commuting'), EvidenceKind.XOR_DUAL, bound.witness)
    for claim in (PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')),
                  PropertyClaim('pseudotelepathy')):
        result = assess_property(game, claim, (artifact,))
        assert result.truth == 'disproved' and result.execution == 'completed'


def test_raw_result_and_forged_hint_metadata_do_not_become_property_proofs():
    game = chsh()
    claim = PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite'))
    raw = AnalysisResult(Fraction(1), 'quantum_finite', 'invented', 'exact', 'optimal')
    result = assess_property(game, claim, (raw,))
    assert result.truth == 'unknown' and result.execution == 'invalid_evidence'
    hint = make_evidence(game, ResourceSpec('quantum_finite'), 'numerical_hint',
                         {'guarantee': 'exact', 'attained_score': '1', 'value': '1', 'certified': True})
    result = assess_property(game, claim, (hint,))
    assert result.truth == 'unknown' and result.execution == 'completed'


def test_zero_probability_entries_change_structural_scope_but_not_perfect_play():
    game = Game.from_rule(chsh().scenario,
        lambda x, a: int(a[0] == a[1]) if x == (0, 0) else 2*a[0] + Fraction(a[1], 2),
        distribution={(0, 0): 1})
    for kind in ('acceptance', 'boolean_predicate', 'xor_score', 'xor_predicate'):
        assert assess_property(game, PropertyClaim(kind, off_support='all')).truth == 'disproved'
        assert assess_property(game, PropertyClaim(kind, off_support='sampled')).truth == 'proved'
    point = _local_artifact(game)
    perfect = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')), (point,))
    assert perfect.truth == 'proved' and perfect.execution == 'completed'


def test_xor_score_recognition_is_distinct_from_xor_winning_predicate_and_solver_scope():
    for reward in (0, 1, Fraction(1, 2)):
        game = Game.from_rule(chsh().scenario, lambda x, a: reward)
        assert assess_property(game, PropertyClaim('xor_score')).truth == 'proved'
        assert assess_property(game, PropertyClaim('xor_predicate')).truth == 'disproved'
    signed = Game.from_rule(chsh().scenario, lambda x, a: 2 if a[0] == a[1] else -1)
    assert assess_property(signed, PropertyClaim('xor_score')).truth == 'proved'
    assert assess_property(signed, PropertyClaim('xor_predicate')).truth == 'disproved'
    assert assess_property(ghz(), PropertyClaim('xor_predicate')).truth == 'proved'
    assert 'xor' not in AnalysisTask(ghz(), ResourceSpec('quantum_finite')).capabilities()


def test_property_budget_failure_is_not_a_negative_property_verdict():
    result = assess_property(chsh(), PropertyClaim('xor_predicate'),
                             budget=VerificationBudget(max_property_entries=1))
    assert result.truth == 'unknown' and result.execution == 'budget_exceeded'


def test_a_certified_value_interval_equal_to_one_is_not_itself_an_attaining_witness():
    from nonlocalgames.claims.admission import EvidenceBundle
    from nonlocalgames.claims.types import CertifiedInterval, EvidenceCheck, ExecutionStatus
    from nonlocalgames.core.identity import verifier_id
    from nonlocalgames.core.resources import Resource
    from nonlocalgames.properties.inference import _perfect
    # A future bound verifier could establish a supremum without constructing a realization.
    bound = EvidenceCheck('synthetic_bound', Resource.QUANTUM_FINITE, Fraction(1), Fraction(1),
                           'value_bound_only.v1')
    bundle = EvidenceBundle(verifier_id(chsh()), ExecutionStatus.COMPLETED,
        intervals=(CertifiedInterval(Resource.QUANTUM_FINITE, Fraction(1), Fraction(1)),), checks=(bound,))
    truth, _ = _perfect(Resource.QUANTUM_FINITE, bundle)
    assert truth == 'unknown'


def test_quantum_verification_budget_failure_is_not_nonexistence():
    game = ghz()
    point = _quantum_artifact(game, exact_ghz_strategy())
    result = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')),
                             (point,), budget=VerificationBudget(max_quantum_joint_dimension=3))
    assert result.truth == 'unknown' and result.execution == 'budget_exceeded'


def test_property_evidence_bundle_is_fail_closed_even_after_a_valid_perfect_witness():
    game = ghz()
    point = _quantum_artifact(game, exact_ghz_strategy())
    forged = AnalysisResult(Fraction(1), 'quantum_finite', 'unchecked', 'exact', 'optimal')
    result = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')),
                             (point, forged))
    assert result.truth == 'unknown' and result.execution == 'invalid_evidence'


def test_zero_denominator_evidence_is_invalid_not_a_property_verdict():
    from nonlocalgames.claims import EvidenceKind, make_evidence
    from nonlocalgames import ResourceSpec
    from nonlocalgames.games import chsh
    from nonlocalgames.properties import PropertyClaim, assess_property
    game = chsh()
    bad = make_evidence(game, ResourceSpec('local'), EvidenceKind.LOCAL_ENUMERATION,
                         {'strategy': [[0, 0], [0, 0]], 'value': '1/0'})
    result = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('local')), [bad])
    assert result.truth == 'unknown' and result.execution == 'invalid_evidence'
