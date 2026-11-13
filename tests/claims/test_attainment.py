from fractions import Fraction

import pytest

from nonlocalgames import Resource, ResourceSpec
from nonlocalgames.analysis import classical_value, certify_nonsignalling, certify_quantum_strategy, certify_xor_upper_bound
from nonlocalgames.claims import EvidenceKind, make_evidence, verify_artifact, verify_evidence_bundle
from nonlocalgames.claims.types import EvidenceCheck
from nonlocalgames.games import chsh, ghz
from nonlocalgames.quantum.exact import exact_ghz_strategy


def test_verified_attainment_sources_stay_distinct():
    game = chsh()
    local = classical_value(game)
    local_art = make_evidence(game, ResourceSpec('local'), EvidenceKind.LOCAL_ENUMERATION,
                              {'strategy': local.witness, 'value': str(local.value)})
    local_check = verify_artifact(game, local_art)
    assert local_check.attained_score == Fraction(3, 4)
    assert local_check.actual_dimensions is None
    ns = certify_nonsignalling(game)
    ns_art = make_evidence(game, ResourceSpec('nonsignalling'), EvidenceKind.NONSIGNALLING_LP,
                           {k: list(map(str, ns.witness[k])) for k in ('primal', 'dual')})
    ns_check = verify_artifact(game, ns_art)
    assert ns_check.resource == Resource.NONSIGNALLING and ns_check.attained_score == 1
    xor = certify_xor_upper_bound(game, solver='SCS', eps=1e-8)
    upper = make_evidence(game, ResourceSpec('quantum_commuting'), EvidenceKind.XOR_DUAL, xor.witness)
    assert verify_artifact(game, upper).attained_score is None
    quantum = certify_quantum_strategy(ghz(), exact_ghz_strategy())
    quantum_art = make_evidence(ghz(), ResourceSpec('quantum_finite'), EvidenceKind.QUANTUM_STRATEGY, quantum.witness)
    checked = verify_artifact(ghz(), quantum_art)
    assert checked.attained_score == 1 and checked.actual_dimensions == (2, 2, 2)


def test_attainment_itself_creates_lower_bound_and_conflict(monkeypatch):
    import nonlocalgames.claims.admission as admission
    checks = iter((EvidenceCheck('attained', Resource.QUANTUM_FINITE, None, None, 'test', attained_score=1),
                   EvidenceCheck('upper', Resource.NONSIGNALLING, None, Fraction(3, 4), 'test')))
    monkeypatch.setattr(admission, 'verify_artifact', lambda *a, **k: next(checks))
    result = verify_evidence_bundle(chsh(), [object(), object()])
    assert result.execution == 'conflict'
    interval = next(i for i in result.intervals if i.resource == Resource.QUANTUM_FINITE)
    assert interval.lower == 1 and interval.upper == Fraction(3, 4)


def test_hint_cannot_accidentally_carry_attainment_or_bounds():
    with pytest.raises(ValueError, match='hints'):
        EvidenceCheck('hint', Resource.QUANTUM_FINITE, None, None, 'hint', False, attained_score=1)
    with pytest.raises(TypeError, match='exact rationals'):
        EvidenceCheck('float', Resource.QUANTUM_FINITE, .999, None, 'wrong')
