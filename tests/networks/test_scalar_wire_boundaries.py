"""Rational network data is bounded at semantic boundaries, not by metadata guessing."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from nonlocalgames import Scenario
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded
from nonlocalgames.networks import RationalBehavior, SourceNetwork, triangle_finner, triangle_xor_strategy
from nonlocalgames.networks.behavior_serialization import behavior_from_dict, behavior_to_dict
from nonlocalgames.networks.compatibility_certificates import (verify_compatibility_violation,
                                                              violation_from_dict, violation_to_dict)
from nonlocalgames.networks.inflation import (spiral_certificate_from_dict, spiral_certificate_to_dict,
                                             triangle_spiral, verify_spiral_certificate)
from nonlocalgames.networks.inflation._program import compile_spiral
from nonlocalgames.networks.inflation.certificates import make_certificate
from nonlocalgames.networks.serialization import classical_from_dict, classical_to_dict


def _common_bit():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    graph = SourceNetwork(3, ((0, 1), (1, 2), (2, 0)))
    behavior = RationalBehavior(scenario, tuple(Fraction(1, 2) if len(set(a)) == 1 else 0
                                               for a in scenario.outputs))
    return graph, behavior, triangle_finner(graph, behavior).violation


def _w_certificate():
    graph, _, _ = _common_bit()
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    behavior = RationalBehavior(scenario, tuple(Fraction(1, 3) if sum(a) == 1 else 0 for a in scenario.outputs))
    program = compile_spiral(graph, behavior)
    vector = [Fraction(0)]*40
    for i in (0, 15, 23, 31):
        vector[i] = Fraction(1)
    vector[39] = Fraction(-1)
    return graph, behavior, make_certificate(program, 'dual', vector)


@pytest.mark.parametrize('text', ['1e1000000000', '1e-1000000000'])
def test_poisoned_rational_value_fields_fail_before_expansion(text):
    _, behavior, violation = _common_bit()
    document = behavior_to_dict(behavior)
    document['probabilities'][0] = text
    with pytest.raises(ScalarBudgetExceeded):
        behavior_from_dict(document)
    document = classical_to_dict(triangle_xor_strategy())
    for key in ('source_probabilities', 'responses'):
        damaged = deepcopy(document)
        damaged[key][0][0] = text
        with pytest.raises(ScalarBudgetExceeded):
            classical_from_dict(damaged)
    document = violation_to_dict(violation)
    for key in ('lhs', 'rhs'):
        with pytest.raises(ScalarBudgetExceeded):
            violation_from_dict({**document, key: text})
    graph, w, certificate = _w_certificate()
    document = spiral_certificate_to_dict(certificate)
    document['coefficients'][0] = text
    with pytest.raises(ScalarBudgetExceeded):
        spiral_certificate_from_dict(document)
    with pytest.raises(ScalarBudgetExceeded):
        verify_spiral_certificate(graph, w, document)


def test_zero_denominator_is_a_value_error_at_network_boundaries():
    _, behavior, violation = _common_bit()
    document = behavior_to_dict(behavior)
    document['probabilities'][0] = '1/0'
    payload = violation_to_dict(violation)
    graph, w, certificate = _w_certificate()
    spiral = spiral_certificate_to_dict(certificate)
    spiral['coefficients'][0] = '1/0'
    for call in (lambda: behavior_from_dict(document),
                 lambda: violation_from_dict({**payload, 'lhs': '1/0'}),
                 lambda: verify_spiral_certificate(graph, w, spiral)):
        with pytest.raises(ValueError):
            call()


def test_legacy_numeric_spellings_reencode_identical_canonical_bytes():
    _, behavior, violation = _common_bit()
    original = behavior_to_dict(behavior)
    changed = deepcopy(original)
    changed['probabilities'] = [' 5e-1 ' if p == '1/2' else '+0' for p in changed['probabilities']]
    assert behavior_to_dict(behavior_from_dict(changed)) == original
    original = classical_to_dict(triangle_xor_strategy())
    changed = deepcopy(original)
    changed['source_probabilities'] = [['2 / 4' for _ in row] for row in changed['source_probabilities']]
    assert classical_to_dict(classical_from_dict(changed)) == original
    original = violation_to_dict(violation)
    changed = {**original, 'lhs': '0.25', 'rhs': '125e-3'}
    assert violation_to_dict(violation_from_dict(changed)) == original


def test_metadata_strings_are_not_interpreted_as_rationals():
    _, _, violation = _common_bit()
    payload = {**violation_to_dict(violation), 'required_relation': '1e1000000000'}
    assert violation_from_dict(payload).required_relation == '1e1000000000'
    _, _, certificate = _w_certificate()
    payload = {**spiral_certificate_to_dict(certificate), 'behavior_fingerprint': '1e'+'9'*62}
    assert spiral_certificate_from_dict(payload).behavior_fingerprint == payload['behavior_fingerprint']
    behavior = RationalBehavior(Scenario((1,), (1,)), (1,))
    assert behavior_from_dict(behavior_to_dict(behavior), max_scalar_chars=1).probabilities == (1,)


def test_table_counts_fail_before_any_numeric_conversion(monkeypatch):
    import nonlocalgames.networks.behavior_serialization as behaviors
    import nonlocalgames.networks.serialization as witnesses
    import nonlocalgames.networks.inflation.certificates as certificates
    _, behavior, _ = _common_bit()
    payload = behavior_to_dict(behavior)
    witness = classical_to_dict(triangle_xor_strategy())
    _, _, certificate = _w_certificate()
    proof = spiral_certificate_to_dict(certificate)
    def forbidden(*args, **kwargs):
        pytest.fail('must validate vector counts before parsing any scalar')
    monkeypatch.setattr(behaviors, 'parse_rational', forbidden)
    monkeypatch.setattr(witnesses, 'parse_rational', forbidden)
    monkeypatch.setattr(certificates, 'parse_rational', forbidden)
    with pytest.raises(ValueError, match='entries'):
        behavior_from_dict({**payload, 'probabilities': ['1e1000000000']})
    with pytest.raises(ValueError, match='max_entries'):
        behavior_from_dict(payload, max_entries=7)
    damaged = deepcopy(witness)
    damaged['responses'][0] = ['1e1000000000']
    with pytest.raises(ValueError, match='sizes'):
        classical_from_dict(damaged)
    with pytest.raises(ValueError, match='max_entries'):
        classical_from_dict(witness, max_entries=1)
    with pytest.raises(ValueError, match='length'):
        spiral_certificate_from_dict({**proof, 'coefficients': ['1e1000000000']})


def test_pure_checkers_bound_direct_exact_inputs_before_arithmetic(monkeypatch):
    graph, behavior, violation = _common_bit()
    with pytest.raises(ScalarBudgetExceeded):
        verify_compatibility_violation(graph, behavior, 'triangle_finner', replace(violation, lhs=Fraction(1, 2**5000)))
    graph, w, certificate = _w_certificate()
    import nonlocalgames.networks.inflation._program as program
    monkeypatch.setattr(program, 'behavior_to_dict', lambda *a, **k: pytest.fail('must reject before hashing'))
    with pytest.raises(ScalarBudgetExceeded):
        verify_spiral_certificate(graph, w, certificate, max_scalar_bits=1)


def test_scalar_budget_failure_does_not_become_spiral_numerical_fallback():
    witness = triangle_xor_strategy()
    behavior = RationalBehavior(witness.scenario, witness.probabilities())
    with pytest.raises(ScalarBudgetExceeded):
        triangle_spiral(witness.network, behavior, max_scalar_bits=3)
    assert triangle_spiral(witness.network, behavior, max_scalar_bits=8).status == 'not_ruled_out'


def test_explicit_scalar_limits_can_be_adjusted_without_changing_data():
    behavior = RationalBehavior(Scenario((1,), (2,)), (Fraction(1, 257), Fraction(256, 257)))
    payload = behavior_to_dict(behavior)
    with pytest.raises(ScalarBudgetExceeded):
        behavior_from_dict(payload, max_scalar_bits=8)
    with pytest.raises(ScalarBudgetExceeded):
        behavior_from_dict(payload, max_scalar_bits=9, max_scalar_chars=6)
    assert behavior_from_dict(payload, max_scalar_bits=9, max_scalar_chars=7) == behavior


def test_compatibility_task_facade_forwards_scalar_limits(tmp_path):
    from nonlocalgames import ResourceSpec
    from nonlocalgames.core.compatibility_tasks import NetworkCompatibilityTask
    from nonlocalgames.core.compatibility_serialization import (compatibility_from_dict, compatibility_to_dict,
                                                                dump_compatibility, load_compatibility)
    behavior = RationalBehavior(Scenario((1,), (2,)), (Fraction(1, 257), Fraction(256, 257)))
    task = NetworkCompatibilityTask(behavior, ResourceSpec('local', network=SourceNetwork(1, ()), shared_randomness='forbidden'))
    payload = compatibility_to_dict(task)
    with pytest.raises(ScalarBudgetExceeded):
        compatibility_from_dict(payload, max_scalar_bits=8)
    assert compatibility_from_dict(payload, max_scalar_bits=9, max_scalar_chars=7) == task
    path = tmp_path / 'compatibility.json'
    dump_compatibility(task, path)
    with pytest.raises(ScalarBudgetExceeded):
        load_compatibility(path, max_scalar_bits=9, max_scalar_chars=6)
    assert load_compatibility(path, max_scalar_bits=9, max_scalar_chars=7) == task


def test_canonical_wire_number_spelling_and_spiral_hash_are_unchanged():
    import hashlib
    from nonlocalgames.core.identity import canonical_json
    graph, behavior, certificate = _w_certificate()
    payload = behavior_to_dict(behavior)
    assert payload['probabilities'] == list(map(str, behavior.probabilities))
    expected_hash = hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()
    assert certificate.behavior_fingerprint == expected_hash
    assert spiral_certificate_to_dict(certificate)['coefficients'] == list(map(str, certificate.coefficients))
    witness = triangle_xor_strategy()
    assert classical_to_dict(witness)['source_probabilities'] == [list(map(str, row)) for row in witness.source_probabilities]
    _, _, violation = _common_bit()
    assert violation_to_dict(violation)['lhs'] == str(violation.lhs)
