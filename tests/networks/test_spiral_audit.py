"""Independent combinatorial and context-binding checks for classical spiral inflation."""
from dataclasses import replace
from fractions import Fraction
from itertools import product
import json

import pytest

from nonlocalgames import Scenario
from nonlocalgames.networks import RationalBehavior, SourceNetwork, triangle_finner, triangle_xor_strategy
from nonlocalgames.networks.inflation import (
    SpiralCertificate, spiral_certificate_from_dict, spiral_certificate_to_dict,
    verify_spiral_certificate,
)
from nonlocalgames.networks.inflation._program import compile_spiral


def _triangle():
    return SourceNetwork(3, ((0, 1), (1, 2), (2, 0)))


def _w_behavior():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    return RationalBehavior(scenario, tuple(Fraction(1, 3) if sum(a) == 1 else 0
                                           for a in scenario.outputs))


def _certificate(program, side, coefficients):
    return SpiralCertificate(side=side, coefficients=tuple(coefficients),
                             output_sizes=program.output_sizes,
                             network_fingerprint=program.network_fingerprint,
                             behavior_fingerprint=program.behavior_fingerprint)


def test_analytic_w_dual_is_a_boolean_covering_inequality():
    behavior, network = _w_behavior(), _triangle()
    assert triangle_finner(network, behavior).status == 'not_ruled_out'
    program = compile_spiral(network, behavior)
    diagonal = [Fraction(0)]*40
    for index in (0, 15, 23, 31):
        diagonal[index] = Fraction(1)
    diagonal[39] = Fraction(-1)
    # This checks all 64 classical assignments independently of the sparse program builder.
    assert program.matrix.shape == (40, 64) and program.matrix.nnz == 320
    for column, assignment in enumerate(product(range(2), repeat=6)):
        a1, b1, c1, a2, b2, c2 = assignment
        covered = int((a1, b1, c1) == (0, 0, 0))
        covered += int((a2, b2, c1) == (1, 1, 1))
        covered += int((a1, b2, c2) == (1, 1, 1))
        covered += int((a2, b1, c2) == (1, 1, 1))
        covered -= int((a2, b2, c2) == (1, 1, 1))
        assert covered >= 0
        expected_rows = {4*a1 + 2*b1 + c1, 8 + 4*a2 + 2*b2 + c1,
                         16 + 4*a1 + 2*b2 + c2, 24 + 4*a2 + 2*b1 + c2,
                         32 + 4*a2 + 2*b2 + c2}
        actual_rows = set(map(int, program.matrix[:, column].nonzero()[0]))
        assert actual_rows == expected_rows
        assert sum(diagonal[row] for row in actual_rows) == covered
    assert sum(y*b for y, b in zip(diagonal, program.rhs)) == -Fraction(1, 27)
    checked = verify_spiral_certificate(network, behavior, _certificate(program, 'dual', diagonal))
    assert checked.status == 'incompatible'
    assert checked.guarantee == 'exact_classical_incompatibility'
    assert checked.resource == 'classical_fanout'


def test_analytic_certificate_rechecks_without_any_lp_solver(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *a, **k: pytest.fail('verification must not solve'))
    behavior, network = _w_behavior(), _triangle()
    program = compile_spiral(network, behavior)
    vector = [Fraction(0)]*40
    for i in (0, 15, 23, 31):
        vector[i] = Fraction(1)
    vector[39] = Fraction(-1)
    saved = json.loads(json.dumps(spiral_certificate_to_dict(_certificate(program, 'dual', vector))))
    checked = verify_spiral_certificate(network, behavior, saved)
    assert checked.guarantee == 'exact_classical_incompatibility'
    with pytest.raises(ValueError):
        verify_spiral_certificate(network, behavior, {**saved, 'resource': 'quantum'})
    with pytest.raises(ValueError):
        verify_spiral_certificate(SourceNetwork(3, ((1, 0), (1, 2), (2, 0))), behavior, saved)


def test_full_probability_context_distinguishes_equal_pair_marginals():
    network, scenario = _triangle(), Scenario((1, 1, 1), (2, 2, 2))
    even = RationalBehavior(scenario, tuple(Fraction(1, 4) if sum(a) % 2 == 0 else 0
                                            for a in scenario.outputs))
    odd = RationalBehavior(scenario, tuple(Fraction(1, 4) if sum(a) % 2 == 1 else 0
                                           for a in scenario.outputs))
    for group in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)):
        assert even.marginal(group, (0, 0, 0)) == odd.marginal(group, (0, 0, 0))
    first, second = compile_spiral(network, even), compile_spiral(network, odd)
    assert first.behavior_fingerprint != second.behavior_fingerprint
    assert first.rhs[8:] == second.rhs[8:] and first.rhs[:8] != second.rhs[:8]
    # A context change must be rejected before an otherwise invalid supplied vector is checked.
    certificate = _certificate(first, 'primal', (Fraction(0),)*64)
    with pytest.raises(ValueError, match='fingerprint|behavior|context'):
        verify_spiral_certificate(network, odd, certificate)


def test_primal_feasibility_means_only_relaxation_feasibility():
    strategy, network = triangle_xor_strategy(), _triangle()
    behavior = RationalBehavior(strategy.scenario, strategy.probabilities())
    program = compile_spiral(network, behavior)
    # Independent XOR source bits give an explicit six-output inflated law.
    vector = [Fraction(0)]*64
    for u1, u2, v1, v2, w1, w2 in product(range(2), repeat=6):
        assignment = (u1 ^ w1, u1 ^ v1, v1 ^ w1,
                      u2 ^ w1, u1 ^ v2, v1 ^ w2)
        index = sum(bit * 2**(5-i) for i, bit in enumerate(assignment))
        vector[index] += Fraction(1, 64)
    result = verify_spiral_certificate(network, behavior, _certificate(program, 'primal', vector))
    assert result.status == 'not_ruled_out'
    assert result.guarantee == 'exact_relaxation_feasibility'
    assert result.resource == 'classical_fanout'
    saved = spiral_certificate_to_dict(_certificate(program, 'primal', vector))
    copied = spiral_certificate_from_dict(saved)
    assert verify_spiral_certificate(network, behavior, copied).status == 'not_ruled_out'
    with pytest.raises(ValueError):
        verify_spiral_certificate(network, behavior, replace(copied, coefficients=(Fraction(1),)*64))


@pytest.mark.parametrize('candidate_kind', ['column', 'complex', 'nonnumeric', 'missing'])
def test_malformed_numerical_candidates_cannot_claim_feasibility(monkeypatch, candidate_kind):
    from types import SimpleNamespace
    import numpy as np
    import scipy.optimize
    from nonlocalgames.networks.inflation import triangle_spiral
    candidates = {'column': np.zeros((64, 1)), 'complex': np.zeros(64, dtype=complex) + 1j,
                  'nonnumeric': ['invalid']*64}
    primal_fields = {'status': 0, 'message': 'synthetic malformed candidate'}
    if candidate_kind != 'missing':
        primal_fields['x'] = candidates[candidate_kind]
    responses = iter((SimpleNamespace(**primal_fields),
                      SimpleNamespace(status=1, message='synthetic solver limit', x=None)))
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *a, **kw: next(responses))
    result = triangle_spiral(_triangle(), _w_behavior())
    assert result.status == 'unknown'
    assert result.guarantee == 'uncertified_solver_output'
    assert result.certificate is None


def test_nonbinary_middle_party_lift_uses_the_same_five_marginal_contracts():
    scenario = Scenario((1, 1, 1), (2, 3, 2))
    original = [Fraction(0)]*12
    for u, v, w in product(range(2), repeat=3):
        original[scenario.output_index((u ^ w, u+v, v ^ w))] += Fraction(1, 8)
    behavior = RationalBehavior(scenario, tuple(original))
    network = SourceNetwork(3, ((2, 1), (0, 2), (1, 0)))
    program = compile_spiral(network, behavior)
    assert program.matrix.shape == (60, 144) and program.matrix.nnz == 720
    inflated_sizes = (2, 3, 2, 2, 3, 2)
    candidate = [Fraction(0)]*144
    for u1, u2, v1, v2, w1, w2 in product(range(2), repeat=6):
        assignment = (u1 ^ w1, u1+v1, v1 ^ w1, u2 ^ w1, u1+v2, v1 ^ w2)
        index = 0
        for size, value in zip(inflated_sizes, assignment):
            index = index*size + value
        candidate[index] += Fraction(1, 64)
    result = verify_spiral_certificate(network, behavior, _certificate(program, 'primal', candidate))
    assert result.status == 'not_ruled_out'
    assert result.guarantee == 'exact_relaxation_feasibility'


def test_failed_numerical_primal_diagnostics_remain_json_portable(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    import numpy as np
    import scipy.optimize
    from nonlocalgames.cli import json_default
    from nonlocalgames.networks.inflation import triangle_spiral
    responses = iter((SimpleNamespace(status=0, message='overflowing finite candidate', x=np.full(64, 1e308)),
                      SimpleNamespace(status=1, message='limit', x=None)))
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *a, **kw: next(responses))
    result = triangle_spiral(_triangle(), _w_behavior())
    assert result.status == 'unknown'
    json.dumps(asdict(result), default=json_default, allow_nan=False)
