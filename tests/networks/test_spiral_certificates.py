"""Saved spiral evidence is rechecked exactly without trusting solver output."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from itertools import repeat
import json
from types import SimpleNamespace

import numpy as np
import pytest

from nonlocalgames import Scenario
from nonlocalgames.networks import RationalBehavior, SourceNetwork, triangle_xor_strategy
from nonlocalgames.networks.inflation import (SpiralCertificate, spiral_certificate_from_dict,
    spiral_certificate_to_dict, triangle_spiral, verify_spiral_certificate)
from nonlocalgames.networks.inflation._program import compile_spiral
from nonlocalgames.networks.inflation.certificates import make_certificate


def _analytic_w():
    graph = SourceNetwork(3, ((0, 1), (1, 2), (2, 0)))
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    behavior = RationalBehavior(scenario, tuple(Fraction(1, 3) if sum(a) == 1 else 0 for a in scenario.outputs))
    program = compile_spiral(graph, behavior)
    dual = [Fraction(0)]*40
    for i in (0, 15, 23, 31):
        dual[i] = Fraction(1)
    dual[39] = Fraction(-1)
    return graph, behavior, make_certificate(program, "dual", dual)


def test_analytic_w_certificate_and_saved_roundtrip_use_no_solver(monkeypatch):
    import scipy.optimize
    graph, behavior, certificate = _analytic_w()
    payload = json.loads(json.dumps(spiral_certificate_to_dict(certificate)))
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **k: pytest.fail("verification must not solve"))
    checked = verify_spiral_certificate(graph, behavior, payload)
    assert checked.status == "incompatible" and checked.resource == "classical_fanout"
    assert checked.diagnostics["farkas_margin"] == Fraction(-1, 27)
    assert checked.diagnostics["minimum_dual_slack"] == 0
    assert spiral_certificate_from_dict(payload) == certificate


def test_certificate_tampering_context_resource_and_order_are_rejected():
    graph, behavior, certificate = _analytic_w()
    payload = spiral_certificate_to_dict(certificate)
    edits = [dict(resource="quantum"), dict(row_schema="future"), dict(schema="future"),
             dict(output_sizes=[2, 2, 3]), dict(behavior_fingerprint="0"*64),
             dict(network_fingerprint="0"*64), dict(side="primal"),
             dict(coefficients=["0"]*40), dict(coefficients=[0.0]*40), dict(extra=True)]
    for fields in edits:
        with pytest.raises((ValueError, TypeError)):
            verify_spiral_certificate(graph, behavior, {**payload, **fields})
    reversed_graph = SourceNetwork(3, tuple(reversed(graph.sources)))
    with pytest.raises(ValueError, match="different graph"):
        verify_spiral_certificate(reversed_graph, behavior, certificate)
    for field in payload:
        changed = deepcopy(payload)
        changed.pop(field)
        with pytest.raises(ValueError):
            verify_spiral_certificate(graph, behavior, changed)


def test_fingerprint_binds_full_behavior_not_only_low_order_marginals():
    strategy = triangle_xor_strategy()
    even = RationalBehavior(strategy.scenario, strategy.probabilities())
    odd = RationalBehavior(strategy.scenario, tuple(Fraction(1, 4) if sum(a) % 2 else 0
                                                    for a in strategy.scenario.outputs))
    for group in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)):
        assert even.marginal(group, (0, 0, 0)) == odd.marginal(group, (0, 0, 0))
    certificate = triangle_spiral(strategy.network, even).certificate
    with pytest.raises(ValueError, match="complete behavior"):
        verify_spiral_certificate(strategy.network, odd, certificate)


def test_vector_guards_reject_unbounded_iterators_before_reconstruction():
    graph, behavior, certificate = _analytic_w()
    with pytest.raises(ValueError, match="vector length"):
        verify_spiral_certificate(graph, behavior, replace(certificate, coefficients=repeat(0)))
    with pytest.raises(ValueError, match="max_coefficients"):
        spiral_certificate_to_dict(certificate, max_coefficients=39)
    with pytest.raises(ValueError, match="max_coefficients"):
        spiral_certificate_from_dict(spiral_certificate_to_dict(certificate), max_coefficients=39)
    with pytest.raises(TypeError):
        verify_spiral_certificate(graph, behavior, replace(certificate, coefficients=(0.0,)*40))
    with pytest.raises(TypeError):
        verify_spiral_certificate(graph, behavior, object())
    with pytest.raises(ValueError):
        verify_spiral_certificate(graph, behavior, SpiralCertificate('dual', (), (2, 2, 2), 'bad', 'bad'))


def test_exact_dual_repair_is_verified_even_with_limit_status(monkeypatch):
    import scipy.optimize
    graph, behavior, certificate = _analytic_w()
    candidate = np.asarray(certificate.coefficients, dtype=float)
    candidate[0] -= 1/1000
    outputs = iter([SimpleNamespace(status=2, message="infeasible", x=None),
                    SimpleNamespace(status=1, message="time limit", x=candidate)])
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **k: next(outputs))
    result = triangle_spiral(graph, behavior)
    assert result.status == 'incompatible' and result.diagnostics['dual_status'] == 1
    assert result.diagnostics['dual_repair_shift'] == Fraction(1, 1000)
    assert result.diagnostics['farkas_margin'] == -Fraction(1, 27) + Fraction(1, 1000)
    assert verify_spiral_certificate(graph, behavior, result.certificate).status == 'incompatible'


def test_coarse_reconstruction_retains_only_numerical_relaxation_evidence():
    graph, _, _ = _analytic_w()
    scenario = Scenario((1, 1, 1), (1, 1, 2))
    behavior = RationalBehavior(scenario, (Fraction(1, 3), Fraction(2, 3)))
    result = triangle_spiral(graph, behavior, max_denominator=1)
    assert result.status == 'not_ruled_out'
    assert result.guarantee == 'numerical_relaxation_feasibility'
    assert result.certificate is None and not result.diagnostics['certified']


def test_finite_overflowing_solver_vector_remains_json_safe(monkeypatch):
    import scipy.optimize
    outputs = iter([SimpleNamespace(status=0, message='malformed huge candidate', x=np.full(64, 1e308)),
                    SimpleNamespace(status=4, message='failed', x=None)])
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *a, **k: next(outputs))
    graph, behavior, _ = _analytic_w()
    result = triangle_spiral(graph, behavior)
    assert result.status == 'unknown'
    assert result.diagnostics['numerical_primal_max_residual'] is None
    json.dumps(result.diagnostics, default=str, allow_nan=False)


@pytest.mark.parametrize('text', ['1e1000000000', '1/0', '1/8'])
def test_creation_and_encoding_never_parse_unbounded_text(text):
    graph, behavior, certificate = _analytic_w()
    program = compile_spiral(graph, behavior)
    with pytest.raises(TypeError, match='already-exact'):
        make_certificate(program, 'dual', (text,)*40)
    with pytest.raises(TypeError, match='already-exact'):
        spiral_certificate_to_dict(replace(certificate, coefficients=(text,)*40))
    with pytest.raises(ValueError, match='vector length'):
        make_certificate(program, 'dual', repeat(text))


def test_encoding_does_not_add_hidden_default_bit_limit_to_verified_coefficients():
    graph, behavior, certificate = _analytic_w()
    factor = 2**5000
    candidate = make_certificate(compile_spiral(graph, behavior), 'dual',
                                 tuple(factor*v for v in certificate.coefficients))
    checked = verify_spiral_certificate(graph, behavior, candidate, max_scalar_bits=6000)
    assert checked.diagnostics['farkas_margin'] == -Fraction(factor, 27)
    payload = spiral_certificate_to_dict(checked.certificate)
    assert spiral_certificate_from_dict(payload, max_scalar_bits=6000) == checked.certificate
    assert verify_spiral_certificate(graph, behavior, payload, max_scalar_bits=6000).status == 'incompatible'
