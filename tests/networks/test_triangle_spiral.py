"""Classical spiral inflation excludes W beyond Finner and retains valid models."""
from fractions import Fraction
from itertools import product
from math import prod
import random
from types import SimpleNamespace

import numpy as np
import pytest

from nonlocalgames import Scenario
from nonlocalgames.networks import ClassicalNetworkStrategy, RationalBehavior, SourceNetwork, triangle_finner, triangle_xor_strategy
from nonlocalgames.networks.inflation import triangle_spiral, verify_spiral_certificate
from nonlocalgames.networks.inflation._program import compile_spiral
from nonlocalgames.networks.inflation.certificates import make_certificate


def _w():
    graph = SourceNetwork(3, ((0, 1), (1, 2), (2, 0)))
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    return graph, RationalBehavior(scenario, tuple(Fraction(1, 3) if sum(a) == 1 else 0 for a in scenario.outputs))


def test_w_passes_finner_but_has_exact_classical_spiral_contradiction():
    graph, behavior = _w()
    assert triangle_finner(graph, behavior).status == "not_ruled_out"
    result = triangle_spiral(graph, behavior)
    assert result.status == "incompatible" and result.resource == "classical_fanout"
    assert result.guarantee == "exact_classical_incompatibility"
    assert result.diagnostics["farkas_margin"] < 0
    assert result.diagnostics["minimum_dual_slack"] >= 0
    assert (result.diagnostics["equalities"], result.diagnostics["variables"], result.diagnostics["nonzeros"]) == (40, 64, 320)
    assert verify_spiral_certificate(graph, behavior, result.certificate).status == "incompatible"


def test_valid_xor_and_degenerate_alphabets_are_only_not_ruled_out():
    strategy = triangle_xor_strategy()
    behavior = RationalBehavior(strategy.scenario, strategy.probabilities())
    result = triangle_spiral(strategy.network, behavior)
    assert result.status == "not_ruled_out" and result.guarantee == "exact_relaxation_feasibility"
    assert result.certificate.side == "primal"
    degenerate = RationalBehavior(Scenario((1, 1, 1), (1, 1, 1)), (1,))
    assert triangle_spiral(strategy.network, degenerate).status == "not_ruled_out"


def _inflate(strategy):
    edge = {frozenset(e): s for s, e in enumerate(strategy.network.sources)}
    ab, bc, ca = (edge[frozenset(e)] for e in ((0, 1), (1, 2), (0, 2)))
    source_laws = tuple(strategy.source_probabilities[s] for s in (ab, bc, ca, ab, bc, ca))
    sizes = strategy.scenario.output_sizes
    nodes = ((0, {ab: 0, ca: 2}), (1, {ab: 0, bc: 1}), (2, {bc: 1, ca: 2}),
             (0, {ab: 3, ca: 2}), (1, {ab: 0, bc: 4}), (2, {bc: 1, ca: 5}))
    q = [Fraction(0)] * prod(sizes)**2
    assignments = tuple(product(*(range(n) for n in sizes + sizes)))
    for latent in product(*(range(len(law)) for law in source_laws)):
        weight = prod((law[v] for law, v in zip(source_laws, latent)), start=Fraction(1))
        rows = []
        for party, context in nodes:
            row = 0
            for source in strategy.network.incident_sources(party):
                row = row*len(strategy.source_probabilities[source]) + latent[context[source]]
            rows.append(strategy.responses[party][row*sizes[party]:(row+1)*sizes[party]])
        for index, assignment in enumerate(assignments):
            q[index] += weight*prod((row[a] for row, a in zip(rows, assignment)), start=Fraction(1))
    return tuple(q)


def test_independent_nonbinary_classical_inflation_construction():
    rng = random.Random(71)
    def law(n):
        weights = [rng.randint(1, 6) for _ in range(n)]
        return tuple(Fraction(v, sum(weights)) for v in weights)
    for graph in (SourceNetwork(3, ((0, 1), (1, 2), (2, 0))), SourceNetwork(3, ((0, 2), (2, 1), (1, 0)))):
        scenario = Scenario((1, 1, 1), (2, 3, 2))
        sources = tuple(law(2) for _ in range(3))
        responses = tuple(tuple(v for _ in range(4) for v in law(n)) for n in scenario.output_sizes)
        strategy = ClassicalNetworkStrategy(graph, scenario, sources, responses)
        behavior = RationalBehavior(scenario, strategy.probabilities())
        program = compile_spiral(graph, behavior)
        result = verify_spiral_certificate(graph, behavior, make_certificate(program, "primal", _inflate(strategy)))
        assert result.status == "not_ruled_out" and result.guarantee == "exact_relaxation_feasibility"


def test_wrong_scope_and_allocation_guards_precede_enumeration(monkeypatch):
    graph, behavior = _w()
    import nonlocalgames.networks.inflation._program as program
    def forbidden(*args, **kwargs):
        pytest.fail("must reject before enumerating assignments")
    monkeypatch.setattr(program, "product", forbidden)
    for budget in ({"max_variables": 63}, {"max_equalities": 39}, {"max_nonzeros": 319}, {"max_variables": True}):
        with pytest.raises(ValueError):
            compile_spiral(graph, behavior, **budget)
    with pytest.raises(ValueError, match="Triangle"):
        compile_spiral(SourceNetwork(3, ((0, 1), (1, 2))), behavior)
    inputs = RationalBehavior(Scenario((2, 1, 1), (1, 1, 1)), (1, 1))
    with pytest.raises(ValueError, match="no external inputs"):
        compile_spiral(graph, inputs)


@pytest.mark.parametrize("status", [1, 2, 3, 4, "infeasible_inaccurate"])
def test_solver_failure_without_exact_certificate_never_excludes(monkeypatch, status):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **k: SimpleNamespace(status=status, message="simulated"))
    result = triangle_spiral(*_w())
    assert result.status == "unknown" and result.guarantee == "uncertified_solver_output"
    assert not result.diagnostics["certified"] and result.certificate is None


@pytest.mark.parametrize("candidate", [np.zeros((64, 1)), np.full(64, np.nan), ["bad"]*64, np.ones(64, dtype=complex)])
def test_malformed_primal_output_cannot_be_numeric_feasibility(monkeypatch, candidate):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **k: SimpleNamespace(status=0, message="simulated", x=candidate))
    assert triangle_spiral(*_w()).status == "unknown"


def test_false_solver_success_with_large_residual_is_unknown(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "linprog", lambda c, **k: SimpleNamespace(status=0, message="simulated", x=np.zeros(len(c))))
    result = triangle_spiral(*_w())
    assert result.status == "unknown" and result.diagnostics["numerical_primal_max_residual"] > .3


@pytest.mark.parametrize("options", [{"primal_feasibility_tolerance": 1e-12}, {"dual_feasibility_tolerance": -1},
    {"ipm_optimality_tolerance": 1e-15}, {"time_limit": float('nan')}, {"maxiter": -1},
    {"maxiter": 2**40}, {"presolve": "off"}, {"disp": 1}, {"bounds": (0, 1)}])
def test_solver_options_cannot_be_silently_reset(monkeypatch, options):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **k: pytest.fail("invalid options must fail before solve"))
    with pytest.raises(ValueError):
        triangle_spiral(*_w(), solver_options=options)
