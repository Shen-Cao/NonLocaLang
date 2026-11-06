"""Adversarial combinations for resource scope and provenance boundaries."""
from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import AnalysisTask, ResourceSpec
from nonlocalgames.core.resource_serialization import resource_from_dict, resource_to_dict
from nonlocalgames.games import chsh
from nonlocalgames.networks import QuantumNetworkStrategy, SourceNetwork, bilocal_entanglement_swapping
from nonlocalgames.quantum import FixedState, QuantumStrategy, chsh_strategy


def test_approximately_product_fixed_state_never_takes_exact_reduction():
    reference = chsh_strategy()
    product = np.diag([1, 0, 0, 0]).astype(complex)
    near_product = (1-1e-9)*product + 1e-9*reference.state
    fixed = FixedState(near_product, (2, 2), tolerance=1e-7)
    task = AnalysisTask(chsh(), ResourceSpec("quantum_finite", fixed_state=fixed,
                                            state_constraint="product"))
    assert task.capabilities() == ("seesaw",)
    for method in (None, "local", "state_sdp", "xor", "npa"):
        with pytest.raises(NotImplementedError):
            task.solve(method=method)
    witness = QuantumStrategy(fixed.state, reference.measurements, tolerance=1e-7)
    evaluated = task.evaluate(witness)
    assert evaluated.guarantee == "numerical_witness"
    assert evaluated.diagnostics["certified"] is False


def test_fixed_measurement_state_result_does_not_report_global_game_value():
    measurements = tuple(tuple((np.eye(2)/2, np.eye(2)/2) for _ in range(2)) for _ in range(2))
    task = AnalysisTask(chsh(), ResourceSpec("quantum_finite", (2, 2)))
    result = task.solve(method="state_sdp", measurements=measurements)
    assert result.value == pytest.approx(.5)
    assert result.guarantee == "numerical_witness"
    assert result.method == "fixed_measurements_state_sdp"
    assert "not the global game optimum" in result.diagnostics["scope"]
    assert task.evaluate(result.witness).value == result.value


def test_resource_method_options_cannot_override_constraints():
    task = AnalysisTask(chsh(), ResourceSpec("quantum_finite", (2, 2),
                                            state_constraint="ppt", ppt_cuts=((0,),)))
    for options in ({"dimensions": (3, 3)}, {"ppt_cuts": ()},
                    {"fixed_state": FixedState(chsh_strategy().state, (2, 2))}):
        with pytest.raises(ValueError, match="ResourceSpec"):
            task.solve(method="seesaw", **options)
    with pytest.raises(ValueError, match="ResourceSpec"):
        task.solve(method="state_sdp", measurements=chsh_strategy().measurements, ppt_cuts=())
    for method in ("npa", "xor"):
        with pytest.raises(NotImplementedError):
            task.solve(method=method)


def test_legacy_schema_cannot_drop_new_restrictions_or_accept_unknown_fields():
    payload = resource_to_dict(ResourceSpec("local"))
    payload["schema"] = "nonlocalgames.resource.v1"
    with pytest.raises(ValueError):
        resource_from_dict(payload)  # v1 with a network field is not a v1 document.
    payload.pop("network")
    assert resource_from_dict(payload) == ResourceSpec("local")
    payload["schema"] = "nonlocalgames.resource.v2"
    with pytest.raises(ValueError):
        resource_from_dict(payload)
    payload = resource_to_dict(ResourceSpec("local"))
    payload["communication"] = "allowed"
    with pytest.raises(ValueError):
        resource_from_dict(payload)


def test_network_provenance_is_checked_before_forgetting_source_structure():
    from nonlocalgames import Game
    from nonlocalgames.networks import BILOCAL_SCENARIO
    strategy = bilocal_entanglement_swapping()
    game = Game.from_rule(BILOCAL_SCENARIO, lambda x, a: a[0] == a[2])
    resource = ResourceSpec("quantum_finite", network=strategy.network,
                            shared_randomness="forbidden")
    task = AnalysisTask(game, resource)
    reversed_graph = SourceNetwork(3, ((1, 0), (2, 1)))
    reordered = QuantumNetworkStrategy(reversed_graph, strategy.sources, strategy.measurements)
    with pytest.raises(ValueError, match="ordering"):
        task.evaluate(reordered)
    with pytest.raises(TypeError, match="QuantumNetworkStrategy"):
        task.evaluate(strategy.as_strategy())
    with pytest.raises(ValueError, match="global fixed state"):
        ResourceSpec("quantum_finite", network=strategy.network,
                     shared_randomness="forbidden", fixed_state=FixedState(
                         strategy.as_strategy().state, strategy.dimensions))


def test_quantum_product_network_exact_value_respects_one_dimensional_bounds():
    task = AnalysisTask(chsh(), ResourceSpec("quantum_finite", (1, 1),
                        state_constraint="product", measurement_class="projective",
                        network=SourceNetwork(2, ((0, 1),)), shared_randomness="forbidden"))
    result = task.solve()
    assert result.value == Fraction(3, 4) and result.guarantee == "exact"
    assert result.witness.dimensions == (1, 1)
    assert task.evaluate(result.witness).value == .75
