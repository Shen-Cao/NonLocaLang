import json

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import classical_value
from nonlocalgames.games import chsh, ghz
from nonlocalgames.networks import (BILOCAL_SCENARIO, SourceNetwork, bilocal_entanglement_swapping,
                                   deterministic_network_strategy, triangle_xor_strategy)
from nonlocalgames.networks.serialization import (classical_from_dict, classical_to_dict,
                                                 network_from_dict, network_to_dict,
                                                 quantum_from_dict, quantum_to_dict)


def test_graph_and_classical_exact_roundtrip():
    strategy = triangle_xor_strategy()
    assert network_from_dict(json.loads(json.dumps(network_to_dict(strategy.network)))) == strategy.network
    decoded = classical_from_dict(json.loads(json.dumps(classical_to_dict(strategy))))
    assert decoded == strategy
    assert decoded.probabilities() == strategy.probabilities()
    payload = classical_to_dict(strategy)
    payload["responses"][0][0] = 1.0
    with pytest.raises(ValueError):
        classical_from_dict(payload)
    graph = network_to_dict(strategy.network)
    graph["global_randomness"] = True
    with pytest.raises(ValueError):
        network_from_dict(graph)


def test_quantum_roundtrip_preserves_sources_factor_order_and_behavior():
    strategy = bilocal_entanglement_swapping()
    decoded = quantum_from_dict(json.loads(json.dumps(quantum_to_dict(strategy))))
    assert decoded.network == strategy.network
    assert tuple(s.dimensions for s in decoded.sources) == tuple(s.dimensions for s in strategy.sources)
    np.testing.assert_array_equal(decoded.probabilities(BILOCAL_SCENARIO), strategy.probabilities(BILOCAL_SCENARIO))
    payload = quantum_to_dict(strategy)
    payload["encoding"] = "exact"
    with pytest.raises(ValueError):
        quantum_from_dict(payload)
    payload = quantum_to_dict(strategy)
    payload["sources"][0]["state"]["imag"] = [[0]]
    with pytest.raises(ValueError):
        quantum_from_dict(payload)
    payload = quantum_to_dict(strategy)
    payload["measurements"][0][0][0]["real"][0][0] = float("nan")
    with pytest.raises(ValueError):
        quantum_from_dict(payload)


@pytest.mark.parametrize("game", [chsh(), ghz(), Game.from_rule(Scenario((1, 1), (2, 2)),
                                                              lambda x, a: 3*a[0]-5*a[1])])
def test_every_classical_linear_optimum_lifts_to_every_network(game):
    ordinary = classical_value(game)
    n = game.scenario.parties
    graphs = [SourceNetwork(n, ()), SourceNetwork(n, (tuple(range(n)),)),
              SourceNetwork(n, tuple((i,) for i in range(n))),
              SourceNetwork(n, tuple((i, i+1) for i in range(n-1)))]
    for graph in graphs:
        witness = deterministic_network_strategy(graph, game.scenario, ordinary.witness)
        assert witness.score(game) == ordinary.value
        assert classical_from_dict(classical_to_dict(witness)).score(game) == ordinary.value


def test_deterministic_lift_rejects_invalid_tables():
    graph = SourceNetwork(2, ((0, 1),))
    for tables in [((0, 0),), ((0, 0), (0, 2)), ((0, 0), (False, 0))]:
        with pytest.raises(ValueError):
            deterministic_network_strategy(graph, chsh().scenario, tables)
