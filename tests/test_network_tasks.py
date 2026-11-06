from fractions import Fraction

import pytest

from nonlocalgames import AnalysisTask, Game, ResourceSpec
from nonlocalgames.core.resource_serialization import resource_from_dict, resource_to_dict
from nonlocalgames.games import chsh
from nonlocalgames.networks import SourceNetwork, bilocal_entanglement_swapping, triangle_xor_strategy


def test_network_contract_rejects_global_randomness_and_incompatible_models():
    network = SourceNetwork(2, ())
    for options in ({'kind': 'local'}, {'kind': 'nonsignalling', 'shared_randomness': 'forbidden'}):
        with pytest.raises(ValueError):
            ResourceSpec(network=network, **options)
    with pytest.raises(ValueError):
        AnalysisTask(chsh(), ResourceSpec('local', network=SourceNetwork(3, ()), shared_randomness='forbidden'))


def test_classical_network_exact_value_and_witness_are_distinct_from_membership():
    for graph in ((), ((0, 1),), ((0,), (1,))):
        task = AnalysisTask(chsh(), ResourceSpec('local', network=SourceNetwork(2, graph),
                                               shared_randomness='forbidden'))
        result = task.solve()
        assert result.value == Fraction(3, 4) and result.guarantee == 'exact'
        assert task.evaluate(result.witness).value == result.value
        assert task.evaluate(result.witness).guarantee == 'exact_witness'


def test_quantum_network_requires_source_factorization_and_preserves_restrictions():
    witness = bilocal_entanglement_swapping()
    from nonlocalgames.networks import BILOCAL_SCENARIO
    game = Game.from_rule(BILOCAL_SCENARIO, lambda x, a: a[0] == a[2])
    resource = ResourceSpec('quantum_finite', (2, 4, 2), network=witness.network,
                            shared_randomness='forbidden', measurement_class='projective')
    task = AnalysisTask(game, resource)
    assert task.evaluate(witness).value == pytest.approx(.5)
    assert task.capabilities() == ()
    for method in ('npa', 'ns', 'seesaw', 'state_sdp'):
        with pytest.raises(NotImplementedError):
            task.solve(method=method)
    with pytest.raises(TypeError):
        task.evaluate(witness.as_strategy())
    too_small = ResourceSpec('quantum_finite', (2, 2, 2), network=witness.network,
                             shared_randomness='forbidden')
    with pytest.raises(ValueError, match='dimension bounds'):
        AnalysisTask(game, too_small).evaluate(witness)


def test_network_and_legacy_resource_serialization():
    r = ResourceSpec('local', network=triangle_xor_strategy().network, shared_randomness='forbidden')
    assert resource_from_dict(resource_to_dict(r)) == r
    plain = resource_to_dict(ResourceSpec('local'))
    plain.pop('network')
    plain['schema'] = 'nonlocalgames.resource.v1'
    assert resource_from_dict(plain) == ResourceSpec('local')


def test_quantum_product_network_exact_strategy_can_be_rechecked():
    task = AnalysisTask(chsh(), ResourceSpec('quantum_finite', (2, 2),
                        state_constraint='product', network=SourceNetwork(2, ((0, 1),)),
                        shared_randomness='forbidden', measurement_class='projective'))
    result = task.solve()
    assert result.guarantee == 'exact' and result.value == Fraction(3, 4)
    assert task.evaluate(result.witness).value == .75
