from fractions import Fraction
from itertools import product
import json

import pytest

from nonlocalgames import Scenario
from nonlocalgames.protocols.classical import ClassicalWiringStrategy
from nonlocalgames.protocols.classical_wire import classical_from_dict, classical_strategy_id, classical_to_dict
from nonlocalgames.protocols.examples import bit_exchange_strategy, bit_exchange_target
from nonlocalgames.protocols.wiring import WiringArena, WiringBudgetExceeded


def _strategy():
    arena = WiringArena(Scenario((2, 1), (2, 2)), ((0, 1), (1, 0)))
    preparations = ((Fraction(1, 2), 0, 0, 0, 0, Fraction(1, 2)), (Fraction(1, 4), Fraction(3, 4)))
    tables = {
        (0, 0): tuple(int(a == (x ^ kept ^ int(incoming == 2)))
                      for x, kept, incoming, a in product(range(2), range(2), range(3), range(2))),
        (0, 1): tuple(int(a == (x ^ kept ^ incoming))
                      for x, kept, incoming, a in product(range(2), repeat=4)),
        (1, 0): tuple(int(a == int(incoming == 2)) for incoming, a in product(range(3), range(2))),
        (1, 1): (Fraction(2, 3), Fraction(1, 3))*2,
    }
    return ClassicalWiringStrategy(arena, (2, 1), (3, 2), preparations, tables)


def test_shared_preparations_and_responses_reproduce_parity_correlations():
    assert bit_exchange_strategy().evaluate() == bit_exchange_target()


def test_mixed_port_sizes_sender_visibility_self_returns_and_private_randomness():
    strategy = _strategy()
    fixed, swapped = strategy.evaluate().behaviors
    assert fixed.probabilities == (Fraction(2, 3), Fraction(1, 3), 0, 0,
                                   0, 0, Fraction(2, 3), Fraction(1, 3))
    assert swapped.probabilities == tuple(Fraction(i, 8) for i in (1, 3, 3, 1, 3, 1, 1, 3))
    assert classical_from_dict(strategy.arena, json.loads(json.dumps(classical_to_dict(strategy)))) == strategy


def test_evaluation_budget_precedes_profile_or_latent_expansion(monkeypatch):
    import nonlocalgames.protocols.classical as evaluator
    strategy = _strategy()
    required = 2 * 2 * 8 * 6 * 2
    assert strategy.evaluation_work(max_work=required) == required
    strategy.evaluate(max_work=required)
    monkeypatch.setattr(evaluator, 'product', lambda *args: pytest.fail('must preflight enumeration'))
    monkeypatch.setattr(Scenario, 'inputs', property(lambda self: pytest.fail('must preflight inputs')))
    with pytest.raises(WiringBudgetExceeded, match='max_work'):
        strategy.evaluate(max_work=required-1)


def test_counts_and_shapes_precede_rational_conversion(monkeypatch):
    import nonlocalgames.protocols.classical as evaluator
    strategy = _strategy()
    monkeypatch.setattr(evaluator, 'parse_rational', lambda *args, **kwargs: pytest.fail('expected preflight'))
    payload = classical_to_dict(strategy)
    with pytest.raises(WiringBudgetExceeded, match='max_entries'):
        classical_from_dict(strategy.arena, payload, max_entries=1)
    payload['responses'][-1]['probabilities'].pop()
    with pytest.raises(ValueError, match='shape'):
        classical_from_dict(strategy.arena, payload)


@pytest.mark.parametrize('bad', ['negative', 'unnormalized', 'scalar_as_number', 'bool_label',
                                 'view_missing', 'view_duplicate', 'wire_exponent', 'wrong_arena'])
def test_witness_wire_rejects_invalid_laws_views_and_binding(bad):
    strategy = _strategy()
    arena = strategy.arena
    data = classical_to_dict(strategy)
    if bad == 'negative':
        data['preparations'][0][0] = '-1/2'
    elif bad == 'unnormalized':
        data['responses'][0]['probabilities'] = ['1']*len(data['responses'][0]['probabilities'])
    elif bad == 'scalar_as_number':
        data['preparations'][0][0] = 1
    elif bad == 'bool_label':
        data['responses'][0]['sender'] = False
    elif bad == 'view_missing':
        data['responses'].pop()
    elif bad == 'view_duplicate':
        data['responses'][1] = data['responses'][0]
    elif bad == 'wire_exponent':
        data['preparations'][0][0] = '1e1000000000'
    else:
        arena = WiringArena(arena.scenario, ((1, 0),))
    with pytest.raises(ValueError):
        classical_from_dict(arena, data)


def test_immutable_witness_id_survives_record_reordering_and_binds_exact_values():
    strategy = _strategy()
    data = classical_to_dict(strategy)
    data['responses'].reverse()
    restored = classical_from_dict(strategy.arena, data)
    assert classical_strategy_id(strategy) == classical_strategy_id(restored)
    with pytest.raises(TypeError):
        restored.responses[0, 0] = ()
    data['preparations'][1].reverse()
    assert classical_strategy_id(classical_from_dict(strategy.arena, data)) != classical_strategy_id(strategy)
