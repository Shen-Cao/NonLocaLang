import json
from fractions import Fraction
from math import prod

from nonlocalgames import Scenario
from nonlocalgames.networks.behavior import RationalBehavior
from nonlocalgames.protocols.classical import ClassicalWiringStrategy
from nonlocalgames.protocols.classical_wire import classical_strategy_id
from nonlocalgames.protocols.compatibility import assess_wiring, wiring_assessment_to_dict
from nonlocalgames.protocols.examples import ABC, AC, bit_exchange_strategy, bit_exchange_target
from nonlocalgames.protocols.serialization import target_id
from nonlocalgames.protocols.wiring import WiringArena, WiringTarget


def _all_zero_strategy(arena):
    scenario = arena.scenario
    sizes = (1,) * scenario.parties
    tables = {(i, sender): tuple(int(a == 0) for _ in range(scenario.input_sizes[i])
                                for a in range(scenario.output_sizes[i]))
              for route in arena.routes for i, sender in enumerate(route)}
    return ClassicalWiringStrategy(arena, sizes, sizes, ((1,),) * scenario.parties, tables)


def test_single_uniform_strategy_reproduces_all_routes_with_distinct_joint_laws():
    strategy = bit_exchange_strategy()
    target = bit_exchange_target()
    necessary = assess_wiring(target)
    witness = assess_wiring(target, strategy=strategy)
    assert (necessary.truth, necessary.execution) == ('unknown', 'completed')
    assert (witness.truth, witness.execution) == ('proved', 'completed')
    assert witness.strategy_id == classical_strategy_id(strategy)
    assert witness.target_id == target_id(target) and witness.query_id == necessary.query_id
    assert witness.query_id != witness.target_id and witness.counterexample is None


def test_conflict_records_a_replayable_event_at_nonzero_local_question():
    scenario = Scenario((2, 2, 1), (2, 1, 1))
    arena = WiringArena(scenario, (AC, ABC))

    def branch(p):
        return RationalBehavior(scenario, tuple(v for x in scenario.inputs
                                for v in ((1, 0) if x[0] == 0 else (p, 1-p))))

    target = WiringTarget(arena, {ABC: branch(Fraction(1, 3)), AC: branch(Fraction(2, 3))})
    result = assess_wiring(target, strategy=_all_zero_strategy(arena))
    assert (result.truth, result.execution) == ('disproved', 'completed')
    event = result.counterexample
    assert (event.party, event.sender, event.question, event.output) == (0, 2, 1, 0)
    assert (event.first_route, event.second_route) == (ABC, AC)
    for route, probability in ((event.first_route, event.first_probability),
                               (event.second_route, event.second_probability)):
        behavior = target.behaviors[arena.routes.index(route)]
        assert behavior.marginal((0,), (1, 1, 0))[(event.output,)] == probability
    data = json.loads(json.dumps(wiring_assessment_to_dict(result)))
    assert data['counterexample']['first_probability'] == '1/3'
    assert data['counterexample']['second_probability'] == '2/3'
    changed = WiringTarget(arena, {ABC: branch(Fraction(2, 3)), AC: branch(Fraction(2, 3))})
    assert assess_wiring(changed).query_id != result.query_id
    assert assess_wiring(changed).truth == 'unknown'


def test_lack_of_local_contradiction_and_failed_candidate_leave_existence_open():
    target = bit_exchange_target()
    wrong = _all_zero_strategy(target.arena)
    miss = assess_wiring(target, strategy=wrong)
    assert (miss.truth, miss.execution) == ('unknown', 'completed')
    assert miss.strategy_id == classical_strategy_id(wrong)
    assert miss.counterexample is None
    invalid = assess_wiring(target, strategy=miss)
    assert (invalid.truth, invalid.execution) == ('unknown', 'invalid_evidence')
    other_arena = WiringArena(target.arena.scenario, (AC,))
    invalid = assess_wiring(target, strategy=_all_zero_strategy(other_arena))
    assert (invalid.truth, invalid.execution) == ('unknown', 'invalid_evidence')


def test_one_common_budget_covers_marginal_scan_evaluation_and_comparison():
    target, strategy = bit_exchange_target(), bit_exchange_strategy()
    scenario, n = target.arena.scenario, len(target.arena.routes)
    scan = n * sum(scenario.input_sizes) * prod(scenario.output_sizes)
    total = scan + n * scenario.entries + strategy.evaluation_work()
    for limit in (scan - 1, total - 1):
        result = assess_wiring(target, strategy=strategy, max_work=limit)
        assert (result.truth, result.execution) == ('unknown', 'budget_exceeded')
    assert assess_wiring(target, strategy=strategy, max_work=total).truth == 'proved'
