"""Joint compatibility of a wiring target with one independent classical strategy."""
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
from math import prod

from ..claims.types import ExecutionStatus, TruthStatus
from ..core.identity import canonical_json
from ..core.numeric_data import positive_budget
from ..core.scalar_wire import scalar_string
from .classical import ClassicalWiringStrategy
from .classical_wire import classical_strategy_id
from .serialization import target_id
from .wiring import WiringBudgetExceeded, WiringTarget


RESOURCE = 'nonlocalgames.independent_classical_wiring.v1'
QUERY = 'nonlocalgames.wiring_compatibility.v1'


@dataclass(frozen=True)
class WiringMarginalConflict:
    """One output marginal differs across two routes with the same local input ports."""
    party: int
    sender: int
    question: int
    output: int
    first_route: tuple[int, ...]
    second_route: tuple[int, ...]
    first_probability: Fraction
    second_probability: Fraction


@dataclass(frozen=True)
class WiringAssessment:
    """Result of checking a target family for one shared strategy."""
    target_id: str
    query_id: str
    truth: TruthStatus
    execution: ExecutionStatus
    rule: str
    counterexample: WiringMarginalConflict | None = None
    strategy_id: str | None = None
    detail: str | None = None
    resource: str = field(default=RESOURCE, init=False)


def wiring_assessment_to_dict(receipt):
    """Encode a receipt with exact marginal probabilities as rational strings."""
    if not isinstance(receipt, WiringAssessment):
        raise TypeError('Expected a WiringAssessment')
    conflict = receipt.counterexample
    event = None
    if conflict is not None:
        event = {'party': conflict.party, 'sender': conflict.sender, 'question': conflict.question,
                 'output': conflict.output, 'first_route': list(conflict.first_route),
                 'second_route': list(conflict.second_route),
                 'first_probability': scalar_string(conflict.first_probability),
                 'second_probability': scalar_string(conflict.second_probability)}
    return {'schema': 'nonlocalgames.wiring_assessment.v1', 'target_id': receipt.target_id,
            'query_id': receipt.query_id, 'resource': receipt.resource, 'truth': receipt.truth.value,
            'execution': receipt.execution.value, 'rule': receipt.rule, 'counterexample': event,
            'strategy_id': receipt.strategy_id, 'detail': receipt.detail}


def _marginal_conflict(target):
    scenario = target.arena.scenario
    for party, count in enumerate(scenario.input_sizes):
        inputs = [0] * scenario.parties
        for question in range(count):
            inputs[party] = question
            seen = {}
            for route, behavior in zip(target.arena.routes, target.behaviors):
                # The retained port and local question are fixed in this loop.
                # The sender identifies the physical incoming port in this arena.
                sender = route[party]
                marginal = behavior.marginal((party,), inputs)
                if sender in seen:
                    first_route, first = seen[sender]
                    for (output,), probability in marginal.items():
                        if probability != first[(output,)]:
                            return WiringMarginalConflict(party, sender, question, output,
                                first_route, route, first[(output,)], probability)
                else:
                    seen[sender] = (route, marginal)
    return None


def assess_wiring(target, *, strategy=None, max_work=10_000_000):
    """Assess exact existence of one strategy realizing every target branch.

    The query allows arbitrary finite classical port alphabets and stochastic
    laws. A supplied witness certifies existence with explicit rational tables.
    Each branch is nonsignalling in the questions, which arrive after routing.
    A single-party marginal can therefore be checked at zero questions elsewhere.
    With the same incoming sender, a party holds the same two preparation ports
    in both routes and applies the same response to the same question.

    max_work counts probability entries scanned for marginalization, target
    entries compared, and the evaluator's joint-preparation terms. Preflight
    reserves the full marginal scan even if a contradiction is found early.
    Content-ID construction and the bit cost of arithmetic are outside this count.
    """
    if not isinstance(target, WiringTarget):
        raise TypeError('Expected a WiringTarget')
    limit = positive_budget(max_work, 'max_work')
    identifier = target_id(target)
    definition = {'target_id': identifier, 'resource': RESOURCE,
                  'quantifier': 'exists_one_strategy_for_all_routes'}
    query = QUERY + ':' + sha256(canonical_json(definition).encode()).hexdigest()

    def receipt(truth, execution, rule, **facts):
        return WiringAssessment(identifier, query, truth, execution, rule, **facts)

    unknown = TruthStatus.UNKNOWN
    completed = ExecutionStatus.COMPLETED
    if strategy is not None and (type(strategy) is not ClassicalWiringStrategy
                                 or strategy.arena != target.arena):
        return receipt(unknown, ExecutionStatus.INVALID_EVIDENCE, 'strategy_binding.v1',
                       detail='Expected a classical wiring strategy bound to this arena')

    arena = target.arena
    scenario = arena.scenario
    marginal_work = len(arena.routes) * sum(scenario.input_sizes) * prod(scenario.output_sizes)
    if marginal_work > limit:
        return receipt(unknown, ExecutionStatus.BUDGET_EXCEEDED, 'marginal_scan.v1',
                       detail='Local marginal check exceeds max_work')
    conflict = _marginal_conflict(target)
    if conflict:
        return receipt(TruthStatus.DISPROVED, completed, 'local_marginal_equality.v1',
                       counterexample=conflict)
    if strategy is None:
        return receipt(unknown, completed, 'local_marginals_consistent.v1')

    comparison_work = len(arena.routes) * scenario.entries
    remaining = limit - marginal_work - comparison_work
    if remaining < 1:
        return receipt(unknown, ExecutionStatus.BUDGET_EXCEEDED, 'strategy_evaluation.v1',
                       detail='Strategy evaluation and target comparison exceed max_work')
    try:
        candidate = strategy.evaluate(max_work=remaining)
    except WiringBudgetExceeded as error:
        return receipt(unknown, ExecutionStatus.BUDGET_EXCEEDED, 'strategy_evaluation.v1',
                       detail=str(error))
    evidence = classical_strategy_id(strategy)
    if candidate.behaviors == target.behaviors:
        return receipt(TruthStatus.PROVED, completed, 'joint_strategy_match.v1', strategy_id=evidence)
    return receipt(unknown, completed, 'candidate_differs.v1', strategy_id=evidence)
