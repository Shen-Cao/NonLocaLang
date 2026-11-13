"""Exact structural scans with an explicit policy for zero-probability questions."""
from dataclasses import dataclass
from math import prod

from ..claims.types import TruthStatus
from .types import OffSupportPolicy, PropertyKind, StructuralCounterexample, STRUCTURAL_KINDS


@dataclass(frozen=True)
class StructuralCheck:
    truth: TruthStatus
    entries: int
    counterexample: StructuralCounterexample | None
    rule: str


def _profile(number, sizes):
    values = []
    for size in reversed(sizes):
        number, value = divmod(number, size)
        values.append(value)
    return tuple(reversed(values))


def scan_structure(game, kind, policy):
    """Recognize stored parity form, without searching over encodings or game lowerings.

    XOR predicates accept exactly one answer parity at every selected input.
    XOR scores merely depend on parity; constant and signed parity scores qualify.
    Recognition applies to any finite party count and does not imply that the
    bipartite XOR SDP backend can analyze that party count.
    """
    kind, policy = PropertyKind(kind), OffSupportPolicy(policy)
    if kind not in STRUCTURAL_KINDS:
        raise ValueError('This scanner only handles structural property kinds')
    scenario, checked = game.scenario, 0
    xor = kind in (PropertyKind.XOR_PREDICATE, PropertyKind.XOR_SCORE)
    if xor:
        for party, size in enumerate(scenario.output_sizes):
            if size != 2:
                return StructuralCheck(TruthStatus.DISPROVED, 0,
                    StructuralCounterexample('xor_requires_binary_outputs', party=party),
                    'binary_output_alphabets_required')
    width = prod(scenario.output_sizes)
    for xi, probability in enumerate(game.distribution):
        if policy == OffSupportPolicy.SAMPLED and probability == 0:
            continue
        offset = xi*width
        even = game.payoff[offset]
        odd = game.payoff[offset+1] if xor else None
        for ai in range(width):
            value = game.payoff[offset+ai]
            checked += 1
            reason = None
            if kind == PropertyKind.ACCEPTANCE and not 0 <= value <= 1:
                reason = 'reward_outside_acceptance_interval'
            elif kind == PropertyKind.BOOLEAN_PREDICATE and value not in (0, 1):
                reason = 'reward_not_boolean'
            elif kind == PropertyKind.XOR_SCORE:
                if value != (odd if ai.bit_count() % 2 else even):
                    reason = 'reward_depends_on_more_than_answer_parity'
            elif kind == PropertyKind.XOR_PREDICATE:
                if even not in (0, 1) or value != (1-even if ai.bit_count() % 2 else even):
                    reason = 'winning_predicate_is_not_exactly_one_parity'
            if reason is not None:
                counterexample = StructuralCounterexample(reason,
                    _profile(xi, scenario.input_sizes), _profile(ai, scenario.output_sizes))
                return StructuralCheck(TruthStatus.DISPROVED, checked, counterexample, 'exact_table_counterexample')
    return StructuralCheck(TruthStatus.PROVED, checked, None,
                           f'exact_{kind.value}_scan:{policy.value}')
