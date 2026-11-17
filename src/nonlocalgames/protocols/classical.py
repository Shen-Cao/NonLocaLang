"""Exact finite classical strategies for hidden kept/sent wirings."""
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod
from types import MappingProxyType

from ..core.numeric_data import positive_budget
from ..core.scalar_wire import parse_rational
from ..networks.behavior import RationalBehavior
from .wiring import WiringArena, WiringBudgetExceeded, WiringTarget, _index


def _sizes(values, parties):
    if not isinstance(values, (list, tuple)) or len(values) != parties:
        raise ValueError('Provide one port alphabet per origin')
    return tuple(positive_budget(size, 'port alphabet size') for size in values)


def _bounded_count(factors, limit, name):
    count = 1
    for size in factors:
        if size > limit // count:
            raise WiringBudgetExceeded(f'Classical wiring strategy exceeds {name}')
        count *= size
    return count


def _check_table(values, expected):
    if not isinstance(values, (list, tuple)) or len(values) != expected:
        raise ValueError('Wiring probability table has wrong shape')


def _probabilities(values, width, scalar_limits):
    values = tuple(parse_rational(p, **scalar_limits) for p in values)
    if any(p < 0 for p in values):
        raise ValueError('Wiring probabilities must be nonnegative')
    if any(sum(values[start:start+width]) != 1 for start in range(0, len(values), width)):
        raise ValueError('Each preparation and response row must be normalized')
    return values


@dataclass(frozen=True, init=False)
class ClassicalWiringStrategy:
    """Independent joint laws on each laboratory's two ports and stochastic local tables.

    Origin i prepares one keep-major/send-minor table. The response at (i,j)
    is indexed by local question, kept value at i, sent value from j, and answer.
    Each reachable (receiver,sender) pair has one table for the entire arena.
    """
    arena: WiringArena
    keep_sizes: tuple[int, ...]
    send_sizes: tuple[int, ...]
    preparations: tuple[tuple[Fraction, ...], ...]
    responses: Mapping

    def __init__(self, arena, keep_sizes, send_sizes, preparations, responses, *,
                 max_entries=1_000_000, max_scalar_bits=4096, max_scalar_chars=8192):
        if not isinstance(arena, WiringArena):
            raise TypeError('Expected a WiringArena')
        scenario = arena.scenario
        count = scenario.parties
        keep_sizes = _sizes(keep_sizes, count)
        send_sizes = _sizes(send_sizes, count)
        if not isinstance(preparations, (list, tuple)) or len(preparations) != count:
            raise ValueError('Provide one joint preparation table per origin')
        views = {(party, sender) for route in arena.routes for party, sender in enumerate(route)}
        if not isinstance(responses, Mapping) or len(responses) != len(views):
            raise ValueError('Provide one response table per reachable local view')
        tables = {}
        for key, table in responses.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError('Each response key must be a (party,sender) pair')
            party, sender = (_index(i, count, 'View index') for i in key)
            tables[party, sender] = table
        if tables.keys() != views:
            raise ValueError('Response keys differ from the reachable local views')

        remaining = positive_budget(max_entries, 'max_entries')
        for i, table in enumerate(preparations):
            size = _bounded_count((keep_sizes[i], send_sizes[i]), remaining, 'max_entries')
            _check_table(table, size)
            remaining -= size
        for (party, sender), table in tables.items():
            factors = (scenario.input_sizes[party], keep_sizes[party], send_sizes[sender],
                       scenario.output_sizes[party])
            size = _bounded_count(factors, remaining, 'max_entries')
            _check_table(table, size)
            remaining -= size
        limits = dict(max_bits=max_scalar_bits, max_chars=max_scalar_chars)
        laws = tuple(_probabilities(table, len(table), limits) for table in preparations)
        tables = {view: _probabilities(table, scenario.output_sizes[view[0]], limits)
                  for view, table in tables.items()}
        object.__setattr__(self, 'arena', arena)
        object.__setattr__(self, 'keep_sizes', keep_sizes)
        object.__setattr__(self, 'send_sizes', send_sizes)
        object.__setattr__(self, 'preparations', laws)
        object.__setattr__(self, 'responses', MappingProxyType(tables))

    def evaluation_work(self, *, max_work=10_000_000):
        """Count party responses in the full latent-assignment and event enumeration."""
        limit = positive_budget(max_work, 'max_work')
        scenario = self.arena.scenario
        factors = (scenario.parties, len(self.arena.routes), *scenario.input_sizes,
                   *scenario.output_sizes, *(len(table) for table in self.preparations))
        return _bounded_count(factors, limit, 'max_work')

    def evaluate(self, *, max_work=10_000_000):
        """Compute a rational target over every route using the same preparations and responses."""
        self.evaluation_work(max_work=max_work)
        scenario = self.arena.scenario
        inputs, outputs = scenario.inputs, scenario.outputs
        behavior = {route: [Fraction(0)]*scenario.entries for route in self.arena.routes}
        for latent in product(*(range(len(table)) for table in self.preparations)):
            weight = prod(law[index] for law, index in zip(self.preparations, latent))
            if not weight:
                continue
            kept, sent = zip(*(divmod(index, width) for index, width in zip(latent, self.send_sizes)))
            for route, probabilities in behavior.items():
                tables = tuple(self.responses[i, sender] for i, sender in enumerate(route))
                for xi, x in enumerate(inputs):
                    offsets = tuple(((x[i]*self.keep_sizes[i]+kept[i])*self.send_sizes[j]+sent[j])
                                    * scenario.output_sizes[i] for i, j in enumerate(route))
                    for ai, a in enumerate(outputs):
                        response = prod(table[offset+answer] for table, offset, answer in zip(tables, offsets, a))
                        probabilities[xi*len(outputs)+ai] += weight*response
        return WiringTarget(self.arena, {
            route: RationalBehavior(scenario, values, max_entries=scenario.entries)
            for route, values in behavior.items()})
