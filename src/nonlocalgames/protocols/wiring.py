"""Finite protocols with independent kept/sent preparations and one hidden permutation."""
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from numbers import Integral

from ..core.game import Scenario
from ..core.numeric_data import positive_budget
from ..networks.behavior import RationalBehavior

DEFAULT_MAX_PARTIES = 32
DEFAULT_MAX_ROUTES = 256
DEFAULT_MAX_TOTAL_ENTRIES = 1_000_000


class WiringBudgetExceeded(ValueError):
    """The requested wiring operation exceeds a declared count or size limit."""


def _index(value, size, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < size:
        raise ValueError(f"{label} must be an integer in its declared range")
    return int(value)


def _route(values, parties):
    entries = tuple(islice(iter(values), parties+1))
    if len(entries) != parties:
        raise ValueError("Each route needs one sender per receiver")
    route = tuple(_index(sender, parties, "Sender") for sender in entries)
    if len(set(route)) != parties:
        raise ValueError("The senders in a route must form a permutation")
    return route


@dataclass(frozen=True, init=False)
class WiringArena:
    """Each lab keeps one port from its own preparation and sends a second to a relay.

    route[i] names the origin of the sent port delivered to receiver i. Questions
    arrive after redistribution. Each lab knows its own question and incoming
    sender. The complete route is private to the independent passive relay.
    Branch targets describe all question tuples, without a sampling prior.
    """
    scenario: Scenario
    routes: tuple[tuple[int, ...], ...]

    def __init__(self, scenario, routes, *, max_routes=DEFAULT_MAX_ROUTES,
                 max_parties=DEFAULT_MAX_PARTIES, max_total_entries=DEFAULT_MAX_TOTAL_ENTRIES):
        if not isinstance(scenario, Scenario):
            raise TypeError("Expected a Scenario")
        max_routes = positive_budget(max_routes, "max_routes")
        max_parties = positive_budget(max_parties, "max_parties")
        max_total_entries = positive_budget(max_total_entries, "max_total_entries")
        if scenario.parties > max_parties:
            raise WiringBudgetExceeded("Wiring scenario exceeds max_parties")
        entries = 1
        for size in scenario.input_sizes + scenario.output_sizes:
            if size > max_total_entries // entries:
                raise WiringBudgetExceeded("Wiring behaviors exceed max_total_entries")
            entries *= size
        unique = set()
        for i, raw in enumerate(routes):
            if i >= max_routes:
                raise WiringBudgetExceeded("Wiring arena exceeds max_routes")
            if i >= max_total_entries // entries:
                raise WiringBudgetExceeded("Wiring behaviors exceed max_total_entries")
            route = _route(raw, scenario.parties)
            if route in unique:
                raise ValueError("Duplicate route in wiring arena")
            unique.add(route)
        if not unique:
            raise ValueError("Wiring arena needs at least one route")
        object.__setattr__(self, "scenario", scenario)
        object.__setattr__(self, "routes", tuple(sorted(unique)))

    def held_ports(self, route, receiver):
        """Persistent (origin, role) IDs of the retained and incoming port."""
        receiver = _index(receiver, self.scenario.parties, "Receiver")
        route = _route(route, self.scenario.parties)
        if route not in self.routes:
            raise ValueError("Route does not belong to this arena")
        return ((receiver, "kept"), (route[receiver], "sent"))

    def local_view(self, route, receiver, question):
        """Visible classical labels indexing the receiver's local response."""
        kept, incoming = self.held_ports(route, receiver)
        party, _ = kept
        question = _index(question, self.scenario.input_sizes[party], "Question")
        return party, question, incoming[0]


@dataclass(frozen=True, init=False)
class WiringTarget:
    """One exactly normalized nonsignalling behavior bound to each hidden route."""
    arena: WiringArena
    behaviors: tuple[RationalBehavior, ...]

    def __init__(self, arena, by_route):
        if not isinstance(arena, WiringArena):
            raise TypeError("Expected a WiringArena")
        if not isinstance(by_route, Mapping):
            raise TypeError("Provide behaviors keyed by route")
        if len(by_route) != len(arena.routes):
            raise ValueError("Provide one behavior for every route")
        branches = {}
        for key, behavior in by_route.items():
            if not isinstance(key, tuple):
                raise TypeError("Branch keys must be sender tuples")
            route = _route(key, arena.scenario.parties)
            if not isinstance(behavior, RationalBehavior):
                raise TypeError("Expected a RationalBehavior per route")
            if behavior.scenario != arena.scenario:
                raise ValueError("Branch behavior differs from the arena scenario")
            branches[route] = behavior
        if branches.keys() != set(arena.routes):
            raise ValueError("Target routes do not match its arena")
        object.__setattr__(self, "arena", arena)
        object.__setattr__(self, "behaviors", tuple(branches[r] for r in arena.routes))
