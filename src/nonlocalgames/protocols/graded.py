"""Exact, signed permutations of fermionic occupation basis states.

Bit i in an occupation mask denotes mode i. Creation operators in a basis
ket are in increasing mode order. Nothing here constructs a Fock-space matrix.
"""
from dataclasses import dataclass
from itertools import accumulate, islice
from numbers import Integral

from ..core.numeric_data import positive_budget
from .wiring import WiringBudgetExceeded, _index


DEFAULT_MAX_MODES = 256


def _bounded_items(items, limit, message):
    for i, item in enumerate(items):
        if i == limit:
            raise WiringBudgetExceeded(message)
        yield item


@dataclass(frozen=True, init=False)
class ModePermutation:
    """destinations[i] is the new position of source mode i (not its inverse)."""

    destinations: tuple[int, ...]

    def __init__(self, destinations, *, max_modes=DEFAULT_MAX_MODES):
        limit = positive_budget(max_modes, "max_modes")
        items = tuple(_bounded_items(destinations, limit, "Mode permutation exceeds max_modes"))
        perm = tuple(_index(d, len(items), "Destination") for d in items)
        if len(set(perm)) != len(perm):
            raise ValueError("Mode destinations must form a permutation")
        object.__setattr__(self, "destinations", perm)

    @classmethod
    def from_packets(cls, destinations, widths, *, max_modes=DEFAULT_MAX_MODES):
        """Move contiguous packets without reversing them; widths are in source order."""
        limit = positive_budget(max_modes, "max_modes")
        packets = cls(destinations, max_modes=limit).destinations
        widths = tuple(islice(iter(widths), len(packets) + 1))
        if len(widths) != len(packets):
            raise ValueError("Provide one width per source packet")
        widths = tuple(positive_budget(w, "Packet width") for w in widths)
        if sum(widths) > limit:
            raise WiringBudgetExceeded("Packet expansion exceeds max_modes")
        at_destination = [0] * len(packets)
        for d, w in zip(packets, widths):
            at_destination[d] = w
        starts = tuple(accumulate(at_destination, initial=0))
        return cls((starts[d] + j for d, w in zip(packets, widths) for j in range(w)),
                   max_modes=limit)

    def apply(self, occupied: int) -> tuple[int, int]:
        """Return (destination mask, amplitude sign) for one canonical basis ket."""
        if isinstance(occupied, bool) or not isinstance(occupied, Integral):
            raise ValueError("Occupation mask must be a nonnegative integer")
        occupied = int(occupied)
        if occupied < 0 or occupied.bit_length() > len(self.destinations):
            raise ValueError("Occupation mask outside the mode register")
        result = parity = 0
        while occupied:
            first = occupied & -occupied
            d = self.destinations[first.bit_length() - 1]
            # Previous particles were earlier at the source. Those later at
            # the destination must anticommute past this creation operator.
            parity ^= (result >> (d + 1)).bit_count() & 1
            result |= 1 << d
            occupied ^= first
        return result, -1 if parity else 1

    def inverse(self):
        """Reverse this relabeling, including its signed action on basis kets."""
        result = [0] * len(self.destinations)
        for source, destination in enumerate(self.destinations):
            result[destination] = source
        return ModePermutation(result, max_modes=max(1, len(result)))

    def then(self, after):
        """Compose in execution order: apply self first, then after."""
        if not isinstance(after, ModePermutation):
            raise TypeError("Expected a ModePermutation")
        if len(self.destinations) != len(after.destinations):
            raise ValueError("Mode registers must have equal size")
        return ModePermutation((after.destinations[d] for d in self.destinations),
                               max_modes=max(1, len(self.destinations)))


def relay_mode_permutation(route, *, modes_per_port=1, max_modes=DEFAULT_MAX_MODES):
    """Route sent packets, retaining the other packet at each lab.

    route[receiver] is the *sender*. Source order is (sent, kept) at each
    origin; output order is (incoming, kept) at each receiver. All ports
    have the same positive width. Intra-packet mode order is unchanged.
    Admission counts both ports per party against max_modes before expansion.
    """
    width = positive_budget(modes_per_port, "modes_per_port")
    limit = positive_budget(max_modes, "max_modes")
    party_limit = limit // (2*width)
    senders = _bounded_items(route, party_limit, "Relay packet expansion exceeds max_modes")
    sender_to_receiver = ModePermutation(senders, max_modes=limit).inverse().destinations
    packets = tuple(d for i, r in enumerate(sender_to_receiver) for d in (2*r, 2*i+1))
    return ModePermutation.from_packets(packets, (width,) * len(packets), max_modes=limit)
