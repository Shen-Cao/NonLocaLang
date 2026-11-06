"""Source incidence data for independent-source, one-round networks."""
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class SourceNetwork:
    """A list of independent sources and their recipient parties.

    Parties are numbered 0 through ``parties-1``. Source order and endpoint
    order are preserved: both determine quantum tensor-factor conventions.
    Repeated source recipient sets are allowed and denote distinct independent
    sources. Empty source lists and isolated parties are allowed; empty sources
    are not. The model has no additional global shared randomness.
    """
    parties: int
    sources: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if isinstance(self.parties, bool) or not isinstance(self.parties, Integral) or self.parties < 1:
            raise ValueError("parties must be a positive integer")
        sources = tuple(tuple(source) for source in self.sources)
        for source in sources:
            if not source or any(isinstance(i, bool) or not isinstance(i, Integral)
                                 or not 0 <= i < self.parties for i in source):
                raise ValueError("Every source needs valid recipient parties")
            if len(set(source)) != len(source):
                raise ValueError("A source cannot contain a party more than once")
        object.__setattr__(self, "parties", int(self.parties))
        object.__setattr__(self, "sources", tuple(tuple(map(int, s)) for s in sources))

    def incident_sources(self, party):
        """Source indices incident to a party, in declared source order."""
        if isinstance(party, bool) or not isinstance(party, Integral) or not 0 <= party < self.parties:
            raise ValueError("party index is outside the network")
        return tuple(i for i, source in enumerate(self.sources) if party in source)

    @property
    def party_ports(self):
        """For each party, its (source index, endpoint index) tensor factors."""
        return tuple(tuple((s, source.index(i)) for s, source in enumerate(self.sources) if i in source)
                     for i in range(self.parties))
