"""Resource assumptions belong to a query, separately from the verifier."""
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

from .game import Game


class Resource(str, Enum):
    LOCAL = "local"
    QUANTUM_FINITE = "quantum_finite"
    QUANTUM_CLOSURE = "quantum_closure"
    QUANTUM_COMMUTING = "quantum_commuting"
    NONSIGNALLING = "nonsignalling"


@dataclass(frozen=True)
class ResourceSpec:
    kind: Resource
    local_dimensions: tuple[int, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "kind", Resource(self.kind))
        if self.local_dimensions is not None:
            dims = tuple(self.local_dimensions)
            if self.kind != Resource.QUANTUM_FINITE:
                raise ValueError("Local dimensions only apply to tensor-product quantum resources")
            if not dims or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1
                               for d in dims):
                raise ValueError("Local dimensions must be positive integers")
            object.__setattr__(self, "local_dimensions", dims)


@dataclass(frozen=True)
class AnalysisTask:
    game: Game
    resource: ResourceSpec

    def __post_init__(self):
        if not isinstance(self.game, Game) or not isinstance(self.resource, ResourceSpec):
            raise TypeError("An AnalysisTask requires a Game and ResourceSpec")
        dims = self.resource.local_dimensions
        if dims is not None and len(dims) != self.game.scenario.parties:
            raise ValueError("Provide one local dimension per party")

    def solve(self, **options):
        """Optimize only where a backend matches the requested semantics.

        NPA is deliberately explicit: it is a relaxation, not a generic solver
        for the finite-dimensional quantum value.
        """
        from ..analysis import classical_value, nonsignalling_value
        if self.resource.kind == Resource.LOCAL:
            return classical_value(self.game, **options)
        if self.resource.kind == Resource.NONSIGNALLING:
            return nonsignalling_value(self.game, **options)
        raise NotImplementedError("Use npa_bound for an explicit relaxation or a quantum strategy "
                                  "for achievable evidence; unrestricted quantum solve is unsupported")
