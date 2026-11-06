"""Composable restrictions on admissible strategies, separate from the verifier."""
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import TYPE_CHECKING

from .game import Game

if TYPE_CHECKING:
    from ..quantum.states import FixedState
    from ..networks.structure import SourceNetwork


class Resource(str, Enum):
    LOCAL = "local"
    QUANTUM_FINITE = "quantum_finite"
    QUANTUM_CLOSURE = "quantum_closure"
    QUANTUM_COMMUTING = "quantum_commuting"
    NONSIGNALLING = "nonsignalling"


class SharedRandomness(str, Enum):
    """External coordinated branch selection, beyond correlations inside a state."""
    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"


class StateConstraint(str, Enum):
    ANY = "any"
    PRODUCT = "product"
    PPT = "ppt"


class MeasurementClass(str, Enum):
    POVM = "povm"
    PROJECTIVE = "projective"


@dataclass(frozen=True)
class ResourceSpec:
    """Resource model and restrictions on each branch of a strategy.

    Dimensions bound the local quantum systems, excluding an explicitly allowed
    external classical branch label. Fixed states are identical in every branch.
    PRODUCT means a tensor product, not an arbitrary separable mixed state.
    PROJECTIVE allows zero projectors. These contracts do not imply a solver.
    """
    kind: Resource
    local_dimensions: tuple[int, ...] | None = None
    shared_randomness: SharedRandomness = field(default=SharedRandomness.ALLOWED, kw_only=True)
    state_constraint: StateConstraint = field(default=StateConstraint.ANY, kw_only=True)
    ppt_cuts: tuple[tuple[int, ...], ...] = field(default=(), kw_only=True)
    measurement_class: MeasurementClass = field(default=MeasurementClass.POVM, kw_only=True)
    fixed_state: "FixedState | None" = field(default=None, kw_only=True)
    network: "SourceNetwork | None" = field(default=None, kw_only=True)

    def __post_init__(self):
        for key, enum in (("kind", Resource), ("shared_randomness", SharedRandomness),
                          ("state_constraint", StateConstraint), ("measurement_class", MeasurementClass)):
            object.__setattr__(self, key, enum(getattr(self, key)))
        if self.network is not None:
            from ..networks.structure import SourceNetwork
            if not isinstance(self.network, SourceNetwork):
                raise TypeError("network must be a SourceNetwork")
            if self.kind not in (Resource.LOCAL, Resource.QUANTUM_FINITE):
                raise ValueError("Source networks currently require local or finite quantum resources")
            if self.shared_randomness != SharedRandomness.FORBIDDEN:
                raise ValueError("Independent-source networks require forbidden external shared randomness")
            if self.fixed_state is not None:
                raise ValueError("Declare independent source states in a network witness, not a global fixed state")
        dims = self.local_dimensions
        if dims is not None:
            dims = tuple(dims)
            if not dims or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1 for d in dims):
                raise ValueError("Local dimensions must be positive integer upper bounds")
            object.__setattr__(self, "local_dimensions", tuple(int(d) for d in dims))
        if self.kind != Resource.QUANTUM_FINITE and (
            dims is not None or self.fixed_state is not None or self.state_constraint != StateConstraint.ANY
            or self.measurement_class != MeasurementClass.POVM or self.ppt_cuts
        ):
            raise ValueError("State, dimension, and measurement restrictions require quantum_finite")
        cuts = tuple(tuple(cut) for cut in self.ppt_cuts)
        if bool(cuts) != (self.state_constraint == StateConstraint.PPT):
            raise ValueError("PPT requires explicit cuts, and cuts are only valid with PPT")
        for cut in cuts:
            if not cut or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in cut):
                raise ValueError("PPT cuts need nonnegative integer party indices")
            if len(set(cut)) != len(cut):
                raise ValueError("Repeated party in PPT cut")
        object.__setattr__(self, "ppt_cuts", cuts)
        if self.fixed_state is not None:
            from ..quantum.states import FixedState, product_state_residual
            if not isinstance(self.fixed_state, FixedState):
                raise TypeError("fixed_state must be a validated FixedState")
            actual = self.fixed_state.dimensions
            if dims is not None and (len(dims) != len(actual) or any(a > d for a, d in zip(actual, dims))):
                raise ValueError("Fixed state exceeds declared local dimension bounds")
            self.fixed_state.validate(ppt_cuts=cuts)
            if self.state_constraint == StateConstraint.PRODUCT and product_state_residual(
                self.fixed_state.state, actual
            ) > self.fixed_state.tolerance:
                raise ValueError("Fixed state does not satisfy the product-state restriction")
        count = len(dims) if dims is not None else (
            len(self.fixed_state.dimensions) if self.fixed_state is not None else (
                self.network.parties if self.network is not None else None))
        if self.network is not None and count is not None and self.network.parties != count:
            raise ValueError("Network and dimension bounds have different party counts")
        if cuts and count is not None:
            from ..quantum.states import validate_ppt_cuts
            object.__setattr__(self, "ppt_cuts", validate_ppt_cuts(cuts, count))


@dataclass(frozen=True)
class AnalysisTask:
    game: Game
    resource: ResourceSpec

    def __post_init__(self):
        if not isinstance(self.game, Game) or not isinstance(self.resource, ResourceSpec):
            raise TypeError("An AnalysisTask requires a Game and ResourceSpec")
        count = self.game.scenario.parties
        if self.resource.network is not None and self.resource.network.parties != count:
            raise ValueError("Network and game have different party counts")
        dims = self.resource.local_dimensions
        if dims is not None and len(dims) != count:
            raise ValueError("Provide one local dimension bound per party")
        fixed = self.resource.fixed_state
        if fixed is not None and len(fixed.dimensions) != count:
            raise ValueError("Fixed state and game have different parties")
        if self.resource.ppt_cuts:
            from ..quantum.states import validate_ppt_cuts
            validate_ppt_cuts(self.resource.ppt_cuts, count)

    def capabilities(self):
        """Return supported method names; witness evaluation is a separate operation."""
        from .dispatch import capabilities
        return capabilities(self)

    def solve(self, *, method=None, **options):
        """Run only a method that enforces the declared restrictions.

        Quantum search and relaxation methods must be explicitly requested.
        Dimension/state restrictions are never silently discarded for NPA/XOR.
        """
        from .dispatch import solve
        return solve(self, method, options)

    def evaluate(self, strategy):
        """Recheck and score a deterministic, quantum, or factorized network witness."""
        from .evaluation import evaluate
        return evaluate(self, strategy)
