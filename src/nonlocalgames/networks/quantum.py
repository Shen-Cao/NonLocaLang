"""Finite quantum witnesses with explicitly independent source states."""
from dataclasses import dataclass, field
from functools import reduce
from math import prod
from numbers import Integral

import numpy as np

from ..core._tensor import _basis_indices
from ..quantum.strategy import QuantumStrategy
from ..quantum.states import FixedState, state_diagnostics
from .structure import SourceNetwork


@dataclass(frozen=True, eq=False)
class QuantumSource:
    """A density matrix on endpoint factors in the source's declared order."""
    state: np.ndarray
    dimensions: tuple[int, ...]
    tolerance: float = 1e-9

    def __post_init__(self):
        dims = tuple(self.dimensions)
        if not dims or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1 for d in dims):
            raise ValueError("Source endpoint dimensions must be positive integers")
        dims = tuple(map(int, dims))
        validator = FixedState(self.state, dims, self.tolerance)
        object.__setattr__(self, "dimensions", dims)
        object.__setattr__(self, "state", validator.state)


@dataclass(frozen=True, eq=False)
class QuantumNetworkStrategy:
    """Product source state with joint local POVMs on each party's incoming factors.

    Local factor order follows source index. Source factors are originally in
    source-major, endpoint-major order; a permutation groups them by party.
    Isolated parties have dimension one and may use local stochastic POVMs.
    All feasibility checks and scores are numerical, not rigorous certificates.
    """
    network: SourceNetwork
    sources: tuple[QuantumSource, ...]
    measurements: tuple
    tolerance: float = 1e-9
    max_total_dimension: int = 1024
    _strategy: QuantumStrategy = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.network, SourceNetwork):
            raise TypeError("network must be a SourceNetwork")
        if isinstance(self.tolerance, bool) or not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        sources = tuple(self.sources)
        if len(sources) != len(self.network.sources) or any(not isinstance(s, QuantumSource) for s in sources):
            raise ValueError("Provide exactly one QuantumSource per declared independent source")
        if any(len(s.dimensions) != len(endpoints) for s, endpoints in zip(sources, self.network.sources)):
            raise ValueError("Every source needs one factor dimension per endpoint")
        if (isinstance(self.max_total_dimension, bool) or not isinstance(self.max_total_dimension, Integral)
                or self.max_total_dimension < 1):
            raise ValueError("max_total_dimension must be a positive integer")
        for source in sources:
            state_diagnostics(source.state, source.dimensions, tolerance=self.tolerance)
        dims = tuple(d for source in sources for d in source.dimensions)
        total = prod(dims)
        if total > self.max_total_dimension:
            raise ValueError("Product source dimension exceeds max_total_dimension")
        offsets, cursor = [], 0
        for source in sources:
            offsets.append(cursor)
            cursor += len(source.dimensions)
        permutation = tuple(offsets[s] + endpoint for ports in self.network.party_ports for s, endpoint in ports)
        state = reduce(np.kron, (source.state for source in sources), np.ones((1, 1)))
        if permutation != tuple(range(len(dims))):
            indices = _basis_indices(dims, permutation, total)
            state = state[np.ix_(indices, indices)]
        strategy = QuantumStrategy(state, self.measurements, self.tolerance)
        expected = tuple(prod(sources[s].dimensions[e] for s, e in ports) for ports in self.network.party_ports)
        if strategy.dimensions != expected:
            raise ValueError("Local POVM dimensions do not match the incident source factors")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "measurements", strategy.measurements)
        object.__setattr__(self, "_strategy", strategy)

    @property
    def dimensions(self):
        return self._strategy.dimensions

    def as_strategy(self):
        """Forget source structure for evaluation; this is not a network equivalence."""
        return self._strategy

    def probabilities(self, scenario):
        return self._strategy.probabilities(scenario)

    def score(self, game):
        return self._strategy.score(game)

    def validate(self):
        return self._strategy.validate()
