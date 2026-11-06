"""Typed behavior-compatibility questions, distinct from scalar game-value tasks."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .resources import Resource, ResourceSpec

if TYPE_CHECKING:
    from ..networks.behavior import RationalBehavior


@dataclass(frozen=True)
class NetworkCompatibilityTask:
    """Check a rational behavior under an explicit independent-source resource.

    Passing a necessary condition is inconclusive. Complete membership is
    advertised only for the implemented unrestricted classical no-input chain.
    Classical fanout inflation must never be applied as a quantum exclusion test.
    """
    behavior: "RationalBehavior"
    resource: ResourceSpec

    def __post_init__(self):
        from ..networks.behavior import RationalBehavior
        if not isinstance(self.behavior, RationalBehavior) or not isinstance(self.resource, ResourceSpec):
            raise TypeError("Expected an exact RationalBehavior and ResourceSpec")
        if self.resource.network is None:
            raise ValueError("Network compatibility requires an explicit source graph")
        if self.resource.network.parties != self.behavior.scenario.parties:
            raise ValueError("Behavior and source graph have different party counts")
        dimensions = self.resource.local_dimensions
        if dimensions is not None and len(dimensions) != self.behavior.scenario.parties:
            raise ValueError("Dimension bounds and behavior have different parties")

    def capabilities(self):
        """List applicable checks, without invoking any optimization backend."""
        graph, scenario = self.resource.network, self.behavior.scenario
        methods = ["source_independence"]
        edges = {frozenset(edge) for edge in graph.sources}
        triangle = graph.parties == 3 and len(graph.sources) == 3 and edges == {
            frozenset((0, 1)), frozenset((1, 2)), frozenset((0, 2))}
        chain = graph.parties == 3 and len(graph.sources) == 2 and edges == {
            frozenset((0, 1)), frozenset((1, 2))}
        if triangle:
            methods.append("triangle_finner")
        if self.resource.kind == Resource.LOCAL and scenario.input_sizes == (1, 1, 1):
            if chain:
                methods.append("bilocal_chain")
            if triangle:
                methods.append("spiral")
        return tuple(methods)

    def check(self, *, method, **options):
        """Apply a check with its explicit scope; no automatic fallback is chosen."""
        if method not in self.capabilities():
            raise NotImplementedError(f"Check {method!r} is not applicable to this resource; "
                                      f"available checks: {self.capabilities()}")
        graph = self.resource.network
        if method == "source_independence":
            from ..networks.compatibility import check_source_independence
            return check_source_independence(graph, self.behavior, **options)
        if method == "triangle_finner":
            from ..networks.compatibility import triangle_finner
            return triangle_finner(graph, self.behavior, **options)
        if method == "bilocal_chain":
            from ..networks.membership import reconstruct_bilocal_chain
            return reconstruct_bilocal_chain(graph, self.behavior, **options)
        from ..networks.inflation import triangle_spiral
        return triangle_spiral(graph, self.behavior, **options)
