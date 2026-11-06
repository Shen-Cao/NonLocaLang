"""Explicit independent-source witnesses and network compatibility tests."""
from .structure import SourceNetwork
from .classical import ClassicalNetworkStrategy, deterministic_network_strategy
from .quantum import QuantumNetworkStrategy, QuantumSource
from .witnesses import BILOCAL_SCENARIO, BilocalWitness, bilocal_witness
from .examples import (bilocal_deterministic_branches, bilocal_entanglement_swapping,
                       triangle_xor_strategy)

__all__ = ["SourceNetwork", "ClassicalNetworkStrategy", "QuantumSource", "QuantumNetworkStrategy",
           "BilocalWitness", "BILOCAL_SCENARIO", "bilocal_witness", "triangle_xor_strategy",
           "bilocal_entanglement_swapping", "bilocal_deterministic_branches", "deterministic_network_strategy"]
