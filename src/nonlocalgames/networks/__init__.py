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

from .behavior import RationalBehavior
from .behavior_serialization import behavior_from_dict, behavior_to_dict
from .compatibility import (CompatibilityResult, CompatibilityViolation, check_source_independence,
                            source_disjoint_pairs, triangle_finner)
from .compatibility_certificates import (verify_compatibility_violation, violation_from_dict,
                                         violation_to_dict)
from .membership import BilocalMembershipResult, reconstruct_bilocal_chain

__all__ += ["RationalBehavior", "behavior_to_dict", "behavior_from_dict", "CompatibilityResult",
            "CompatibilityViolation", "check_source_independence", "source_disjoint_pairs",
            "triangle_finner", "verify_compatibility_violation", "violation_to_dict",
            "violation_from_dict", "BilocalMembershipResult", "reconstruct_bilocal_chain"]
