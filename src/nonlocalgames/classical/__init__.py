"""Explicit local witnesses, independent of optimization backends."""
from .strategy import DeterministicStrategy, deterministic_quantum_strategy

__all__ = ["DeterministicStrategy", "deterministic_quantum_strategy"]
