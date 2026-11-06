"""Primitives for finite nonlocal games and explicit resource queries."""
from .core import (AnalysisTask, Game, MeasurementClass, Resource, ResourceSpec,
                   Scenario, SharedRandomness, StateConstraint)

__version__ = "0.1.0"
__all__ = ["AnalysisTask", "Game", "Resource", "ResourceSpec", "SharedRandomness", "StateConstraint", "MeasurementClass", "Scenario"]

from .core import NetworkCompatibilityTask

__all__ += ["NetworkCompatibilityTask"]
