"""Result records distinguish exact mathematics from numerical evidence."""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class AnalysisResult:
    """An optimization result, including its interpretation and raw witness.

    ``numerical_relaxation`` means that the mathematical program is an upper
    relaxation; its floating-point objective is NOT a certified upper bound.
    ``numerical_optimum`` likewise makes no rigorous numerical error claim.
    """

    value: Fraction | float
    resource: str
    method: str
    guarantee: str
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    witness: Any = None
