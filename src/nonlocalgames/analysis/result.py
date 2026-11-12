"""Result records distinguish exact mathematics from numerical evidence."""

from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite
from numbers import Integral, Rational, Real
from typing import Any

from ..core.evidence_labels import EXACT_SCORE_GUARANTEES, VALUE_RESULT_GUARANTEES, Guarantee


@dataclass(frozen=True)
class AnalysisResult:
    """An optimization result, including its interpretation and raw witness.

    ``numerical_relaxation`` means that the mathematical program is an upper
    relaxation; its floating-point objective is NOT a certified upper bound.
    ``numerical_optimum`` likewise makes no rigorous numerical error claim.
    Runtime validation checks label and scalar types, not the truth of the claim.
    Even an exact-labeled record remains untrusted until the appropriate verifier
    checks its evidence and input/resource context.
    """

    value: Fraction | float
    resource: str
    method: str
    guarantee: Guarantee | str
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    witness: Any = None

    def __post_init__(self):
        guarantee = Guarantee(self.guarantee)
        if guarantee not in VALUE_RESULT_GUARANTEES:
            raise ValueError("This evidence label does not describe a scalar AnalysisResult")
        object.__setattr__(self, "guarantee", guarantee)
        for name in ("resource", "method", "status"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a nonempty string")
            if not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if guarantee in EXACT_SCORE_GUARANTEES:
            if isinstance(self.value, bool) or not isinstance(self.value, (Fraction, Integral)):
                raise TypeError("An exact score or certified bound must be a Fraction or integer, not a Boolean")
        elif isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise TypeError("An analysis score must be a finite real number, not a Boolean")
        elif not isinstance(self.value, Rational):
            try:
                finite = isfinite(self.value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("An analysis score must be finite") from error
            if not finite:
                raise ValueError("An analysis score must be finite")
