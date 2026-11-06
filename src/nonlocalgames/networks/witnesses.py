"""Nonlinear compatibility witnesses for specified independent-source models."""
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, sqrt
from numbers import Real

import numpy as np

from ..core.game import Scenario, rational

BILOCAL_SCENARIO = Scenario((2, 1, 2), (2, 4, 2))


@dataclass(frozen=True)
class BilocalWitness:
    """Evaluation of the Branciard bilocal inequality, not a game-value bound.

    ``value`` is a floating display of sqrt(|I|)+sqrt(|J|). For rational input
    data, ``violates_bound`` is established by exact rational comparisons,
    independent of the displayed square root. The certificate concerns the
    supplied distribution, not confidence intervals for sampled experimental data.
    """
    I: Fraction | float  # noqa: E741
    J: Fraction | float
    value: float
    bound: int
    violates_bound: bool
    guarantee: str
    assumptions: tuple[str, ...]


def bilocal_witness(probabilities, *, scenario=BILOCAL_SCENARIO, tolerance=1e-9):
    """Check sqrt(|I|)+sqrt(|J|)<=1 for the A--B--C classical bilocal model.

    Alice and Charlie have binary inputs/outputs; Bob has one input and four
    outputs b=2*b0+b1. I averages <A_x B0 C_z>; J averages
    (-1)^(x+z)<A_x B1 C_z>. Source AB and source BC must be independent.
    Data must be flat in the ordinary input-major/output-minor order.

    Reference: Branciard et al., https://arxiv.org/abs/1112.4502.
    """
    if scenario != BILOCAL_SCENARIO:
        raise ValueError("Bilocal witness requires input sizes (2,1,2) and outputs (2,4,2)")
    if not isinstance(tolerance, Real) or not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    values = tuple(probabilities)
    if len(values) != scenario.entries:
        raise ValueError("Probability data does not match the bilocal scenario")
    if any(isinstance(v, (complex, np.complexfloating)) for v in values):
        raise TypeError("Probabilities must be real")
    exact = not any(isinstance(v, (float, np.floating)) for v in values)
    values = tuple(map(rational, values)) if exact else tuple(map(float, values))
    width = len(scenario.outputs)
    if exact:
        valid = all(v >= 0 for v in values) and all(sum(values[k:k+width]) == 1 for k in range(0, len(values), width))
    else:
        valid = all(isfinite(v) and v >= -tolerance for v in values) and all(
            abs(sum(values[k:k+width]) - 1) <= tolerance for k in range(0, len(values), width))
    if not valid:
        raise ValueError("Each input's output probabilities must be nonnegative and normalized")
    correlator_i = correlator_j = Fraction(0) if exact else 0.0
    for xi, (x, _, z) in enumerate(scenario.inputs):
        for ai, (a, b, c) in enumerate(scenario.outputs):
            p = values[xi * width + ai]
            correlator_i += (-1)**(a + (b // 2) + c) * p / 4
            correlator_j += (-1)**(x + z + a + (b % 2) + c) * p / 4
    value = sqrt(abs(float(correlator_i))) + sqrt(abs(float(correlator_j)))
    if exact:
        u, v = abs(correlator_i), abs(correlator_j)
        remainder = 1 - u - v
        violates = remainder < 0 or 4*u*v > remainder**2
    else:
        violates = value > 1 + tolerance
    return BilocalWitness(correlator_i, correlator_j, value, 1, bool(violates),
                          "exact_rational_witness" if exact else "numerical_witness",
                          ("independent classical AB and BC sources",
                           "no extra global shared randomness or communication",
                           "external inputs independent of the source variables"))
