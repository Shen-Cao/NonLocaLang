"""Exact shared-randomness local values by deterministic best responses."""

from fractions import Fraction
from itertools import product
from math import prod

from .result import AnalysisResult


def classical_value(game, *, max_strategies: int = 1_000_000) -> AnalysisResult:
    """Optimize exactly over local responses with unrestricted shared randomness.

    Enumerate all parties except one; that party's response is optimized
    independently at each question. The guard counts enumerated joint response
    tables, not the larger full deterministic strategy set. Complexity remains
    exponential. Source independence constraints are not represented here.
    """
    if isinstance(max_strategies, bool) or not isinstance(max_strategies, int) or max_strategies < 1:
        raise ValueError("max_strategies must be a positive integer")
    scenario = game.scenario
    counts = tuple(a**x for x, a in zip(scenario.input_sizes, scenario.output_sizes))
    responder = max(range(len(counts)), key=counts.__getitem__)
    enumerated = prod(counts) // counts[responder]
    if enumerated > max_strategies:
        raise ValueError(f"exact local search needs {enumerated} strategies; limit is {max_strategies}")
    others = tuple(i for i in range(len(counts)) if i != responder)
    spaces = [product(range(scenario.output_sizes[i]), repeat=scenario.input_sizes[i]) for i in others]
    best_value = None
    best_strategy = None
    for responses in product(*spaces):
        tables = dict(zip(others, responses))
        scores = [[Fraction(0) for _ in range(scenario.output_sizes[responder])]
                  for _ in range(scenario.input_sizes[responder])]
        for x in scenario.inputs:
            a = [tables[i][x[i]] if i != responder else 0 for i in range(len(counts))]
            for output in range(scenario.output_sizes[responder]):
                a[responder] = output
                scores[x[responder]][output] += game.coefficient(x, tuple(a))
        reply = tuple(max(range(len(row)), key=row.__getitem__) for row in scores)
        value = sum((row[a] for row, a in zip(scores, reply)), Fraction(0))
        if best_value is None or value > best_value:
            tables[responder] = reply
            best_value = value
            best_strategy = tuple(tables[i] for i in range(len(counts)))
    return AnalysisResult(
        value=best_value, resource="local_shared_randomness", method="exact_best_response",
        guarantee="exact", status="optimal",
        diagnostics={"enumerated_strategies": enumerated, "full_strategy_count": prod(counts),
                     "best_response_party": responder}, witness=best_strategy,
    )
