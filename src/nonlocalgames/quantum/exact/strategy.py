"""Exactly feasible finite tensor-product quantum witnesses."""
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce

from ...core.game import Scenario
from .algebra import kron, trace_product
from .budgets import ExactBudgetExceeded, coerce_budget
from .matrix import RationalMatrix
from .scalars import exact_scalar
from .validation import (collect_measurements, declared_dimensions, validate_game_scalars,
                         validate_scenario, validate_strategy_data)


@dataclass(frozen=True, init=False)
class ExactQuantumStrategy:
    """Rational state and POVMs, in the original declared physical dimensions.

    These checks establish ordinary finite tensor-product feasibility. They do
    not implicitly establish fixed-state, projective, PPT, product, or network
    promises. Every input has a full POVM, including unused promised questions.
    """
    state: RationalMatrix
    measurements: tuple[tuple[tuple[RationalMatrix, ...], ...], ...]
    dimensions: tuple[int, ...]
    _budget: object = field(repr=False, compare=False)

    def __init__(self, state, measurements, dimensions=None, *, budget=None):
        budget = coerce_budget(budget)
        if not isinstance(state, RationalMatrix):
            raise TypeError('Exact state must be a RationalMatrix')
        state.check_budget(budget)
        measurements, actual, _, total = collect_measurements(measurements, state, budget)
        actual = declared_dimensions(dimensions, actual, budget)
        if state.dimension != total:
            raise ValueError('State dimension differs from the local tensor factors')
        validate_strategy_data(state, measurements, actual, budget)
        object.__setattr__(self, 'state', state)
        object.__setattr__(self, 'measurements', measurements)
        object.__setattr__(self, 'dimensions', actual)
        object.__setattr__(self, '_budget', budget)

    @property
    def scenario(self):
        return Scenario(tuple(len(party) for party in self.measurements),
                        tuple(len(party[0]) for party in self.measurements))

    def validate(self, *, budget=None):
        budget = self._budget if budget is None else coerce_budget(budget)
        self.state.check_budget(budget)
        measurements, actual, entries, total = collect_measurements(self.measurements, self.state, budget)
        if total != self.state.dimension or actual != self.dimensions:
            raise ValueError('Strategy tensor dimensions are inconsistent')
        diagnostics = validate_strategy_data(self.state, measurements, actual, budget)
        return {**diagnostics, 'dimensions': actual, 'matrix_entries': entries}

    def _evaluation_budget(self, scenario, budget):
        budget = self._budget if budget is None else coerce_budget(budget)
        validate_scenario(scenario, budget)
        if scenario != self.scenario:
            raise ValueError('Exact strategy and game interfaces differ')
        self.state.check_budget(budget)
        if 2*self.state.dimension > budget.max_psd_size:
            raise ExactBudgetExceeded('State realification exceeds max_psd_size')
        collect_measurements(self.measurements, self.state, budget)
        return budget

    def probabilities(self, scenario=None, *, budget=None):
        """Exact full conditional probabilities; a nonzero imaginary value is rejected."""
        scenario = self.scenario if scenario is None else scenario
        budget = self._evaluation_budget(scenario, budget)
        probabilities = []
        for x in scenario.inputs:
            row = []
            for a in scenario.outputs:
                factors = (self.measurements[i][x[i]][a[i]] for i in range(scenario.parties))
                joint = reduce(lambda left, right: kron(left, right, budget=budget), factors)
                real, imag = trace_product(self.state, joint, budget=budget)
                if imag != 0:
                    raise ValueError('Exact Born probability has a nonzero imaginary part')
                if real < 0 or real > 1:
                    raise ValueError('Exact Born probability lies outside [0,1]')
                row.append(real)
            if sum(row, Fraction(0)) != 1:
                raise ValueError('Exact Born probabilities do not sum to one')
            probabilities.extend(row)
        return tuple(probabilities)

    def score(self, game, *, budget=None):
        """Return an exact achieved score, not a claim of global optimality."""
        budget = self._budget if budget is None else coerce_budget(budget)
        coefficients = validate_game_scalars(game, budget)
        score = Fraction(0)
        for coefficient, probability in zip(coefficients, self.probabilities(game.scenario, budget=budget)):
            score = exact_scalar(score+coefficient*probability, max_bits=budget.max_scalar_bits)
        return exact_scalar(score, max_bits=budget.max_scalar_bits)
