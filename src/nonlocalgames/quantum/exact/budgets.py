"""Explicit limits for exact matrix parsing, validation, and Born evaluation."""
from dataclasses import dataclass
from numbers import Integral
from operator import index


class ExactBudgetExceeded(ValueError):
    """A declared exact-data or computation-size budget was exceeded."""


@dataclass(frozen=True)
class ExactBudget:
    """Count/size limits; exact Schur intermediates have no wall-clock guarantee.

    Matrix entries count complex scalars, each stored as a rational real/imaginary
    pair. Scalar bits bound supplied numbers and stored algebra outputs; they do
    not assert a bound on every internal exact elimination intermediate.
    """
    max_joint_dimension: int = 16
    max_psd_size: int = 32
    max_matrix_entries: int = 100_000
    max_events: int = 100_000
    max_parties: int = 32
    max_scalar_bits: int = 4096

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
            object.__setattr__(self, name, index(value))


def coerce_budget(budget=None):
    if budget is None:
        return ExactBudget()
    if not isinstance(budget, ExactBudget):
        raise TypeError('Expected ExactBudget')
    return budget


def bounded_dimension(value, budget):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError('Matrix dimensions must be positive integers')
    value = index(value)
    if value > budget.max_joint_dimension:
        raise ExactBudgetExceeded('Dimension exceeds max_joint_dimension')
    return value
