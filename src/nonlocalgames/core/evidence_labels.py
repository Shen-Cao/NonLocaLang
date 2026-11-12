"""Runtime evidence vocabulary; labels alone never authenticate a mathematical claim."""
from enum import StrEnum


class Guarantee(StrEnum):
    """Stable JSON strings used to describe the asserted kind of evidence.

    Constructing a record with one of these labels does not check a certificate,
    prove its stated property, or establish that its resource assumptions apply.
    Trusted consumers must invoke the relevant verifier on the bound input data.
    Some labels describe non-value results or internal block diagnostics; merely
    registering a label does not imply that every analysis backend supports it.
    """
    EXACT = "exact"
    EXACT_WITNESS = "exact_witness"
    CERTIFIED_UPPER_BOUND = "certified_upper_bound"
    CERTIFIED_LOWER_BOUND = "certified_lower_bound"
    NUMERICAL_OPTIMUM = "numerical_optimum"
    NUMERICAL_RELAXATION = "numerical_relaxation"
    NUMERICAL_WITNESS = "numerical_witness"
    NUMERICAL_FEASIBLE = "numerical_feasible"
    NUMERICAL_BLOCK_OPTIMUM = "numerical_block_optimum"
    EXACT_NECESSARY_CONDITION = "exact_necessary_condition"
    EXACT_MEMBERSHIP = "exact_membership"
    EXACT_RATIONAL_WITNESS = "exact_rational_witness"
    EXACT_RELAXATION_FEASIBILITY = "exact_relaxation_feasibility"
    EXACT_CLASSICAL_INCOMPATIBILITY = "exact_classical_incompatibility"
    NUMERICAL_RELAXATION_FEASIBILITY = "numerical_relaxation_feasibility"
    UNCERTIFIED_SOLVER_OUTPUT = "uncertified_solver_output"


EXACT_SCORE_GUARANTEES = frozenset({
    Guarantee.EXACT, Guarantee.EXACT_WITNESS,
    Guarantee.CERTIFIED_UPPER_BOUND, Guarantee.CERTIFIED_LOWER_BOUND,
})

VALUE_RESULT_GUARANTEES = EXACT_SCORE_GUARANTEES | frozenset({
    Guarantee.NUMERICAL_OPTIMUM, Guarantee.NUMERICAL_RELAXATION, Guarantee.NUMERICAL_WITNESS,
})
