"""Recheck task witnesses while preserving resource and evidence distinctions."""
from ..analysis import AnalysisResult
from ..classical import DeterministicStrategy
from .resources import Resource, SharedRandomness, StateConstraint


def evaluate(task, strategy):
    """Evaluate a valid local table, finite quantum strategy, or network witness.

    Local optimizers keep their historical response-table representation.
    Accepting those tables here provides an exact witness check; it never
    promotes a supplied witness score to an optimum claim.
    """
    r = task.resource
    if r.network is not None:
        from ..networks.tasks import evaluate as evaluate_network
        return evaluate_network(task, strategy)
    from ..quantum.strategy import QuantumStrategy
    if isinstance(strategy, QuantumStrategy):
        from ..quantum.validation import evaluate as evaluate_quantum
        return evaluate_quantum(task, strategy)
    from ..quantum.exact import ExactQuantumStrategy
    if isinstance(strategy, ExactQuantumStrategy):
        from ..quantum.exact.evaluation import evaluate as evaluate_exact
        return evaluate_exact(task, strategy)
    local = r.kind == Resource.LOCAL
    product = (r.kind == Resource.QUANTUM_FINITE and r.state_constraint == StateConstraint.PRODUCT
               and r.fixed_state is None)
    if not local and not product:
        raise NotImplementedError("Deterministic table evaluation requires local or unrestricted product-state resources")
    if isinstance(strategy, DeterministicStrategy):
        if strategy.scenario != task.game.scenario:
            raise ValueError("Witness and task scenarios differ")
        witness = strategy
    else:
        witness = DeterministicStrategy(task.game.scenario, strategy)
    resource = "quantum_product" if product else (
        "local_shared_randomness" if r.shared_randomness == SharedRandomness.ALLOWED else "local_product")
    return AnalysisResult(witness.score(task.game), resource, "deterministic_witness_evaluation",
                          "exact_witness", "evaluated",
                          {"certified": True, "witness_kind": "deterministic_response_tables",
                           "shared_randomness": r.shared_randomness.value,
                           "scope": "supplied witness score; optimality is not asserted"}, witness.responses)
