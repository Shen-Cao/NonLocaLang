"""Capability checking for resource-restricted value tasks."""
from dataclasses import replace

from .resources import MeasurementClass, Resource, SharedRandomness, StateConstraint


def capabilities(task):
    r, s = task.resource, task.game.scenario
    if r.kind == Resource.LOCAL:
        return ("local",)
    if r.kind == Resource.NONSIGNALLING:
        return ("ns", "ns_exact")
    if r.kind == Resource.QUANTUM_FINITE and r.state_constraint == StateConstraint.PRODUCT and r.fixed_state is None:
        return ("local",)
    unrestricted = (r.local_dimensions is None and r.fixed_state is None
                    and r.state_constraint == StateConstraint.ANY
                    and r.measurement_class == MeasurementClass.POVM)
    methods = []
    if s.parties == 2 and unrestricted:
        methods.append("npa")
        if s.output_sizes == (2, 2) and all(
            task.game.reward(x, (0, 0)) == task.game.reward(x, (1, 1))
            and task.game.reward(x, (0, 1)) == task.game.reward(x, (1, 0)) for x in s.inputs
        ):
            methods.append("xor")
    if r.kind == Resource.QUANTUM_FINITE:
        if s.parties == 2 and r.measurement_class == MeasurementClass.POVM and (
            r.local_dimensions is not None or r.fixed_state is not None
        ):
            methods.append("seesaw")
        if r.fixed_state is None and r.state_constraint != StateConstraint.PRODUCT:
            methods.append("state_sdp")
    return tuple(methods)


def solve(task, method, options):
    from ..analysis import (AnalysisResult, certify_nonsignalling, classical_value,
                            nonsignalling_value, npa_bound, xor_quantum_value)
    r = task.resource
    if method is None:
        if r.kind == Resource.LOCAL or (r.state_constraint == StateConstraint.PRODUCT and r.fixed_state is None):
            method = "local"
        elif r.kind == Resource.NONSIGNALLING:
            method = "ns"
        else:
            raise NotImplementedError("Select an explicit quantum method from task.capabilities()")
    if method not in capabilities(task):
        raise NotImplementedError(f"Method {method!r} does not enforce this resource contract; "
                                  f"available methods: {capabilities(task)}")
    if method == "local":
        result = classical_value(task.game, **options)
        if r.kind == Resource.LOCAL:
            label = "local_shared_randomness" if r.shared_randomness == SharedRandomness.ALLOWED else "local_product"
        else:
            label = "quantum_product"
        return replace(result, resource=label, diagnostics={**result.diagnostics,
                       "shared_randomness": r.shared_randomness.value,
                       "witness_kind": "deterministic_response_tables",
                       "scope": "linear score; no membership equivalence asserted"})
    if method == "ns":
        return nonsignalling_value(task.game, **options)
    if method == "ns_exact":
        return certify_nonsignalling(task.game, **options)
    if method in ("npa", "xor"):
        result = (npa_bound if method == "npa" else xor_quantum_value)(task.game, **options)
        return replace(result, diagnostics={**result.diagnostics, "requested_resource": r.kind.value})
    from ..quantum.validation import validate_strategy
    if method == "seesaw":
        from ..quantum import seesaw
        if any(key in options for key in ("dimensions", "fixed_state", "ppt_cuts")):
            raise ValueError("State, dimension, and PPT restrictions must come from ResourceSpec")
        dims = r.fixed_state.dimensions if r.fixed_state is not None else r.local_dimensions
        result = seesaw(task.game, dimensions=dims, fixed_state=r.fixed_state,
                        ppt_cuts=r.ppt_cuts, **options)
        residuals = validate_strategy(task, result.strategy)
        return AnalysisResult(result.value, r.kind.value, "restricted_seesaw", "numerical_witness",
                              "heuristic", {**residuals, "history": result.history,
                              "restart_scores": result.restart_scores, "seed": result.seed,
                              "solver": result.solver, "solver_statuses": result.solver_statuses,
                              "shared_randomness": r.shared_randomness.value}, result.strategy)
    from ..quantum import optimize_state
    from ..quantum.validation import validate_measurements
    if "ppt_cuts" in options:
        raise ValueError("PPT restrictions must come from ResourceSpec")
    options = dict(options)
    if "measurements" not in options:
        raise ValueError("state_sdp requires supplied measurements")
    options["measurements"] = validate_measurements(
        task, options["measurements"], tolerance=options.get("validation_tolerance", 1e-7),
        max_joint_dimension=options.get("max_joint_dimension", 256))
    result = optimize_state(task.game, ppt_cuts=r.ppt_cuts, **options)
    residuals = validate_strategy(task, result.strategy)
    return AnalysisResult(result.value, r.kind.value, "fixed_measurements_state_sdp",
                          "numerical_witness", result.status,
                          {**result.diagnostics, **residuals, "state_block_objective": result.objective,
                           "scope": "fixed supplied measurements, not the global game optimum"}, result.strategy)
