"""Recompute one named balanced routing strategy and compare a supplied target."""
from hashlib import sha256

from ..core.identity import canonical_json
from .fermion_reference import REFERENCE_STATISTICS, reference_arena, reference_target
from .quadratic import scalar_to_pair
from .quadratic_behavior import QuadraticWiringTarget
from .quadratic_wire import target_id
from .serialization import arena_id


def reference_definition(statistics):
    if statistics not in REFERENCE_STATISTICS:
        raise ValueError(f"statistics must be one of {REFERENCE_STATISTICS}")
    definition = {"schema": "nonlocalgames.six_mode_reference.v1",
                  "arena_id": arena_id(reference_arena()), "statistics": statistics,
                  "preparation": "one_balanced_particle_per_origin",
                  "port_basis": "incoming_then_kept", "party_order": ["A", "B", "C"],
                  "outputs": ["0", "1+", "1-", "2"], "measurements": "B.6_B.17_B.18",
                  "source": "arXiv:2606.12363v1"}
    if statistics == "fermionic_even_pair":
        definition.update(schema="nonlocalgames.even_pair_reference.v1",
                          preparation="one_balanced_fermion_pair_per_origin",
                          encoding="0_to_00_and_1_to_11_per_port", modes_per_port=2,
                          encoding_source="arXiv:2611.25074v1")
    return definition


def compare_reference(target, *, statistics="fermionic"):
    """Check equality with the specified implementation at every route and question."""
    if not isinstance(target, QuadraticWiringTarget):
        raise TypeError("Expected a quadratic wiring target")
    definition = reference_definition(statistics)
    if target.arena != reference_arena():
        raise ValueError("The reference requires its specified five routes and question/answer alphabets")
    expected = reference_target(statistics=statistics)
    mismatches = 0
    first = None
    entries = 0
    scenario = expected.arena.scenario
    outputs = scenario.outputs
    inputs = scenario.inputs
    for route, actual_branch, expected_branch in zip(target.arena.routes, target.behaviors, expected.behaviors):
        for i, (actual, predicted) in enumerate(zip(actual_branch.probabilities, expected_branch.probabilities)):
            entries += 1
            if actual != predicted:
                mismatches += 1
                if first is None:
                    x, a = divmod(i, len(outputs))
                    first = {"route": list(route), "questions": list(inputs[x]), "answers": list(outputs[a]),
                             "expected": scalar_to_pair(predicted), "actual": scalar_to_pair(actual)}
    target_key, expected_key = target_id(target), target_id(expected)
    if entries != 2560 or (mismatches == 0) != (target_key == expected_key):
        raise RuntimeError("Reference comparison disagrees with target coverage or identity")
    identifier = sha256(canonical_json(definition).encode("utf-8")).hexdigest()
    return {"schema": "nonlocalgames.reference_comparison.v1", "reference": definition,
            "reference_id": definition["schema"] + ":" + identifier, "target_id": target_key,
            "expected_target_id": expected_key, "matches_reference": mismatches == 0,
            "checked_entries": entries, "mismatches": mismatches, "first_mismatch": first}
