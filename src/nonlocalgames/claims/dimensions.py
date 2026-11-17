"""Checked value bounds for finite tensor-product POVM models with local dimension caps."""
from ..core.resources import Resource, ResourceSpec
from .types import DimensionInterval

_DETERMINISTIC_CHECKERS = ('deterministic_score.v1', 'local_enumeration_replay.v1')


def dimension_only(resource):
    return (type(resource) is ResourceSpec and resource.kind == Resource.QUANTUM_FINITE
            and resource.local_dimensions is not None
            and resource == ResourceSpec(Resource.QUANTUM_FINITE, resource.local_dimensions))


def _attainment_rule(check, caps):
    if not check.is_proof or check.attained_score is None:
        return None
    if check.resource == Resource.LOCAL and check.checker in _DETERMINISTIC_CHECKERS:
        return 'deterministic_embedding_dimension_one.v1'
    actual = check.actual_dimensions
    if (check.resource == Resource.QUANTUM_FINITE and check.checker == 'rational_quantum_strategy.v1'
            and actual is not None and len(actual) == len(caps)
            and all(d <= cap for d, cap in zip(actual, caps))):
        return 'finite_witness_dimension_fit.v1'
    return None


def dimension_interval(bundle, caps, target_id):
    """Select attained lower scores and the upper bound on the containing unrestricted set."""
    candidates = []
    for check in bundle.checks:
        rule = _attainment_rule(check, caps)
        if rule is not None:
            candidates.append((check.attained_score, check.artifact_id, rule))
    lower = max((score for score, _, _ in candidates), default=None)
    lower_refs = tuple(sorted({ref for score, ref, _ in candidates if score == lower}))
    rules = {rule for score, _, rule in candidates if score == lower}
    unrestricted = next(i for i in bundle.intervals if i.resource == Resource.QUANTUM_FINITE)
    if unrestricted.upper is not None:
        rules.add('unrestricted_upper_to_dimension_cap.v1')
    interval = DimensionInterval(Resource.QUANTUM_FINITE, lower, unrestricted.upper,
                                 lower_refs, unrestricted.upper_refs, target_resource_id=target_id,
                                 local_dimensions=caps)
    return interval, tuple(sorted(rules))
