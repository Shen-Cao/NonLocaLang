from copy import deepcopy
from fractions import Fraction as F
import json

import pytest

from nonlocalgames import Scenario
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded
from nonlocalgames.protocols.compatibility import assess_wiring
from nonlocalgames.protocols.quadratic import Quadratic as Q
from nonlocalgames.protocols.quadratic_behavior import QuadraticBehavior, QuadraticWiringTarget
from nonlocalgames.protocols.quadratic_wire import target_to_dict, target_from_dict, target_id
from nonlocalgames.protocols.serialization import target_id as rational_target_id
from nonlocalgames.protocols.wiring import WiringArena, WiringBudgetExceeded


def _target(irrational=True):
    arena = WiringArena(Scenario((1, 1), (2, 1)), [(1, 0), (0, 1)])
    p = Q(F(1, 2), F(1, 8) if irrational else 0)
    laws = {r: QuadraticBehavior(arena.scenario, probs)
            for r, probs in zip(arena.routes, [(p, 1-p), (1-p, p)])}
    return QuadraticWiringTarget(arena, laws)


def test_family_roundtrip_binds_the_radical_coefficients_and_physical_routes():
    target = _target()
    document = json.loads(json.dumps(target_to_dict(target)))
    document['branches'].reverse()
    document['arena']['routes'].reverse()
    assert target_id(target_from_dict(document)) == target_id(target)
    swapped = QuadraticWiringTarget(target.arena, dict(zip(target.arena.routes, reversed(target.behaviors))))
    assert target_id(swapped) != target_id(target)
    rational = _target(irrational=False)
    assert target_id(rational) != target_id(target)
    assert target_id(rational) != rational_target_id(rational.as_rational())
    with pytest.raises(TypeError):
        assess_wiring(target)
    with pytest.raises(ValueError, match='rational projection'):
        target.as_rational()


def test_wire_binding_and_scope_errors_precede_scalar_decoding(monkeypatch):
    import nonlocalgames.protocols.quadratic_wire as codec
    valid = target_to_dict(_target())
    monkeypatch.setattr(codec, 'scalar_from_pair', lambda *a, **kw: pytest.fail('must reject before decoding entries'))
    for kwargs in ({'max_routes': 1}, {'max_parties': 1}, {'max_total_entries': 3}):
        with pytest.raises(WiringBudgetExceeded):
            target_from_dict(valid, **kwargs)
    wrong = deepcopy(valid)
    wrong['branches'][1]['probabilities'].pop()
    with pytest.raises(ValueError, match='table'):
        target_from_dict(wrong)
    wrong = deepcopy(valid)
    wrong['branches'][0]['route'] = wrong['branches'][1]['route']
    with pytest.raises(ValueError, match='duplicate'):
        target_from_dict(wrong)
    with pytest.raises(ValueError, match='encoding'):
        target_from_dict({**valid, 'encoding': 'a_minus_b_sqrt2'})


def test_budget_overrides_reach_quadratic_pair_decoding():
    valid = target_to_dict(_target())
    for kwargs in ({'max_scalar_chars': 1}, {'max_scalar_bits': 3}):
        with pytest.raises(ScalarBudgetExceeded):
            target_from_dict(valid, **kwargs)
    with pytest.raises(ValueError, match='max_bytes'):
        target_from_dict(valid, max_bytes=10)
