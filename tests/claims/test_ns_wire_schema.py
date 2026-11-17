"""NS proof admission requires the portable vector encoding on both LP sides."""
import pytest

from nonlocalgames import Game, ResourceSpec, Scenario
from nonlocalgames.claims import ValueClaim, assess_value, make_evidence


@pytest.mark.parametrize('field, malformed', [
    ('primal', '1'), ('primal', {'1': None}), ('primal', [1]),
    ('dual', '0'), ('dual', {'0': False}), ('dual', [0]),
])
def test_malformed_but_iterable_vectors_cannot_prove_an_ns_bound(field, malformed):
    game = Game(Scenario((1,), (1,)), (1,), (0,))
    resource = ResourceSpec('nonsignalling')
    payload = {'primal': ['1'], 'dual': ['0'], field: malformed}
    artifact = make_evidence(game, resource, 'nonsignalling_lp', payload)
    result = assess_value(game, ValueClaim(resource, 'at_most', 0), [artifact])
    assert result.truth == 'unknown' and result.execution == 'invalid_evidence'
    assert field in result.issues[0].message
