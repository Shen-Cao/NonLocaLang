"""Exact quantum wire data reject implicit numerical or exponent conversions."""
from copy import deepcopy
from fractions import Fraction
import json

import pytest

from nonlocalgames.games import chsh, ghz
from nonlocalgames.quantum.exact import ExactBudget, ExactBudgetExceeded, exact_chsh_strategy, exact_ghz_strategy
from nonlocalgames.quantum.exact.serialization import strategy_from_dict, strategy_to_dict


def test_exact_strategy_roundtrip_retains_complex_data_and_physical_dimensions():
    for strategy, game in ((exact_chsh_strategy(), chsh()), (exact_ghz_strategy(), ghz())):
        payload = json.loads(json.dumps(strategy_to_dict(strategy)))
        restored = strategy_from_dict(payload)
        assert restored == strategy and restored.score(game) == strategy.score(game)
        assert payload['dimensions'] == list(strategy.dimensions)
        assert payload['encoding'] == 'rational_complex'
        assert 'tolerance' not in payload
        assert all(isinstance(v, str) for row in payload['state']['real'] for v in row)


@pytest.mark.parametrize('entry', [0.5, 1, True, '0.5', '2/4', '1e1000000000', '1/0', '-0'])
def test_noncanonical_wire_entries_are_rejected(entry):
    payload = strategy_to_dict(exact_chsh_strategy())
    payload['state']['real'][0][0] = entry
    with pytest.raises((TypeError, ValueError)):
        strategy_from_dict(payload)


def test_schema_dimensions_and_extra_tolerance_cannot_be_reinterpreted():
    original = strategy_to_dict(exact_chsh_strategy(Fraction(2, 5)))
    for change in (
        lambda p: p.update(tolerance=1e-6),
        lambda p: p.update(encoding='complex_float64'),
        lambda p: p.update(dimensions=[1, 4]),
        lambda p: p['measurements'][0][0].pop(),
    ):
        data = deepcopy(original)
        change(data)
        with pytest.raises(ValueError):
            strategy_from_dict(data)


def test_declared_budget_failure_precedes_matrix_decode(monkeypatch):
    import nonlocalgames.quantum.exact.serialization as serialization
    payload = strategy_to_dict(exact_chsh_strategy())
    def forbidden(*args, **kwargs):
        pytest.fail('dimensions/counts must be checked before decoding matrix entries')
    monkeypatch.setattr(serialization, 'matrix_from_dict', forbidden)
    for budget in (ExactBudget(max_joint_dimension=3), ExactBudget(max_psd_size=7),
                   ExactBudget(max_matrix_entries=20), ExactBudget(max_parties=1)):
        with pytest.raises(ExactBudgetExceeded):
            strategy_from_dict(payload, budget=budget)
