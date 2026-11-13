"""Portable game readers bound semantic scalars before exact arithmetic."""
from fractions import Fraction
import json

import pytest

from nonlocalgames import AnalysisTask, Game, ResourceSpec, Scenario
from nonlocalgames.core.identity import verifier_id
from nonlocalgames.core.resource_serialization import task_from_dict, task_to_dict
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded
from nonlocalgames.core.serialization import from_dict, load, to_dict
from nonlocalgames.games import chsh


@pytest.mark.parametrize('field', ['distribution', 'payoff'])
@pytest.mark.parametrize('text', ['1e1000000000', '1e-1000000000', '9'*1500])
def test_game_rejects_compact_expansion_and_oversized_components(field, text):
    data = to_dict(chsh())
    data[field][0] = text
    with pytest.raises(ScalarBudgetExceeded):
        from_dict(data)


def test_legacy_scalar_spellings_keep_exact_content_identity():
    game = chsh()
    data = to_dict(game)
    data['distribution'] = ['+0.25', '25e-2', '1 / 4', '2_5e-2']
    data['payoff'] = ['1.0' if value else '-0e3' for value in game.payoff]
    restored = from_dict(data)
    assert restored == game
    assert verifier_id(restored) == verifier_id(game)
    assert to_dict(restored) == to_dict(game)


def test_game_checks_lengths_and_expansion_budget_before_scalar_parse(monkeypatch):
    import nonlocalgames.core.serialization as codec
    monkeypatch.setattr(codec, 'parse_rational', lambda *a, **k: pytest.fail('parse before preflight'))
    with pytest.raises(ValueError, match='max_entries'):
        from_dict(to_dict(chsh()), max_entries=15)
    data = to_dict(chsh())
    data['payoff'].pop()
    with pytest.raises(ValueError, match='table sizes'):
        from_dict(data)


def test_explicit_large_budget_roundtrips_without_global_digit_setting():
    value = Fraction(10**5000 + 1, 3)
    game = Game(Scenario((1,), (1,)), (1,), (value,))
    data = to_dict(game)
    with pytest.raises(ScalarBudgetExceeded):
        from_dict(data)
    restored = from_dict(data, max_scalar_bits=17000, max_scalar_chars=11000)
    assert restored == game
    assert verifier_id(restored) == verifier_id(game)


def test_file_and_task_readers_propagate_budgets(tmp_path):
    data = to_dict(chsh())
    path = tmp_path / 'game.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='max_file_bytes'):
        load(path, max_file_bytes=10)
    with pytest.raises(ScalarBudgetExceeded, match='max_bits'):
        load(path, max_scalar_bits=2)
    task = task_to_dict(AnalysisTask(chsh(), ResourceSpec('local')))
    with pytest.raises(ScalarBudgetExceeded):
        task_from_dict(task, max_scalar_bits=2)
    with pytest.raises(ValueError, match='max_entries'):
        task_from_dict(task, max_game_entries=15)
    assert task_from_dict(task).game == chsh()


def test_zero_denominator_is_a_value_error():
    data = to_dict(chsh())
    data['payoff'][0] = '1/0'
    with pytest.raises(ValueError, match='denominator'):
        from_dict(data)
