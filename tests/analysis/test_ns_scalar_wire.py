from fractions import Fraction

import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import verify_nonsignalling_certificate
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded


@pytest.mark.parametrize('primal, dual', [(['1e1000000000'], ['0']),
                                        (['1'], ['-1e1000000000'])])
def test_ns_scalars_rejected_before_sparse_compilation(monkeypatch, primal, dual):
    import nonlocalgames.analysis.certificates as certificates
    monkeypatch.setattr(certificates, '_nonsignalling_program',
                        lambda *a, **k: pytest.fail('must bound scalars first'))
    game = Game.from_rule(Scenario((1,), (1,)), lambda x, a: 0)
    with pytest.raises(ScalarBudgetExceeded):
        verify_nonsignalling_certificate(game, primal, dual)


def test_ns_certificate_can_use_explicitly_raised_scalar_budgets():
    value = Fraction(10**1300)
    game = Game.from_rule(Scenario((1,), (1,)), lambda x, a: value)
    with pytest.raises(ScalarBudgetExceeded):
        verify_nonsignalling_certificate(game, ['1'], ['-1e1300'])
    result = verify_nonsignalling_certificate(game, ['1'], ['-1e1300'], max_scalar_bits=4400)
    assert result.value == value and result.guarantee == 'exact'
