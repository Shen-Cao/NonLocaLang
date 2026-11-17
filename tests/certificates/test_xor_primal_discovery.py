from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import AnalysisResult
from nonlocalgames.certificates.xor import verify_xor_dual
from nonlocalgames.certificates.xor_primal import verify_xor_primal_certificate
from nonlocalgames.certificates.xor_primal_discovery import certify_xor_lower_bound
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded
from nonlocalgames.games import chsh


def _mock_solver(monkeypatch, matrix):
    import nonlocalgames.analysis.xor as numerical
    result = AnalysisResult(17.0, 'quantum', 'mock', 'numerical_optimum', 'optimal',
                            {'solver': 'mock'}, matrix)
    monkeypatch.setattr(numerical, '_solve_xor', lambda *a, **kw: (result, None))


def test_indefinite_numerical_point_is_repaired_with_a_checked_endpoint(monkeypatch):
    game = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: a[0] == a[1])
    _mock_solver(monkeypatch, [[0.9, 1.3], [1.1, 1.0]])
    result = certify_xor_lower_bound(game, max_denominator=10, repair_steps=0)
    assert result.witness['gram'] == [['1', '1'], ['1', '1']]
    assert result.diagnostics['identity_mixture'] == Fraction(1, 6)
    assert result.diagnostics['numerical_primal_objective'] == 17.0
    assert verify_xor_primal_certificate(game, result.witness).value == result.value == 1


@pytest.mark.parametrize('bad', [np.full((4, 4), np.nan), np.zeros((4, 4), complex),
                                [[1]], [['0']*4]*4, None])
def test_unusable_solver_gram_fails_explicitly(monkeypatch, bad):
    _mock_solver(monkeypatch, bad)
    with pytest.raises(RuntimeError, match='primal Gram'):
        certify_xor_lower_bound(chsh())


def test_scalar_exhaustion_cannot_trigger_psd_repair(monkeypatch):
    import nonlocalgames.certificates.xor_primal_discovery as discovery
    _mock_solver(monkeypatch, np.full((4, 4), 100.0))
    monkeypatch.setattr(discovery, '_repair', lambda *a: pytest.fail('budget must precede repair'))
    with pytest.raises(ScalarBudgetExceeded):
        certify_xor_lower_bound(chsh(), max_scalar_bits=4)


def test_chsh_numerical_discovery_yields_recheckable_narrow_interval():
    game = chsh()
    result = certify_xor_lower_bound(game, solver='SCS', eps=1e-8, max_denominator=10_000)
    lower = verify_xor_primal_certificate(game, result.witness).value
    upper = verify_xor_dual(game, (Fraction(89, 1000),)*4).value
    assert Fraction(85, 100) < lower <= upper < Fraction(86, 100)
