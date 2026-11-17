from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as F
from threading import Event

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis.xor import xor_quantum_value
from nonlocalgames.analysis.xor_template import XorScreenSession
from nonlocalgames.games import biased_chsh, chsh


def test_forward_and_reverse_grids_agree_with_fresh_solves_and_detach_witnesses():
    games = [biased_chsh(p) for p in (0, F(1, 10), F(1, 4), 1)]
    expected = [xor_quantum_value(g, solver="SCS", eps=1e-8, warm_start=False).value for g in games]
    a = XorScreenSession(games[0].scenario, solver="SCS", eps=1e-8)
    b = XorScreenSession(games[0].scenario, solver="SCS", eps=1e-8)
    previous = None
    for i in range(len(games)):
        j = len(games)-1-i
        forward, reverse = a.solve(games[i]), b.solve(games[j])
        assert forward.value == pytest.approx(expected[i], abs=1e-7)
        assert reverse.value == pytest.approx(expected[j], abs=1e-7)
        assert forward.guarantee == "numerical_optimum"
        assert forward.diagnostics["certified"] is False
        assert forward.diagnostics["parameterized_template"] and forward.diagnostics["warm_start"] is False
        assert not np.shares_memory(forward.witness, reverse.witness)
        if previous is not None:
            assert not np.shares_memory(previous, forward.witness)
            assert previous[0, 0] == 123
        assert forward.witness[0, 0] == pytest.approx(1, abs=1e-7)
        previous = forward.witness
        previous[0, 0] = 123


def test_signed_weights_and_zero_support_are_recompiled_on_each_solve():
    scenario = Scenario((2, 3), (2, 2))
    def score(x, a):
        if x == (0, 0):
            return a[0]
        return 1 - (x[1]+2)*(1 if a[0] == a[1] else -1)
    game = Game.from_rule(scenario, score, distribution={x: F(1, 5) for x in scenario.inputs if x != (0, 0)})
    session = XorScreenSession(scenario, solver="SCS", eps=1e-8)
    expected = xor_quantum_value(game, solver="SCS", eps=1e-8, warm_start=False)
    assert session.solve(game).value == pytest.approx(expected.value, abs=1e-7)
    with pytest.raises(ValueError, match="parity"):
        session.solve(Game.from_rule(scenario, score))
    with pytest.raises(ValueError, match="scenario"):
        session.solve(Game.from_rule(Scenario((3, 2), (2, 2)), lambda x, a: 1))
    for value in (F(1, 10**400), 10**400):
        with pytest.raises(ValueError, match="float64"):
            session.solve(Game.from_rule(scenario, lambda x, a: value if a[0] == a[1] else 0))
    assert session.solve(game).value == pytest.approx(expected.value, abs=1e-7)


def test_canonicalization_reuse_rebuild_after_failure_and_option_ownership(monkeypatch):
    import nonlocalgames.analysis.xor_template as module
    original = module._solve_problem
    game = chsh()
    nested = [3]
    session = XorScreenSession(game.scenario, solver="SCS", warm_start=False, test_option=nested)
    nested[0] = 5
    programs = []
    def solve(problem, gram, constraints, solver, options):
        assert solver == "SCS" and options["warm_start"] is False and options["enforce_dpp"] is True
        assert problem.is_dpp()
        opts = options.pop("test_option")
        assert opts == [3]
        opts.clear()
        programs.append(problem)
        if len(programs) == 3:
            raise RuntimeError("Interrupted after preceding successful solve")
        return original(problem, gram, constraints, solver, options)
    monkeypatch.setattr(module, "_solve_problem", solve)
    session.solve(game)
    session.solve(game)
    with pytest.raises(RuntimeError, match="Interrupted"):
        session.solve(game)
    last = session.solve(biased_chsh(0))
    assert last.value == pytest.approx(1, abs=1e-5)
    assert programs[0] is programs[1] is programs[2] and programs[3] is not programs[2]


def test_overlapping_requests_refuse_without_invalidating_active_session(monkeypatch):
    import nonlocalgames.analysis.xor_template as module
    original = module._solve_problem
    entered, release = Event(), Event()
    game = chsh()
    session = XorScreenSession(game.scenario, solver="SCS", eps=1e-8)
    def paused(*args):
        entered.set()
        assert release.wait(timeout=10)
        return original(*args)
    monkeypatch.setattr(module, "_solve_problem", paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(session.solve, game)
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(RuntimeError, match="already in use"):
                session.solve(biased_chsh(1))
        finally:
            release.set()
        result = active.result(timeout=10)
    assert result.value == pytest.approx((2 + 2**0.5)/4, abs=1e-7)
    assert session.solve(biased_chsh(1)).value == pytest.approx(1, abs=1e-7)


def test_fixed_session_configuration_and_shape_preflight(monkeypatch):
    import nonlocalgames.analysis.xor_template as module
    monkeypatch.setattr(module.XorScreenSession, "_build", lambda self: pytest.fail("validate first"))
    for solver in (None, "", "  "):
        with pytest.raises(ValueError, match="solver name"):
            XorScreenSession(chsh().scenario, solver=solver)
    for option in ({"warm_start": True}, {"solver_path": ["SCS"]}, {"method": "other"}, {"ignore_dpp": True}):
        with pytest.raises(ValueError):
            XorScreenSession(chsh().scenario, solver="SCS", **option)
    with pytest.raises(ValueError, match="max_matrix_size"):
        XorScreenSession(chsh().scenario, solver="SCS", max_matrix_size=3)
    with pytest.raises(ValueError, match="two parties"):
        XorScreenSession(Scenario((1,), (2,)), solver="SCS")
