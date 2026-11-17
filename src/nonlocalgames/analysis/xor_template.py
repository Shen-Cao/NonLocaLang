"""Sequential reuse of one bipartite XOR vector SDP for numerical screening."""
from copy import deepcopy
from dataclasses import replace
from threading import Lock

from ..certificates.linalg import positive_integer
from ..core.game import Game, Scenario
from .xor import _gram_problem, _numeric_coefficients, _solve_problem


class XorScreenSession:
    """A fixed-interface parameterized XOR SDP with explicit solver selection.

    Each solve compiles the verifier coefficients and checks XOR applicability.
    CVXPY reuses the parameterized canonicalization across successful solves.
    A session handles one request at a time and each solve disables warm starts.
    """

    def __init__(self, scenario: Scenario, *, solver: str, max_matrix_size=1000, **solver_options):
        if not isinstance(scenario, Scenario):
            raise TypeError("Expected a Scenario")
        if scenario.parties != 2 or scenario.output_sizes != (2, 2):
            raise ValueError("XOR SDP requires two parties with two outputs each")
        limit = positive_integer(max_matrix_size, "max_matrix_size")
        if sum(scenario.input_sizes) > limit:
            raise ValueError(f"XOR Gram matrix exceeds limit {limit} (max_matrix_size)")
        if not isinstance(solver, str) or not solver.strip():
            raise ValueError("XOR screening requires an explicit solver name")
        if solver_options.get("warm_start", False) is not False:
            raise ValueError("XOR screen sessions require warm_start=False")
        if solver_options.keys() & {"method", "solver_path", "ignore_dpp", "enforce_dpp", "gp",
                                    "qcp", "requires_grad"}:
            raise ValueError("XOR screen sessions require one solver and fixed DPP compilation")

        self._scenario = scenario
        self._limit = limit
        self._solver = solver
        self._options = deepcopy(dict(solver_options, warm_start=False, enforce_dpp=True))
        self._lock = Lock()
        self._template = self._build()

    def _build(self):
        import cvxpy as cp
        offset = cp.Parameter()
        weights = cp.Parameter(self._scenario.input_sizes)
        problem, gram, constraints = _gram_problem(self._scenario.input_sizes, offset, weights)
        if not problem.is_dpp():
            raise RuntimeError("XOR screen template failed the DPP check")
        return offset, weights, problem, gram, constraints

    def solve(self, game: Game):
        """Solve with fresh coefficients and return a detached numerical result.

        If parameter assignment or the solver fails, discard the entire CVXPY
        problem. The next call builds a new problem with the original settings.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("XOR screen session is already in use")
        try:
            if not isinstance(game, Game):
                raise TypeError("Expected a finite Game")
            if game.scenario != self._scenario:
                raise ValueError("Verifier scenario differs from the XOR screen session")
            constant, coefficients = _numeric_coefficients(game, self._limit)
            if self._template is None:
                self._template = self._build()
            offset, weights, problem, gram, constraints = self._template
            try:
                offset.value = constant
                weights.value = coefficients
                result = _solve_problem(problem, gram, constraints, self._solver, deepcopy(self._options))
                return replace(result, witness=result.witness.copy(),
                               diagnostics={**result.diagnostics, "parameterized_template": True,
                                            "warm_start": False})
            except BaseException:
                self._template = None
                raise
        finally:
            self._lock.release()
