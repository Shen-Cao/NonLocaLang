"""Versioned definition and replay limits for the weighted CHSH grid experiment."""
from dataclasses import asdict
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
import re

from ..claims import VerificationBudget
from ..core.identity import canonical_json
from .provenance import runtime_provenance, validate_provenance
from .weighted_chsh import MAX_GRID_DENOMINATOR, WEIGHTED_CHSH_METHOD

RUN_SCHEMA = "nonlocalgames.weighted_chsh_run.v1"
MAX_STEPS = 10_000


def _fixed_fields():
    return {"schema": RUN_SCHEMA, "family": "nonlocalgames.biased_chsh.v1",
            "grid": "p=k/steps; k=0,...,steps in ascending order",
            "query": {"kind": "gap_greater", "higher": "quantum_finite", "lower": "local",
                      "threshold": "0", "resources": "standard_unrestricted"},
            "methods": {"local": "local_enumeration_replay.v1", "quantum": WEIGHTED_CHSH_METHOD,
                        "upper": "rational_xor_dual.v1"},
            "seed": None}


def _bounded_integer(value, maximum, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return int(value)


def _settings(steps, grid_denominator, screen, solver, eps, verification_budget):
    steps = _bounded_integer(steps, MAX_STEPS, "steps")
    grid_denominator = _bounded_integer(grid_denominator, MAX_GRID_DENOMINATOR, "grid_denominator")
    if not isinstance(screen, bool):
        raise ValueError("screen must be a Boolean")
    if not isinstance(solver, str) or not re.fullmatch("[a-zA-Z][a-zA-Z0-9_]{0,63}", solver):
        raise ValueError("solver must be an explicit solver name")
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise ValueError("eps must be a positive finite real number")
    try:
        eps = float(eps)
    except OverflowError as error:
        raise ValueError("eps is outside the floating-point range") from error
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite real number")
    if not isinstance(verification_budget, VerificationBudget):
        raise TypeError("Expected VerificationBudget")
    solver = solver.upper()
    return {"steps": steps, "grid_denominator": grid_denominator, "screen": screen,
            "solver": solver, "eps": eps, "solver_options": {"eps": eps} if screen and solver == "SCS" else {},
            "screen_method": "xor_vector_sdp.v1" if screen else None,
            "verification_budget": asdict(verification_budget)}


def _with_identity(body):
    digest = sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return {**body, "run_id": RUN_SCHEMA + ":" + digest}


def create_manifest(*, steps=20, grid_denominator=16, screen=True, solver="SCS", eps=1e-8,
                    verification_budget=None):
    """Bind all scientific settings and the executing implementation to one run."""
    budget = VerificationBudget() if verification_budget is None else verification_budget
    settings = _settings(steps, grid_denominator, screen, solver, eps, budget)
    provenance = runtime_provenance(settings["solver"] if screen else None)
    return _with_identity({**_fixed_fields(), **settings, "provenance": provenance})


def validate_manifest(data):
    """Recompute the archived identity without requiring its original environment."""
    encoded = canonical_json(data, max_bytes=64_000)
    fixed = _fixed_fields()
    settings_fields = {"steps", "grid_denominator", "screen", "solver", "eps", "solver_options",
                       "screen_method", "verification_budget", "provenance", "run_id"}
    if not isinstance(data, dict) or set(data) != set(fixed) | settings_fields:
        raise ValueError("Invalid weighted CHSH manifest fields")
    saved = data["verification_budget"]
    if (not isinstance(saved, dict) or set(saved) != set(VerificationBudget.__dataclass_fields__)
            or any(type(v) is not int for v in saved.values())):
        raise ValueError("Manifest must provide every integer verification budget")
    settings = _settings(data["steps"], data["grid_denominator"], data["screen"], data["solver"],
                         data["eps"], VerificationBudget(**saved))
    provenance = data["provenance"]
    validate_provenance(provenance, settings["solver"] if settings["screen"] else None)
    checked = _with_identity({**fixed, **settings, "provenance": provenance})
    if canonical_json(checked) != encoded:
        raise ValueError("Weighted CHSH manifest settings or run identity differ")
    return checked


def replay_budget(manifest, limit=None):
    """Intersect recorded budgets with the local caller's authorized replay limits."""
    limit = VerificationBudget() if limit is None else limit
    if not isinstance(limit, VerificationBudget):
        raise TypeError("Expected VerificationBudget for replay")
    saved = validate_manifest(manifest)["verification_budget"]
    return VerificationBudget(**{name: min(value, getattr(limit, name)) for name, value in saved.items()})
