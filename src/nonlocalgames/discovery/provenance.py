"""Identify executing package source and numerical dependencies for discovery runs."""
from hashlib import sha256
from importlib import metadata
import os
from pathlib import Path
import platform
import re
import subprocess

from .. import __version__

_SOURCE_SCHEMA = "nonlocalgames.python_source_tree.v1"
_PACKAGES = ("nonlocalgames", "numpy", "scipy", "sympy", "cvxpy")
_SOLVER_DISTRIBUTIONS = {
    "COPT": "coptpy", "ECOS_BB": "ecos", "GLPK": "cvxopt", "GLPK_MI": "cvxopt",
    "GUROBI": "gurobipy", "HIGHS": "highspy", "PROXQP": "proxsuite", "SCIP": "pyscipopt",
    "SDPA": "sdpa-python",
}


def _source_hash(package):
    """SHA256 of length-prefixed relative POSIX paths and source bytes, in path order."""
    digest = sha256()
    for path in sorted(package.rglob("*.py"), key=lambda p: p.relative_to(package).as_posix()):
        for field in (path.relative_to(package).as_posix().encode("utf-8"), path.read_bytes()):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def _git_revision(package):
    """Return HEAD only when the package initializer exists in that revision."""
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        env.pop(key, None)
    try:
        probe = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD:./__init__.py", "HEAD"],
                               capture_output=True, timeout=5, env=env, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    # The first record resolves the file in HEAD; the second identifies the commit.
    return probe.stdout.decode("ascii").splitlines()[1]


def _package_version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _solver_distribution(solver):
    return _SOLVER_DISTRIBUTIONS.get(solver, solver.lower())


def runtime_provenance(solver):
    """Record source and runtime versions, including the selected solver distribution."""
    package = Path(__file__).resolve().parents[1]
    versions = {name: _package_version(name) for name in _PACKAGES if name != "nonlocalgames"}
    versions["nonlocalgames"] = __version__
    distribution = _solver_distribution(solver) if solver is not None else None
    return {"source_schema": _SOURCE_SCHEMA, "source_sha256": _source_hash(package),
            "git_revision": _git_revision(package), "python": platform.python_version(),
            "implementation": platform.python_implementation(), "packages": versions,
            "solver_distribution": distribution,
            "solver_version": None if distribution is None else _package_version(distribution)}


def validate_provenance(data, solver):
    fields = {"source_schema", "source_sha256", "git_revision", "python", "implementation",
              "packages", "solver_distribution", "solver_version"}
    if not isinstance(data, dict) or set(data) != fields or data["source_schema"] != _SOURCE_SCHEMA:
        raise ValueError("Invalid discovery provenance schema or fields")
    source = data["source_sha256"]
    revision = data["git_revision"]
    if not isinstance(source, str) or not re.fullmatch("[0-9a-f]{64}", source):
        raise ValueError("Invalid package source SHA256 digest")
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(
            "(?:[0-9a-f]{40}|[0-9a-f]{64})", revision)):
        raise ValueError("Invalid Git revision")
    packages = data["packages"]
    if not isinstance(packages, dict) or set(packages) != set(_PACKAGES):
        raise ValueError("Invalid package version fields")
    distribution = _solver_distribution(solver) if solver is not None else None
    if data["solver_distribution"] != distribution:
        raise ValueError("Solver provenance differs from the requested screening method")
    versions = [data["python"], data["implementation"], *packages.values()]
    if distribution is None:
        if data["solver_version"] is not None:
            raise ValueError("Unexpected solver version for disabled screening")
    else:
        versions.append(data["solver_version"])
    if any(not isinstance(v, str) or not v or len(v) > 256 for v in versions):
        raise ValueError("Runtime version strings must be nonempty and bounded")
