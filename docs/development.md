# Development and reproducibility

Install the package in an isolated Python environment with `python -m pip install
-e '.[dev]'`. Dependencies and minimum supported versions are declared in
`pyproject.toml`; no environment name, absolute installation path, hardware target,
or solver license is required. Experiment output records the Python and numerical
package versions. For an archived experiment, save an environment lock alongside
its generated data; dependency ranges alone do not promise bitwise reproducibility.

Run the relevant mathematical tests after modifying a backend, then the full suite
and lint before committing:

```sh
python -m pytest -q
python -m ruff check src tests scripts
bash scripts/build_report.sh
```

CI tests Python 3.11 and 3.12 using portable dependencies. Commercial solvers are
optional cross-checks. Do not infer a verified certificate from a solver status,
small residual, or agreement between solvers. Tests use tolerances only for
numerical claims; exact fractions and algebraic identities compare exactly.

Regression tests are grouped under `tests/core`, `tests/analysis`,
`tests/certificates`, `tests/classical`, `tests/quantum`, `tests/networks`, and
`tests/cli`, `tests/claims`, `tests/properties`, and `tests/equivalence`; pytest
discovers these recursively. Importlib test loading keeps
module names isolated across these directories; shared fixtures belong in
`conftest.py` rather than cross-imported test modules.

Keep mathematical modules focused: verifier data, resource semantics, transforms,
analysis, and strategy witnesses have separate modules. New dialects need explicit
capability checks and a counterexample test for any tempting but invalid lowering.
Unsupported resource combinations should fail visibly. New stochastic searches
must accept a seed, retain feasible witnesses, and state their termination rules.

Generated files belong in `build/` or `output/`. Downloaded papers may be cached
in `references/`, which is ignored; tracked documents must cite a DOI, arXiv ID,
or public URL. Bibliographic metadata belongs in `report/bibliography.bib`.
Do not commit rendered PDFs, solver logs, raw experiment output, caches, virtual
environments, credentials, or local editor settings.

Before staging, inspect `git status --short`, `git diff --check`, and `.gitignore`.
Stage named paths for one coherent change. Use conventional commit subjects such
as `feat(core): ...`, `feat(analysis): ...`, `test: ...`, and `docs(research): ...`.
Put editorial history and implementation decisions in commit messages; keep source
comments and research prose about semantics, reasoning, and reproducible use.

## Comparing replay cost

Run `python scripts/benchmark_replay.py --repeats 4` with each installed revision
in the same environment. The script warms the workload, times 21 weighted CHSH
members without a numerical screen, replays their raw evidence, and checks a
fixed sample of eight members per XOR3 orbit. It prints separate timing samples
and hashes of the checked artifact IDs, assessments, and member records.
Compare those hashes before interpreting timings. Runtime measurements depend
on the machine and are excluded from correctness assertions.

For the numerical screen, run
`python scripts/benchmark_xor_screen.py --solver SCS --accuracy 1e-8 --repeats 2`.
Each balanced block runs the 21-point grid in both method orders. Both paths
disable warm starts. The JSON report includes the one-time session construction,
first call, per-point compilation and solver times, and maximum objective
difference. Compare this screening cost separately from exact verification.
