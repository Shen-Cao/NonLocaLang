# NonLocaLang

Primitives and analysis tools for quantum nonlocal games. The research goal is to
make game construction composable while keeping resource assumptions and the
strength of each analysis result explicit.

The working core covers finite, one-round games with classical inputs and outputs.
The [theoretical report](report/main.tex) develops the broader representation,
including the structures required for networks, sequential games, and quantum
interfaces. These extensions are research designs, not implemented backends.

## Quick start

Use Python 3.11 or newer in an environment of your choice:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nonlocalgames chsh --method local
python -m nonlocalgames chsh --method npa1ab --solver SCS
```

SCS is the portable SDP solver used in examples and tests. Other CVXPY solvers,
including MOSEK when installed and licensed, can be selected explicitly. No local
machine configuration or commercial solver is required.

```python
from fractions import Fraction
from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import classical_value, npa_bound
from nonlocalgames.quantum import chsh_strategy, verify_chsh_sos

game = Game.from_rule(
    Scenario(input_sizes=(2, 2), output_sizes=(2, 2)),
    lambda x, a: (a[0] ^ a[1]) == x[0] * x[1],
    name="CHSH",
)
assert classical_value(game).value == Fraction(3, 4)
print(chsh_strategy().score(game))            # approximately 0.85355339
print(npa_bound(game, solver="SCS").guarantee)  # numerical_relaxation
assert verify_chsh_sos().verified            # exact symbolic identity + attainment
```

For exact input data use integers, `Fraction`, or rational strings. Floating-point
probabilities and payoffs require explicit conversion, so rounding is never hidden.

## Available tools

| Tool | Scope | Evidence |
|---|---|---|
| Rule builder and JSON | Rational finite interfaces, promises, signed scores | Exact representation |
| Relabelling, hidden mixture, parallel product | Explicit local information semantics | Exact construction |
| Local optimizer | Any party count; bounded enumeration | Exact optimum and deterministic witness |
| Nonsignalling LP | Any party count, including promised questions | Numerical solve; optional exact rational primal/dual certification |
| NPA `1`, `1+AB` | Bipartite, arbitrary finite alphabets | Numerical outer-relaxation optimum |
| XOR SDP | Bipartite binary outputs, parity-only reward | Numerical evaluation of an exact characterization |
| Quantum strategy evaluator | Finite tensor-product state and POVMs | Numerically validated witness and score |
| See-saw search | Bipartite, specified local dimensions | Seeded heuristic with validated strategy witnesses |
| CHSH symbolic checker | Canonical CHSH involution algebra | Exact SOS bound and attaining realization |

The benchmark library includes CHSH, GHZ, magic square, and GYNI. No generic
self-testing decision procedure is claimed. An optimum value or an apparently
unique numerical optimizer does not establish self-testing.

## Experiments and report

After installation:

```sh
mkdir -p output
python scripts/benchmark.py --solver SCS > output/benchmarks.json
python scripts/search_xor.py --steps 20 > output/xor-search.json
bash scripts/build_report.sh
```

The report build requires a LaTeX distribution with `latexmk` and BibTeX. Its PDF
is generated at `build/report/main.pdf`. Generated data, PDFs, caches, and downloaded
references are ignored by Git. Public bibliographic identifiers are kept in
[report/bibliography.bib](report/bibliography.bib).

See [architecture and stack choices](docs/architecture.md),
[analysis semantics and API](docs/analysis.md),
[development workflow](docs/development.md), and the
[research roadmap](docs/research.md).

```text
src/nonlocalgames/    core IR, transforms, analysis backends, quantum strategies
scripts/             reproducible experiment and report entry points
tests/               mathematical regressions and interface validation
report/              modular LaTeX article and public bibliography
docs/                architecture, usage, and research development
```
