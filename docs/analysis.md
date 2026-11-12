# Analysis contracts and examples

A game specifies a score, not a single intrinsic value. For a resource set R,
its value is the supremum of that score over R. Supremum need not mean attainment.
Use `ResourceSpec` and `AnalysisTask` for local or nonsignalling queries; select
quantum evidence methods explicitly.

```python
from nonlocalgames import AnalysisTask, Resource, ResourceSpec
from nonlocalgames.games import chsh

result = AnalysisTask(chsh(), ResourceSpec(Resource.LOCAL)).solve()
print(result.value, result.witness)  # exact fraction and local response tables
```

`Resource.QUANTUM_FINITE`, `QUANTUM_CLOSURE`, and `QUANTUM_COMMUTING` are distinct
semantic tags. A dimension tuple is allowed only with the finite tensor-product
tag and must match the parties. Quantum `solve()` requires an explicit supported method. `capabilities()` checks
the complete restriction set; a resource tag alone is not a promise that an
appropriate optimizer exists.

## Result interpretation

`AnalysisResult` contains `value`, `resource`, `method`, `guarantee`, `status`,
`diagnostics`, and `witness`.

| Guarantee | Meaning |
|---|---|
| `exact` | Exact local optimization, or verified rational primal/dual optima that coincide |
| `certified_upper_bound` | Verified rational upper bound; a lower witness or interval is included only when separately verified |
| `numerical_optimum` | Floating optimum of the stated mathematical program; numerical error is not certified |
| `numerical_relaxation` | Floating estimate of an outer-relaxation optimum; not a certified upper bound |

In particular, subtracting a local optimum from an NPA value does **not** prove a
quantum advantage: the relaxation may admit nonquantum points. A validated
finite strategy supplies achievable numerical evidence; a symbolic or rigorously
rounded witness supplies proof. XOR is special: its vector SDP is an exact
characterization of the unrestricted bipartite quantum value, but its numerical
solution still needs rigorous error control to certify a gap.

### Exact local optimization

`classical_value(game, max_strategies=1_000_000)` enumerates deterministic response
tables for all but one party and computes that party's best response separately
for each question. Shared randomness cannot improve a linear objective over the
local polytope. The omitted party is chosen to minimize the remaining enumeration.
The witness is a tuple of tables, indexed first by party and then local input.
The guard counts enumerated strategies, not total arithmetic operations.

### Nonsignalling LP

`nonsignalling_value(game, max_variables=100_000)` uses sparse SciPy/HiGHS linear
constraints. For every party, its complementary marginal is independent of that
party's question. Together these imply all subset nonsignalling conditions. The
LP includes all input profiles even when their sampling probabilities vanish.
The witness is a matrix with one row per joint input and one column per joint
output, both in lexicographic order. Primal/dual residuals are diagnostics.

`certify_nonsignalling(game, max_denominator=1_000_000)` reconstructs rational
primal/dual candidates and checks every equality, positivity condition, and dual
inequality exactly. If both witnesses pass, it returns a certified interval; if
the endpoints coincide, `guarantee` is `exact`. Reconstruction can fail, in which
case it raises instead of claiming a bound. This is a separate path from the
uncertified numerical LP; small residuals alone never trigger certification.
`verify_nonsignalling_certificate(game, primal, dual)` independently checks saved
rational witnesses without invoking an optimizer. It accepts rational strings,
integers, or Fractions, rejects floating input, and records constraint schema
`nonlocalgames.nonsignalling.v1` so dual coordinates have explicit meaning.

### NPA and XOR

```python
from nonlocalgames.analysis import npa_bound, xor_quantum_value
from nonlocalgames.games import chsh

outer = npa_bound(chsh(), level="1+AB", solver="SCS", eps=1e-8)
xor = xor_quantum_value(chsh(), solver="SCS", eps=1e-8)
print(outer.status, outer.diagnostics)
```

The NPA generator uses full-outcome projectors and operator-word reductions for
idempotence, orthogonality, completeness, and cross-party commutation. Same-party
operators are never assumed to commute. Level `1` additionally imposes probability
nonnegativity, so its precise convention is Q1 intersect NS. Level `1+AB` includes
cross-party products in the moment basis. Only these levels are implemented.
The real symmetric formulation preserves real Bell objectives by taking real
parts of feasible complex moments; it does not assert that physical measurements
must be real. Mathematical NPA limits describe commuting correlations, which
cannot generally be identified with finite-dimensional tensor-product correlations.

`npa_bound` guards moment-matrix size at 300, and `xor_quantum_value` at 1000 by
default. These are configurable memory safeguards. Their diagnostic matrix
eigenvalues and constraint violations do not substitute for verified dual
certificates. Solver failures raise; `optimal_inaccurate` remains visibly marked.
No computed value is clipped to the physical interval.

The XOR checker requires the reward to depend only on answer parity at every
positive-probability input. Unused questions remain in the Gram model, but their
zero-weight reward entries need not satisfy parity symmetry. Its witness is a Gram matrix, not an extracted
state and collection of POVMs.

### Explicit quantum witnesses and symbolic CHSH

`QuantumStrategy(state, measurements, tolerance=1e-9)` accepts a density matrix and
POVMs ordered `[party][input][outcome]`. Tensor factors follow party order. It
checks Hermiticity, positivity, trace, POVM sums, and dimensions; `score(game)`
evaluates the Born rule. All checks are numerical. The library supplies `chsh_strategy()`, `ghz_strategy()`, and
`magic_square_strategy()` matching their named verifiers. `conjugate()` simultaneously
conjugates state and effects, illustrating a fundamental ambiguity of classical
input/output tests.

`verify_chsh_sos()` symbolically verifies the CHSH sum-of-squares identity,
CHSH square relation, a Bob anticommutator identity, and exact Bell-state attainment.
It applies to the canonical uniform CHSH convention. It is not a certificate for
an arbitrary input game or a proof of a general robustness function. The report
defines the extra equivalence and extraction requirements for self-testing.

### Fixed-dimension see-saw search

```python
from nonlocalgames.games import chsh
from nonlocalgames.quantum import seesaw

candidate = seesaw(chsh(), dimensions=(2, 2), seed=7, restarts=3,
                   max_iterations=30, solver="SCS", solver_options={"eps": 1e-8})
print(candidate.value, candidate.strategy.validate())
```

Each multipartite sweep optimizes local POVMs via SDP and the state via a largest
eigenvector or a constrained state block. Binary projective measurements instead
use analytic spectral updates in the declared local dimensions.
Returned POVMs undergo PSD projection and congruence normalization, followed by
whole-strategy validation and score recomputation. The best actual witness is
retained; the score history is never forced to be monotone. `history` describes
the winning restart, `iterations` locates its best witness, and `restart_scores`
records each restart's best score. Solver statuses remain available. Dimensions
and restart budgets are explicit. A seed controls initialization, but solver and
library changes may alter the result. This is a local heuristic with no guarantee
of global optimality, and a floating strategy score is not a certified lower bound.

## Construction and persistence

```python
from fractions import Fraction
from nonlocalgames.core.transforms import mix, parallel_product, relabel
from nonlocalgames.core.serialization import dump, load
from nonlocalgames.games import chsh

game = chsh()
double = parallel_product(game, game)
mixed = mix([game, game], [Fraction(1, 3), Fraction(2, 3)])
dump(mixed, "output/game.json")
assert load("output/game.json") == mixed
```

The output directory must exist. `relabel` maps new input/output labels to old
ones; each party's output permutation can depend on its new local input.
Serialization uses schema `nonlocalgames.finite.v1`, rational strings, and strict
field checking. JSON does not include executable rules.

## Known benchmark values

| Verifier | Local | Tensor quantum | Nonsignalling |
|---|---:|---:|---:|
| Uniform CHSH | 3/4 | (2+sqrt(2))/4 | 1 |
| Three-party GHZ promise | 3/4 | 1 | 1 |
| Magic square (even rows, odd columns) | 8/9 | 1 | 1 |
| Three-party GYNI, even-parity inputs | 1/4 | 1/4 | 1/3 |

These are literature expectations (sources are in the report), not claims that
all quantum values are proved by the current executable backends. The benchmark
script independently computes local and NS values; CHSH additionally exercises
NPA, XOR, the strategy evaluator, and the symbolic certificate. The optional
magic-square SDP is a larger outer relaxation. GYNI demonstrates that a
nonsignalling gap alone need not imply a quantum gap.

## Restricted quantum tasks

`ResourceSpec` now composes dimension bounds, fixed states, state-family
constraints, external shared randomness, and measurement classes. See
[resource contracts](contracts.md) for semantics. Quantum searches remain
explicit, and their results are `numerical_witness` evidence:

```python
from nonlocalgames import AnalysisTask, ResourceSpec
from nonlocalgames.games import chsh
from nonlocalgames.quantum import FixedState, chsh_strategy

reference = chsh_strategy()
resource = ResourceSpec("quantum_finite", local_dimensions=(2, 2),
                        fixed_state=FixedState(reference.state, (2, 2)))
result = AnalysisTask(chsh(), resource).solve(
    method="seesaw", seed=7, restarts=2, max_iterations=15,
    solver="SCS", solver_options={"eps": 1e-8})
```

`state_constraint="ppt", ppt_cuts=((0,),)` restricts the state across the stated
bipartition. Both see-saw state updates and the separate `state_sdp` method
preserve the restriction. A `state_sdp` query requires supplied `measurements`
and optimizes only the state; its optimum is not an upper bound on a game where
measurements can also vary. For canonical CHSH measurements, the two-qubit PPT
state block gives approximately 0.676777, whereas optimizing both measurements
and PPT states can reach 0.75. Tests distinguish these two claims.

Fixed states never change during search. Dimension bounds are checked against
the actual fixed-state factors. Projective searches support one or two outcomes per party using spectral updates;
more outcomes are rejected. `state_sdp` can accept supplied projective measurements
with any number of outcomes after checking them. Every returned strategy is rechecked against the task contract.
PPT repairs mix with the identity to restore all required positivity constraints;
raw objectives and recomputed witness scores remain separate. A direct low-level
`optimize_state` result exposes both as `.objective` and `.value`.

Run `python scripts/resource_benchmark.py` to compare restrictions using declared
solver settings and initialization seed. The script reports exact local/product
values alongside explicitly numerical quantum witnesses.

## Rechecking exact deterministic witnesses

`AnalysisTask.evaluate(result.witness)` accepts the deterministic response tables
returned by exact local and symbolic product-state optimization. It validates the
input/output labels and computes the score exactly, returning `exact_witness`.
This checks the supplied witness, not optimality; the optimizer's separate exact
claim comes from its exhaustive search. `classical.DeterministicStrategy` provides
a typed version with exact behavior generation, and
`classical.deterministic_quantum_strategy` embeds the tables using one-dimensional
product states and identity/zero projectors. Prescribed numerical fixed states
are not silently replaced by that embedding.


## Certified XOR quantum upper bounds

`certify_xor_upper_bound(game, solver="SCS", eps=1e-8)` discovers a dual diagonal,
reconstructs rational entries, and verifies the dual slack with exact Schur
elimination. When rounding loses positivity, a rational diagonal shift restores
it; diagnostics record the shift and resulting objective increase. The returned
`certified_upper_bound` is a Fraction. The numerical primal objective is kept as
diagnostic evidence and is never used as a certified lower bound.

`verify_xor_dual(game, diagonal)` checks a supplied exact diagonal without solving
an SDP. `verify_xor_certificate(game, result.witness)` independently checks the
serialized fingerprint, diagonal, PSD condition, and claimed bound. Certificate
checking rejects floats and tampered verifier data. The canonical fingerprint
excludes only the descriptive game name.

```sh
python -m nonlocalgames chsh --method xor-certified --witness
python -m nonlocalgames task --file output/unrestricted-xor-task.json --method xor_certified --witness
```

The task method requires an unrestricted bipartite XOR quantum resource contract.
It does not silently drop fixed states, dimensions, measurements, or source
restrictions. A raw unrestricted bound may mathematically apply to subsets, but
that is a separate relaxation claim. Discovery failure remains an error, not an
implicit solver switch. Matrix-size and rational-reconstruction budgets are
configurable through the Python API. Exact rational checking can be expensive
for large matrices or large coefficient bit lengths.
