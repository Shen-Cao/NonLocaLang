# Certified classical triangle spiral inflation

`triangle_spiral` tests a fixed classical fanout inflation of a no-input triangle.
It can exclude behaviors that pass the triangle Finner inequalities. It does not
solve general triangle membership, and its certificates do not exclude quantum
sources. The construction follows [Wolfe, Spekkens, and Fritz](https://arxiv.org/abs/1609.00672).

## Run a check and recheck its certificate

```python
from fractions import Fraction
from nonlocalgames import Scenario
from nonlocalgames.networks import RationalBehavior, SourceNetwork, triangle_finner
from nonlocalgames.networks.inflation import (
    triangle_spiral, spiral_certificate_to_dict, verify_spiral_certificate,
)

scenario = Scenario((1, 1, 1), (2, 2, 2))
network = SourceNetwork(3, ((0, 1), (1, 2), (2, 0)))
w = RationalBehavior(scenario, tuple(
    Fraction(1, 3) if sum(a) == 1 else 0 for a in scenario.outputs
))
assert triangle_finner(network, w).status == "not_ruled_out"
result = triangle_spiral(network, w)
assert result.status == "incompatible"
assert result.resource == "classical_fanout"
saved = spiral_certificate_to_dict(result.certificate)
checked = verify_spiral_certificate(network, w, saved)
assert checked.diagnostics["farkas_margin"] < 0
```

The verifier rebuilds the marginal problem and checks rational inequalities without
calling an optimizer. W is an incompatibility target here, not a demonstrated
quantum-versus-classical advantage: this example supplies no quantum realization.
Saved certificates bind the full probability table, output
alphabets, ordered source graph, and row schema. Renaming/reordering graph data or
changing the behavior requires a new context binding and a fresh check. Tampered
coefficients, hashes, resource tags, or schema versions are rejected.

## Evidence and fallback policy

| Status | Guarantee | Meaning |
|---|---|---|
| `incompatible` | `exact_classical_incompatibility` | A rational Farkas vector proves that this classical model cannot generate the data |
| `not_ruled_out` | `exact_relaxation_feasibility` | A rational joint distribution satisfies the inflated marginal constraints; original-network membership remains unresolved |
| `not_ruled_out` | `numerical_relaxation_feasibility` | Solver success and numerical residual checks pass, but rational reconstruction did not certify the inflated distribution |
| `unknown` | `uncertified_solver_output` | No usable exact certificate or validated numerical feasibility result was obtained |

An exact feasible inflated distribution is not a realization of the original
triangle. Numerical infeasibility, a time limit, or an inaccurate status alone never
produces exact exclusion. An exact candidate can still be checked after an interrupted
solve: the certificate, not the solver status, establishes the mathematical claim.
Diagnostics retain solver statuses, residuals, reconstruction failures, and any
exact dual repair. Failed nonfinite numerical diagnostics remain serializable.

## What is compiled

For source copies on edges AB, BC, CA, the six responses are ordered
`(A1, B1, C1, A2, B2, C2)`. The five marginal blocks, each ordered by A,B,C outputs,
are:

| Inflated variables | Required marginal |
|---|---|
| A1, B1, C1 | pABC |
| A2, B2, C1 | pAC × pB |
| A1, B2, C2 | pAB × pC |
| A2, B1, C2 | pBC × pA |
| A2, B2, C2 | pA × pB × pC |

The integer marginal matrix A obeys `A q = b(p)`, with `q >= 0`. If the product of
output cardinalities is d, there are d² variables, 5d equalities, and exactly 5d²
nonzero matrix entries. The binary example has 64 variables and 40 equalities.
These are five maximal ancestrally-independent-expressible sets of this particular
spiral; no complete hierarchy or general source-graph compiler is claimed.

The matrix is independent of the input data. A rational vector y with
`A.T @ y >= 0` and `b(p) @ y < 0` gives a Farkas contradiction. Because b contains
products of marginal probabilities, a structurally valid y defines a polynomial
network inequality. It is generally not a linear Bell functional on the original
probability table. A data-bound exported certificate is deliberately narrower than
an automatically reusable symbolic polynomial.

The copied hidden variables admit a common joint classical distribution. Quantum
source systems cannot generally be copied to produce the same joint distribution.
The package therefore labels this backend `classical_fanout` even when a particular
excluded distribution has a separate known quantum impossibility proof.

## Budgets and numerical options

`max_variables`, `max_equalities`, and `max_nonzeros` are checked before inflated
assignments are enumerated. `max_denominator` bounds rational reconstruction;
`validation_tolerance` controls only the numerical fallback residual checks.
The exact verifier has no tolerance.

Discovery uses SciPy/HiGHS. `solver_options` supports `presolve`, `disp`, `maxiter`,
`time_limit`, `primal_feasibility_tolerance`, `dual_feasibility_tolerance`, and
`ipm_optimality_tolerance`. Unsupported keys and values that HiGHS would reset are
rejected. In particular, feasibility tolerances must be at least 1e-10 and the IPM
optimality tolerance at least 1e-12. A zero time/iteration limit is permitted and may
return unresolved evidence. Solver time/iteration limits apply separately to the
primal and dual discovery calls, not to exact rational verification. Entry-count
guards do not bound rational coefficient bit complexity or total wall time. See the [SciPy API](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html)
and [HiGHS option ranges](https://ergo-code.github.io/HiGHS/dev/options/definitions/).

The fixed native fragment needs no optional Inflation installation or commercial
solver. Broader classical inflations and genuine quantum inflation remain separate
extensions. Certificates describe exact supplied probabilities; they do not provide
finite-sample statistical confidence.
