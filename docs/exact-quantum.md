# Exact finite-dimensional quantum witnesses

The exact quantum fragment checks a concrete state and local POVMs using rational
real and imaginary matrix entries. It supplies an achieved score, hence a lower
bound for the unrestricted finite tensor-product quantum value. It does not claim
that the strategy is globally optimal.

`RationalMatrix(real, imag=None)` stores immutable square matrices over Q+iQ.
Entries are integers, Fractions, or reduced canonical rational strings. Floats,
Boolean entries, decimal/exponent wire notation, and malformed fractions are
rejected. Integer components are normalized to Python's unbounded integers, even
when an input Fraction was built from a fixed-width array scalar.

`ExactQuantumStrategy(state, measurements, dimensions=None)` uses the same
party/input/outcome ordering as numerical strategies. It verifies exact state
Hermiticity, trace one, positivity, effect positivity, and POVM completeness.
Every input has a complete POVM, including zero-probability promised questions.
`probabilities()` and `score(game)` compute the Born rule exactly and reject a
nonzero imaginary probability, an out-of-range probability, or failed normalization.

## Positivity without numerical eigenvalues

For a Hermitian H=A+iB, positivity is equivalent to positivity of the real matrix
`[[A,-B],[B,A]]`. The existing rational Schur checker verifies this matrix exactly,
including singular zero pivots. Realification is only a verification device: it
neither doubles the physical local dimensions nor changes the tensor-product
model. Complex tensor products are formed in the original physical dimensions.

```python
from fractions import Fraction
from nonlocalgames.games import chsh
from nonlocalgames.quantum.exact import exact_chsh_strategy
from nonlocalgames.certificates.quantum import (
    certify_quantum_strategy, verify_quantum_lower_certificate,
)

result = certify_quantum_strategy(chsh(), exact_chsh_strategy())
assert result.value == Fraction(3880899, 4546756)
assert result.value > Fraction(3, 4)
assert result.guarantee == "certified_lower_bound"
assert verify_quantum_lower_certificate(chsh(), result.witness).value == result.value
```

The reference uses a Bell-state density matrix with rational entries and rational
Bloch coefficients `803761/1136689` and `803760/1136689`, whose squared sum is exactly
one. Its score is strictly below the irrational CHSH optimum. `exact_ghz_strategy()`
uses rational complex X/Y effects and attains score one for the three-party GHZ
promise. The existing magic-square reference also admits exact rational data.

`strategy_to_dict/from_dict` use an explicit `rational_complex` schema with no
numerical tolerance. A lower-bound certificate additionally binds the exact game
fingerprint and claimed score. Rechecking decodes the full state/POVM data and
recomputes feasibility and score; it does not trust a result label or numerical
solver. A strategy can be rescored on a matching interface, while a bound
certificate cannot silently change its bound verifier.

## Explicit rationalization of a numerical candidate

```python
from fractions import Fraction
from nonlocalgames.games import chsh
from nonlocalgames.quantum import chsh_strategy
from nonlocalgames.certificates.quantum import certify_quantum_lower_bound

result = certify_quantum_lower_bound(
    chsh(), chsh_strategy(), max_denominator=1_000_000, repair_steps=30)
assert result.value > Fraction(3, 4)
print(result.diagnostics["rationalization"])
```

The rounding procedure Hermitian-rounds real/imaginary entries, corrects state
trace exactly, and corrects each POVM sum exactly. It then mixes the state with
I/D and every effect of a POVM with I/k using a common rational mixing parameter
for that POVM. The endpoint parameter one is feasible. Exact PSD checks guide a
bounded bisection, retaining a verified feasible endpoint. No numerical eigenvalue
establishes positivity, and the final score is recomputed from the modified data.

This procedure preserves physical dimensions and leaves the source arrays
unchanged. It does not automatically preserve a fixed state, projectivity, PPT,
product-state structure, or source independence. A numerical network witness is
not flattened implicitly. Coarse rounding or a zero-step repair can produce a
weak lower bound; that is still a valid point, not a failed optimality theorem.

`AnalysisTask.evaluate(exact_strategy)` supports finite tensor-product POVM
resources with optional dimension upper bounds and either external-randomness
policy. It revalidates the point and reports `exact_witness`, not an optimum.
Fixed-state, PPT, projective, and network contracts require additional exact
checks and are rejected by this evaluator. These restrictions are independent
of whether a particular supplied strategy happens to satisfy them.

## Claims and budgets

`EvidenceKind.QUANTUM_STRATEGY` admits a lower certificate under the standard
`quantum_finite` source model. Its lower bound may propagate upward to closure,
commuting, and nonsignalling values, but never downward to the classical value.
Combined with an exact classical upper bound, it can prove a strict quantum gap.
An exact point at score one is an attained realization; the generic `ValueClaim`
API still asks only about a supremum and does not itself label pseudo-telepathy
or self-testing.

`ExactBudget` bounds physical joint dimension, the realification checker size,
complex matrix entries, event count, parties, and exact scalar bit lengths. Budget
exhaustion propagates distinctly from a failed PSD condition. These limits do not
promise a wall-clock bound for rational elimination. Claim admission has separate
quantum-specific count/bit limits in `VerificationBudget`; `max_quantum_joint_dimension`
bounds the physical joint dimension, while `max_matrix_size` bounds the realified
PSD matrix dimension. These are computation limits, not new resource assumptions.

Rational feasible points can approximate ordinary finite-dimensional strategies
arbitrarily well when precision and repair budgets are allowed to increase. This
supports strict-gap certification in principle, but supplies no uniform dimension,
runtime, or search-success guarantee. Density does not promise an exact rational
witness for every perfect or optimal strategy.

Run `python scripts/quantum_certificate_demo.py` for checked CHSH, GHZ, and
magic-square gap claims. The optional independent command
`julia --startup-file=no --history-file=no scripts/julia/check_exact_references.jl`
checks the three reference constructions with standard-library `Rational{BigInt}`
arithmetic. It is a cross-check, not a Julia optimization adapter or a mandatory
runtime dependency.
