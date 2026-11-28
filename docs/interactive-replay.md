# Exact two-round interactive replay

`nonlocalgames.protocols.interactive.replay(protocol, strategy)` evaluates **one
supplied strategy** for a trusted verifier. It returns an exact acceptance
probability and every transcript leaf, including impossible leaves. It does not
optimize, issue a general upper bound, or add a claim to the proof ledger.

Use the [typed event form](protocol-events.md) to check register transfers and local
reads explicitly; it lowers to the same verifier and strategy tables.

The [adaptive Pauli-frame example](adaptive-pauli-frame.md) follows the individual
leaves of a Bell measurement and a later basis question.

```python
from fractions import Fraction
from nonlocalgames.protocols.interactive import replay
from nonlocalgames.protocols.interactive.examples import (
    correlated_memory, delayed_basis, sign_discrimination, split_bell_label,
)

report = replay(*sign_discrimination())
assert report.acceptance_probability == Fraction(9, 10)
assert sum(t.probability for t in report.transcripts) == 1
assert sum(t.accepted_mass for t in report.transcripts) == Fraction(9, 10)
assert replay(*correlated_memory("swapped")).acceptance_probability == 0
assert replay(*delayed_basis(keep_qubit=False)).acceptance_probability == Fraction(25, 26)
assert replay(*split_bell_label()).acceptance_probability == 1
assert replay(*split_bell_label("dephase")).acceptance_probability == Fraction(1, 2)
assert replay(*split_bell_label("reset")).acceptance_probability == Fraction(1, 4)
assert replay(*split_bell_label(independent=True)).acceptance_probability == Fraction(1, 4)
```

## What each participant can use

1. The verifier samples a `PrivateCase` with rational probability. Its table
   index is the private classical label. It prepares a density matrix in
   **R_keep ⊗ Q2 ⊗ Q1** order. It keeps R_keep and Q2, and sends Q1 plus the
   explicitly declared classical `question` (use a constant for no classical
   information).
2. Only that classical question selects a `FirstMove`. Its instrument acts on
   Q1, returns classical answer y and keeps a quantum memory M of the same
   dimension. The prover receives a register, **not** a classical description of
   its state. An outcome can contain multiple weighted Kraus terms or an empty
   tuple for the zero map. Nonnegative rational weights let us express, for
   example, a half-weight channel without representing an irrational square root.
3. The verifier selects `case.after_first[y]` and sends its classical question v
   **and the already prepared wire Q2**. The verifier cannot act on or replace
   Q2 in response to y. Its conditional state can still depend on y because
   Q1 was measured. Dimension one omits this quantum message.
   Exactly `LocalHistory(first_question, y, v)` selects the prover's final POVM
   on **Q2 ⊗ M**. No private case, reference operator, state description, future
   question, or future answer is a strategy dispatch key. Cases with the same
   local view must use the *same* move. Deliberately sending a private bit as
   a classical question is a different verifier.
4. The prover replies z. The verifier's `Continuation.accept[z]` is a binary
   acceptance effect on R_keep. A classical payoff in [0,1] is a scalar effect
   when R_keep is trivial (dimension 1). These tests need not sum to identity over z:
   each occurs **after** the reply z, not instead of it.

Writing ρ for the initial state, a branch of the instrument produces the
unnormalized joint state

    τ_y = Σ_k w_yk (I_R_keep ⊗ I_Q2 ⊗ K_yk) ρ (I_R_keep ⊗ I_Q2 ⊗ K_yk)†.

At transfer, only ownership changes: the cut `(R_keep, Q2) | M` becomes
`R_keep | (Q2, M)` without reordering tensor factors, taking partial traces,
or recreating a product of marginals. For prior π, final POVM effect M_z and
verifier effect E_z, the leaf probability is π tr[(I_R_keep ⊗ M_z) τ_y]; the
accepted mass is π tr[(E_z ⊗ M_z) τ_y].
The returned verifier-side transcript includes the private case index for
analysis only. No probabilities are postselected or divided by a branch trace.

## Four examples and their controls

* **Private sign.** Equiprobable states ρ_s = [[4,2s],[2s,1]]/5, s ∈ {+1,−1},
  give **9/10** with an X measurement. This particular example also has an
  independent upper-bound proof: with Y = diag(3/5,3/10),
  Y − ρ_s/2 = [[1,−s],[−s,1]]/5 is positive semidefinite. Any guess POVM
  therefore succeeds with probability at most tr Y = 9/10. The verifier sends
  no further distinguishing information here. This proof is not a property of
  the general replay result. `reveal=True` explicitly delivers the sign and
  yields 1. `dephase=True` removes the off-diagonal entries before sending Q1;
  the states become identical, and the hidden-sign control scores 1/2.
* **Correlated memory.** Keep one half of Φ⁺ at the verifier, send the other,
  and compare X-readout parity with the first bit y. Conditional half-weight
  I/Z operations (`"matched"`) score 1; exchanging the labels (`"swapped"`)
  scores 0; sampling the bit and I/Z independently (`"independent"`) scores
  1/2. All three have the same fair bit law and the same average dephasing
  channel. Keeping only those two marginals loses the observable distinction.
* **Late basis.** Send a uniformly drawn signed eigenstate for one of two axes
  (±5,0,12)/13; send the basis label only *after* the first move. Retaining
  the qubit scores 1. Measuring Z, resetting memory to |0⟩, and guessing from
  the stored bit scores 25/26. These are scores of the two supplied strategies,
  not a general resource comparison from the evaluator. Giving the basis away
  in round one lets a measure-and-reset strategy score 1. The tests separately
  exercise a verifier that sends a private bit XOR the first answer, so decoding
  needs both the adaptive question and the prover's earlier answer.
* **Split Bell label.** The verifier chooses Φ⁺, Φ⁻, Ψ⁺, or Ψ⁻ uniformly and
  sends one qubit, waits for a fixed acknowledgement, then sends the other.
  Both classical questions are constant. Each qubit alone is I/2 for every
  label. A Bell POVM after retaining Q1 identifies all four labels (score 1).
  Dephasing Q1 in Z retains a classical bit in memory; the same POVM recovers
  parity but not phase (1/2). Resetting Q1 to |0⟩ scores 1/4. Changing the
  verifier to send the product of the same two marginals, I/4, also scores
  1/4 even with full retention.

  For this uniform ensemble only, the 1/2 classical-memory ceiling has an
  independent proof. With only a classical record of Q1 and no entangled side
  memory, a guess z has a separable effect F_z: sum, over records, of the
  conditional Q2 effect tensored with the first measurement's effect.
  A normalized product vector overlaps a Bell state with squared magnitude at
  most 1/2, so tr(F_z ρ_z) ≤ tr(F_z)/2 for Bell projector ρ_z. Summing with
  prior 1/4 and using Σ_z F_z = I₄ gives success ≤ 1/2. Replay itself checks only the examples'
  supplied instruments and POVMs; it does not optimize over classical memory.

  The [noisy-memory comparison](noisy-bell-memory.md) shows two channels with
  identical Bell transcripts that differ under a stored-wire Z readout, then
  two conjugate channels that agree on both readouts but differ on phase parity.

## Checked inputs, budgets, and trust boundary

Use the types exported from `nonlocalgames.protocols.interactive` and
`nonlocalgames.quantum.exact.RationalMatrix` (square matrices over Q+iQ).
`TwoRoundProtocol(reference_dimension, message_dimension, cases, second_message_dimension=1)`
fixes the dimensions of R_keep, Q1 and Q2 across all cases. The default Q2
dimension of 1 preserves the original R ⊗ Q state, memory-only final POVM,
and transcript. This compatibility applies to the Python constructors and
in-memory replay. Pickles are not a supported storage or interchange format
across versions; reconstruct saved protocols using the current constructors.

All input tables are explicit tuples; classical labels are nonnegative 31-bit
Python integers. Probabilities/weights use integers, Fractions, or canonical
rational strings, not floats. `replay` validates table coverage, normalized
priors, density matrices, every final POVM, trace preservation
Σ_yk w_yk K_yk† K_yk = I, and 0 ≤ E_z ≤ I, including on zero-probability
branches. Missing, duplicate, and unused local views are errors.

By default the joint dimension is at most 16, realified PSD checks at most
32, transcript leaves 256, total input matrix-entry occurrences 100,000,
Kraus terms 128, and scalar components 4,096 bits. When changing one limit,
start from `ReplayLimits().exact` and use `dataclasses.replace` to keep the
other replay defaults. A fresh `ExactBudget()` uses a larger generic
`max_events` default (100,000, not replay's 256).

```python
from dataclasses import replace
from nonlocalgames.protocols.interactive import ReplayLimits

defaults = ReplayLimits()
limits = replace(defaults, exact=replace(defaults.exact, max_matrix_entries=24))
assert limits.exact.max_events == 256
assert replay(*sign_discrimination(), limits=limits).acceptance_probability == Fraction(9, 10)
```

Preflight charges all declared leaves (also when their realized probability
is zero), matrix occurrences (also when shared), Kraus terms and a dense work
proxy (default 2,000,000 units) before dense replay allocations. The joint
state dimension is R_keep × Q2 × Q1. Final-effect entries and realified PSD
work use Q2 × Q1, not Q1 alone. PSD checks use cubic realified dimension;
Kraus evolution charges two joint matrix products per term per case/answer;
terminal traces charge quadratic joint work per leaf. Exceeding a budget
raises `ExactBudgetExceeded`, **not** a rejection or an impossibility result.
These are size guards, not process memory or time caps: Python objects already
exist on entry, and intermediate exact arithmetic/Schur elimination can grow.

The checker interprets trusted Python data; it is not an isolation boundary
for arbitrary Python, monkey-patched objects or malicious construction code.
There are no strategy callbacks or untrusted-file decoder. This slice has one
prover, one initial preparation, a classical follow-up that can include a
second prepared wire, and a verifier reference measured at the end. It has no
independent multi-prover dispatch, undeclared persistent quantum memory or
advice, adaptive quantum preparation, intermediate verifier gates,
circuit-depth/light-cone checks, general comb optimizer, or full interactive
proof theorem. It is not a translation into the single-round `Game` type.
Use the distinct [Alice-then-Bob dialect](two-prover-replay.md) when only the
second prover's delivered question may select its POVM and it cannot access
the first prover's output register.
