# Alice-then-Bob exact replay

Use `replay_two_prover(protocol, strategy)` to score **one supplied pair of
local tables**. Alice and Bob each act once, in that order. The verifier can
send information from Alice to Bob by its choice of question; the strategy
cannot add that channel on its own. This differs from [one-prover two-round
replay](interactive-replay.md), where the same prover knows its first answer
and retains its first quantum register.

```python
from fractions import Fraction
from nonlocalgames.protocols.interactive import replay_two_prover
from nonlocalgames.protocols.interactive.examples import two_prover_bit_relay

hidden, strategy = two_prover_bit_relay()
forwarded, unchanged = two_prover_bit_relay(forward_answer=True)
assert strategy == unchanged
assert replay_two_prover(hidden, strategy).acceptance_probability == Fraction(1, 2)
report = replay_two_prover(forwarded, strategy)
assert report.acceptance_probability == 1
assert len(report.transcripts) == 16
```

The verifier in this example draws independent fair bits x and c. Alice receives
x and replies y=x. Bob receives v and replies z=v; the verifier scores z=x.
In the hidden version v=c: since all quantum wires are trivial and Bob's coin
is independent of x, no Bob row can distinguish the two x values. The forwarded
version changes only the verifier's continuation to v=y. Bob now always guesses
correctly, even though his own question is still marginally fair. This is
intentional communication through the verifier, not a violation of a
no-signalling theorem.

## Wires, dispatch, and the checked quantity

Pass `TwoProverProtocol(reference_dimension=..., alice_dimension=...,
bob_dimension=..., cases=...)` and `TwoProverStrategy(alice, bob)` from
`nonlocalgames.protocols.interactive`. All tables are explicit tuples with
the same bounded exact scalar, matrix, and label rules as the one-prover API.

1. A private `PrivateCase` with prior π prepares density matrix ρ in **R_keep
   ⊗ Q_B ⊗ Q_A** order. The verifier delivers only Q_A and `case.question` (x)
   to Alice. One `FirstMove` per x selects a weighted Kraus instrument on Q_A;
   its answer is y. Alice never transfers her residual register to Bob.
2. The verifier selects `case.after_first[y]`, a `Continuation`, and delivers
   its `question` (v) and the **already prepared Q_B** to Bob. A single
   `BobMove(v, effects)` supplies a POVM on Q_B for *every* occurrence of v.
   Neither Alice's x nor her y nor the private case is a Bob dispatch key.
   Correlations present in ρ can nevertheless steer the state conditional on y.
3. Bob replies z. The verifier uses `Continuation.accept[z]`, a binary
   acceptance effect E on R_keep. The full result labels each leaf
   `(private_case, alice_question, alice_answer, bob_question, bob_answer)`;
   this analyst transcript is not the local input of either prover. The
   probabilities include priors and zeros, with no postselection.

Because Alice's residual system is never used, for outcome y its entire
instrument branch enters through the effect A[x,y] = Σₖ wₖ Kₖ†Kₖ. With Bob's
effect B[v,z], the leaf probability and accepted mass are exactly

    π tr[(I_R ⊗ B[v,z] ⊗ A[x,y]) ρ]
    π tr[(E   ⊗ B[v,z] ⊗ A[x,y]) ρ].

The sum of the second quantity over **all** leaves is the reported score.
This contraction preserves steering, rather than replacing ρ by a product of
marginals. For example, let Q_B,Q_A be in (3|00⟩ + 4|11⟩)/5, with one
constant v. Z readouts always agree, with probabilities 9/25 and 16/25; the
product of the *same* individual marginals agrees only 337/625 of the time.
Keeping or resetting Alice's post-measurement system does not affect either
probability law.

## Admission and scope

Missing, duplicate and unused local rows are errors. All verifier continuations
using the same v must agree with its Bob row's answer alphabet, even for
zero-prior cases or impossible first outcomes. Preflight checks shapes and
input scalar limits, event/entry totals, `max_parties >= 2`, Kraus-term count,
and the dense-work budget **before** dense replay allocations. Physics checks
then verify densities, completeness of the instruments/POVMs, and 0 ≤ E ≤ I.
A `ReplayLimits` refusal is not a physical impossibility result.

Bob's matrix entries and PSD work are charged once per **declared effect** of
dimension dim(Q_B), not for a fabricated copy at every Alice history. Each
Alice outcome effect is built once; the work estimate includes each K†K,
weighted sum and completeness sum. Every declared leaf, including impossible
ones, is charged for its B⊗A product, two joint lifts and traces, and both
verifier-effect PSD checks. Each state occurrence is charged at joint dimension
dim(R_keep) × dim(Q_B) × dim(Q_A). These are size proxies, not time/RAM limits;
exact rational intermediates can grow. No all-history table of joint effects
is materialized.

This is a fixed trusted preparation, fixed wire order and finite declared
strategy. There are no additional prover-owned preshared states or hidden
communication devices, adaptive quantum preparations, intermediate verifier
channels, callbacks, arbitrary event programs, or strategy optimization here.
The checker does not certify noncommunication in an experiment, self-testing,
or a circuit-depth bound. There is no serialized protocol format or CLI.

Run the local-view, admission and independent complex-state checks with:

```sh
python -m pytest -q tests/protocols/interactive/test_two_prover*.py
```
