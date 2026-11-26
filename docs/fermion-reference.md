# Fermionic routing: a six-mode reference and an even-packet control

Three laboratories independently split one fermion between a retained mode and
a sent mode. A passive relay redistributes the sent ports. Each laboratory then
receives a bit and measures its incoming and retained modes. The definitions
follow [Moradi Kalarde et al., v1, Fig. 2 and Appendix B](https://arxiv.org/abs/2606.12363v1).

```sh
mkdir -p output/fermion
python -m nonlocalgames fermion-reference target > output/fermion/target.json
python -m nonlocalgames fermion-reference check output/fermion/target.json
python -m nonlocalgames fermion-reference check output/fermion/target.json --statistics ordinary_tensor
python -m nonlocalgames fermion-reference check output/fermion/target.json --statistics fermionic_even_pair
```

The default compares the data with the specified single-mode fermionic
preparation, route action and measurements. The other two commands compare
it with a zero-phase ordinary tensor strategy and its even-packet fermionic
encoding. Each receipt records both target IDs, the reference ID, the number of
entries compared, and the first disagreement when one exists. A completed
comparison exits with status 0 even when `matches_reference` is `false`.
Check that JSON field to tell whether the target matched. A mismatch
answers the question about that reference. Exclusion of every allowed qubit
strategy in the paper uses cross-wiring rigidity (Theorem B.3); these
comparisons do not establish that theorem or self-test any device.

## Conventions

- Parties are ordered A, B, C. A route lists the sender at each receiver.
  ABC is `(2,0,1)`. A self-return brings both ports from one preparation to the
  same laboratory.
- At each site the occupation basis is ordered **incoming, retained**.
  The one-particle measurement block uses **retained only, incoming only**
  (`01,10`). The four answers `0,1+,1-,2` use indices 0 through 3.
  The detector oracle uses the Appendix B plus/minus labels for every route.
- Measurements preserve local particle number. In the one-particle block,
  Alice's observable is `(X + (-1)^x Z)/sqrt(2)`. Bob and Charlie use `Z`
  on question zero and `X` on question one. The plus effect is `(I+O)/2`.
  The other two effects measure vacuum and double occupancy.

There are eight choices of whether each origin's particle was kept or sent.
List the occupied destination modes for each choice in origin order. Putting
these modes in final order multiplies its amplitude by the sign of their
permutation. The density coefficient for a pair of choices is the product
of their signs divided by eight. In the ordinary tensor control the signs are
positive. Contracting these coefficients with local projectors includes
interference between the all-kept and all-sent contributions.

The projectors and output probabilities belong to the ordered field
`Q(sqrt(2))`. A scalar is encoded as two canonical rational strings
`[a,b]`, denoting `a+b*sqrt(2)` in the positive real embedding. Each target
binds every branch to its route and validates nonnegativity, normalization,
and nonsignalling in the late classical questions. Existing rational
compatibility claims have their own wire schema.

## Controls

After combining the two single-occupation outcomes, the probability of one
particle at every site is 1/4 in a three-cycle and 1/2 in a two-cycle. Both
references pass this check. In BC, set both participating parties' inputs to
one and sum over Alice's answer. Within the one-particle branch the fermionic
signs are opposite and the ordinary tensor control gives equal signs. Each
opposite-sign pair has unconditional probability 1/4 in the fermionic reference.

One independent test treats the state as three occupied one-particle orbitals
on six modes, applies local optical rotations and squares the corresponding
three-row determinants. It compares all `5 × 8 × 64 = 2560` entries against
the density calculation using numerical tolerances. A second oracle works
exactly, without permuting modes: disjoint cycles factor; uneven token counts
fix the split whenever feasible; the all-one sector interferes with relative
phase `(-1)^(cycle length - 1)` for single modes, and +1 for even packets.
Export, reference comparison and target validation use exact coefficient pairs.

### Same target does not mean same physical model

`fermionic_even_pair` encodes each logical port as `0 → 00, 1 → 11`.
Each origin independently prepares `(|11;00> + |00;11>)/sqrt(2)` across
its sent and kept ports. There are twelve physical modes, six occupied in
every term. The same graded mode permutation acts on the eight coherent
terms; no 4096-by-4096 matrix is needed. Moving an occupied pair through
another occupied pair makes four fermionic crossings, so it contributes
no minus sign. Measurements act as the logical projectors on the reached
code space; their action outside that space is immaterial (one can assign
its orthogonal complement to answer 0). The labels count logical tokens,
not individual physical fermions.

This construction is the double-mode encoding discussed for the fixed
three-party loop in [Moradi Kalarde et al., Theorem 1](https://arxiv.org/html/2611.25074v1).
In this evaluator its 2,560 probabilities agree exactly with `ordinary_tensor`,
despite having a distinct reference ID. Against the single-mode `fermionic`
target it differs in **0, 0, 24, 24, 16** cells for **ABC, ACB, AB, AC, BC**.
Comparing these two specified strategies does not exclude all double-mode
strategies or prove uniqueness up to parity-preserving isometries.

### What a receiver knows

Sources are fixed before routing. A readout can use its receiver, incoming
sender and own question, but not the global route or outgoing destination.
For example, Alice receiving Charlie's port has the same readout in ABC
and AC. The 30 route/receiver/question occurrences share 18 local contexts,
including the self-returns. A Python calibration control can exchange the
two sign labels in one such context, across **all** routes where it occurs:

```python
from nonlocalgames.protocols.fermion_reference import ReadoutContext, balanced_strategy_target

alice_from_charlie = ReadoutContext(receiver=0, incoming_sender=2, question=1)
calibrated = balanced_strategy_target(swap_sign_labels=[alice_from_charlie])
```

This is a restricted family of fixed sources, projectors and optional local
output swaps, not a solver for general preparations or POVMs. Sharing the
keys matters: if a party were told the route, an ordinary tensor strategy
could reproduce even the single-mode table by flipping only B in AB, C in
AC and B in BC at that party's question 1. These are not allowed local keys.

### Applying a graded permutation to other occupation states

Use `ModePermutation` when the relabeling is source-to-destination, or
`relay_mode_permutation` for a sender-at-each-receiver route. An occupation
mask uses bit `i` for mode `i`; `apply` returns the moved mask and the sign
of its **amplitude**, which must be retained when adding coherent paths.

```python
from nonlocalgames.protocols.graded import ModePermutation, relay_mode_permutation

cycle = ModePermutation((1, 2, 0))
assert cycle.apply(0b011) == (0b110, +1)
assert cycle.inverse().apply(0b011) == (0b101, -1)
assert cycle.then(cycle.inverse()).apply(0b011) == (0b011, +1)

swap = ModePermutation.from_packets((1, 0), widths=(2, 2))
assert swap.apply(0b0101) == (0b0101, -1)  # |10;10> -> -|10;10>
assert swap.apply(0b0011) == (0b1100, +1)  # |11;00> -> +|00;11>
assert relay_mode_permutation((1, 2, 0)).destinations == (4, 1, 0, 3, 2, 5)
```

Packet widths are supplied in source order and may differ. Internal mode
order is preserved. The default bound is 256 modes (override with
`max_modes`); this is a basis-state primitive, not a dense-state simulator.
