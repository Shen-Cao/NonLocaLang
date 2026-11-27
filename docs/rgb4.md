# RGB4: exact static triangle benchmark

Can three parties, each with one fixed measurement and no communication, produce
a specified joint distribution using only independent classical pair sources?
At the RGB4 point below, an exact quantum construction matches all 64
probabilities. It violates a necessary classical condition **on the exact
fair-token coarse law**. Dephasing the sources gives an explicit classical model
with the **same** coarse law. The question is about the full distribution, not
the optimum of a linear game.

## Run the comparison

```python
from fractions import Fraction
from nonlocalgames.networks import (
    RGB4_NETWORK, RationalBehavior, rgb4_behavior, rgb4_dephased_strategy,
    rgb4_token_counting_margin,
)

quantum = rgb4_behavior()
classical = rgb4_dephased_strategy()
dephased = RationalBehavior(classical.scenario, classical.probabilities())

assert rgb4_token_counting_margin(RGB4_NETWORK, quantum) == Fraction(115125, 38614472)
assert rgb4_token_counting_margin(RGB4_NETWORK, dephased) == Fraction(-316875, 38614472)
```

`rgb4_behavior()` returns an ordinary `RationalBehavior`, so it also works with
`behavior_to_dict` and `behavior_from_dict`. There is one input per party
(the sole setting, not a choice). The integer outcome indices `0,1,2,3` mean
`0,1_0,1_1,2`, respectively.

## How the table is constructed

The three independent sources α=BC, β=CA, γ=AB each prepare
`(|01⟩+|10⟩)/√2` in the listed endpoint order. The local factor orders are
A: βγ, B: γα, C: αβ. In particular B's **measurement** uses γ before α even
though the framework indexes B's incident **sources** as α then γ.

At each party the zero- and two-token kets are `|00⟩` and `|11⟩`. The
one-token kets are `(12|01⟩+5|10⟩)/13` and `(5|01⟩−12|10⟩)/13`.
The evaluator sums the eight product-source amplitudes for each outcome triple
and squares that sum. The result at this single rational point has 20 nonzero
entries and 44 zeros; the least common denominator is `8 × 13⁶ = 38,614,472`.
The tests independently evaluate
[Eq. (11) of Sekatski, Boreiri and Brunner](https://arxiv.org/pdf/2209.09921v2#page=3)
by sectors, including all cyclic permutations and the zeros. For the dephased
control the two all-one paths add as probabilities, not amplitudes: a fair bit
on each edge selects the endpoint that gets its token, and each party samples
the squared measurement overlap for its two incident bits. This is an exact
`ClassicalNetworkStrategy`, not a fit.

## Why the sign matters, and when it does not

Merge `1_0` and `1_1` to count one token. The comparison checks the entire
coarse distribution: `(1,1,1)` has mass 1/4, the six permutations of `(0,1,2)`
have mass 1/8 each, and every other triple has zero mass. An exact table that
does not pass this check is outside the scope of the comparison and raises
`ValueError`, even if the discrepancy is tiny.

The external mathematical dependency is [Renou and Beigi's token-counting
rigidity theorem, Thm. 1](https://arxiv.org/pdf/2202.00905v2#page=4). On an
independent-source classical triangle with this coarse law, the coarse counts
must come from three independent, fair token directions. Each of the two
oriented cycles in which everyone gets one token has mass 1/8. Consider the
cycle in which every party sees local bits `10`. Flipping the opposite edge
changes the other two counts to `(2,0)` for A (cyclically for B and C), without
changing the distribution of the two source variables at the party in question.
The two orientation events have equal mass. Thus the probability mass in this
cycle where A refines to `1_0` equals `P(1_0,2,0)`. The same reasoning applies
to B and C. Within the cycle, either all three refine to `1_1`, or at least one
refines to `1_0`. The union bound therefore gives

```text
1/8 ≤ P(1_1,1_1,1_1) + P(1_0,2,0) + P(0,1_0,2) + P(2,0,1_0).
```

The function returns **left minus right**. Positive excludes this classical
triangle model, assuming the cited rigidity theorem, independent sources and
no extra common randomness or communication. Nonpositive alone proves nothing
about compatibility; the dephased table is classical because it has a witness.
This is not an unconditional Bell inequality, a noise-robust or finite-sample
test, a self-test, or a bound on entanglement or private randomness.
