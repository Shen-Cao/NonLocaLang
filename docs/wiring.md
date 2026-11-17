# Compatibility across hidden wirings

A wiring target specifies a conditional behavior for each route. The question
is whether **one finite classical strategy** can produce all the behaviors.
The preparation in each laboratory and its response to any given local view
are shared across routes. Branch targets include every question profile.

## One blind exchange

Each origin jointly prepares two classical ports, keeps the first and
sends the second to an independent passive relay. Preparations and private
randomness are independent across laboratories. A route `r` is a permutation:
`r[i]` is the sender whose transmitted port arrives at receiver `i`. If
`r[i] == i`, both ports came from the same joint preparation. Questions arrive
after routing. Receiver `i` can use its local question, retained data, incoming
data, and the incoming sender's identity. The relay supplies no global route tag.

For a finite witness, let `q[i](k,s)` be its joint preparation law and
`D[i,j](a | x,k,s)` its stochastic response for incoming sender `j`.
The evaluator computes each branch exactly:

```text
P_r(a | x) = sum_{k,s} product_i q[i](k_i,s_i)
                         × product_i D[i,r[i]](a_i | x_i,k_i,s_{r[i]}).
```

A source may correlate its own two ports. For self-return the evaluator draws
both port values together. For distinct origins the two values at a receiver
have independent preparation laws. Each reachable receiver/sender pair has
one response table in the entire witness.

## A necessary equality

Fix receiver `i`, local question `x_i` and incoming origin `j`. Summing over
the other answers eliminates their normalized response tables. The remaining
probability for `a_i` depends only on the distribution of the two ports at
`i` and its response table. Both are the same in any route with `r[i] == j`.
This proves equality of the unconditional marginals across these routes.
Nonsignalling in the late questions allows the checker to fix other questions
at zero. Conditioning on other parties' outcomes would invalidate this step.

For example, with receivers ordered A,B,C, the route ABC is `(2,0,1)` and AC
is `(2,1,0)`. Alice receives Charlie's port in both. At the same local
question, demanding that Alice always output zero in ABC and one in AC yields
two feasible individual branches with an incompatible joint target.

A strategy that exactly matches every branch proves compatibility. A marginal
contradiction disproves it. Other outcomes remain unknown. In particular, a
proposed strategy can miss a target that another strategy realizes. Assessment
records bind the resource, target, and existential query; replay uses the
target and the actual strategy tables.

## Reproduce the controls

```sh
mkdir -p output/wiring
python -m nonlocalgames wiring example bit-exchange > output/wiring/target.json
python -m nonlocalgames wiring example bit-exchange-strategy > output/wiring/witness.json
python -m nonlocalgames wiring check output/wiring/target.json --strategy output/wiring/witness.json
python -m nonlocalgames wiring check output/wiring/target.json
python -m nonlocalgames wiring example inconsistent-alice > output/wiring/impossible.json
python -m nonlocalgames wiring check output/wiring/impossible.json
```

The expected truth values are `proved`, `unknown`, and `disproved`, respectively.
In the positive control each party copies its private fair bit to its two ports
and outputs the XOR of its retained and received values. In the three-cycle
the four even-parity triples each have probability 1/4. In AC the possible
triples are 000 and 101, each with probability 1/2.

In Python, construct a `WiringArena` from a `Scenario` and routes. Pass a
route-keyed mapping of `RationalBehavior` objects to `WiringTarget`.
`ClassicalWiringStrategy` accepts kept and sent alphabet sizes, one joint law
per origin (kept-major, sent-minor), and a mapping keyed by `(receiver,sender)`
whose tables are flattened in question, kept value, incoming value, answer order.
`assess_wiring(target, strategy=witness)` checks the compatibility query.

Budgets limit parties, routes, stored entries, scalar input sizes, and counted
evaluation work. Work grows with the product of the joint preparation-table
lengths even for small output alphabets. JSON decoding, content hashing, branch nonsignalling validation,
and the bit complexity of exact arithmetic have separate costs. `budget_exceeded` is an operational
outcome with unknown truth.

The [fermionic rewiring experiment](https://arxiv.org/abs/2606.12363v1)
motivates the timing, persistent port ownership, and local-view constraint.
The executable resource here is independent finite classical preparation with
rational probability tables for its constructive witnesses. The [six-mode fermionic reference](fermion-reference.md) implements the specified
graded route action, parity-preserving measurements, and exact algebraic output
probabilities. Its comparison receipt addresses equality with a fixed strategy.
