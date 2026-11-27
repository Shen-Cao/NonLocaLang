# Network witnesses and compatibility

Networks constrain how resources reach the parties. Source factorization affects
which behaviors are possible even when a linear game has the same classical
optimum as an ordinary Bell game. Compatibility, nonlinear tests, and linear-value
optimization therefore have separate guarantees.

## Source structure and witness data

`SourceNetwork(parties, sources)` records source recipient lists. For example,
`SourceNetwork(3, ((0, 1), (1, 2)))` is a bilocal chain. Each source is independent;
there is no additional global shared variable. Source order and endpoint order are
preserved because they determine quantum tensor-factor ordering. Repeated recipient
sets represent separate independent sources. Isolated parties and an empty source
list are permitted. In a quantum witness, each party's POVM dimension equals
the product of its incident source-factor dimensions. A unary source such as
`(0,)` supplies an explicit independent local ancilla and counts toward that
party's dimension bound. An isolated party has dimension one. A projective
measurement contract never introduces an implicit dilation.

`ClassicalNetworkStrategy` stores one rational latent-variable distribution per
source and one local rational conditional-response table per party. Its rule builder
uses `rule(local_input, incident_source_values, local_output)`. Each response table
is ordered by local input, incident source values in source-index order, then output.
`probabilities()` returns exact input-major/output-minor Fractions. This gives a
constructive witness for the resulting behavior; it is not an unrestricted network
membership solver.

## Linear game tasks

For a linear game score, unrestricted classical network response functions
have exactly the ordinary local optimum: every deterministic local strategy
can ignore all sources, and every network behavior is ordinary local when the
source variables are collected into one joint hidden variable. This remains
true for fixed valid source laws. Consequently the network set's convex hull
is the ordinary local polytope, even though network membership can differ.
An exact local optimizer is therefore valid for this linear-value question;
it cannot decide network membership or optimize nonlinear witnesses.

```python
from fractions import Fraction
from nonlocalgames import AnalysisTask, Resource, ResourceSpec, SharedRandomness
from nonlocalgames.games import chsh
from nonlocalgames.networks import SourceNetwork

# No common source: only independent local response randomness is available.
network = SourceNetwork(2, ())
task = AnalysisTask(chsh(), ResourceSpec(
    Resource.LOCAL, network=network,
    shared_randomness=SharedRandomness.FORBIDDEN,
))
result = task.solve()
assert result.value == Fraction(3, 4)
assert result.guarantee == "exact"
```

For quantum networks, `.evaluate()` requires a `QuantumNetworkStrategy` carrying
the declared graph. Generic quantum see-saw and NPA dispatch are rejected because
they do not enforce source factorization. An explicit network witness may also be
checked against dimension, state, and measurement restrictions. Those restrictions
refer to the assembled party state, not independently to a source's internal labels.
A network contract cannot also supply an arbitrary fixed global state.

## Quantum witnesses and tensor conventions

`QuantumSource(state, dimensions)` records the endpoint factors of one source.
`QuantumNetworkStrategy` forms their tensor product, permutes factors into party
order, and applies each party's joint POVM on its incoming systems. A party's local
factors follow source-index order. `as_strategy()` deliberately forgets the source
structure for ordinary Born-rule evaluation; it does not prove that an arbitrary
global quantum state has the declared source decomposition.

```python
from nonlocalgames.networks import (
    BILOCAL_SCENARIO, bilocal_entanglement_swapping, bilocal_witness,
    triangle_xor_strategy,
)

triangle = triangle_xor_strategy()
exact_behavior = triangle.probabilities()

swapping = bilocal_entanglement_swapping()
evidence = bilocal_witness(swapping.probabilities(BILOCAL_SCENARIO))
print(evidence.I, evidence.J, evidence.value)  # approximately 1/2, 1/2, sqrt(2)
assert evidence.guarantee == "numerical_witness"
```

See the [static RGB4 triangle](rgb4.md) for a fixed, exactly contracted quantum
example and an explicit source-dephased classical control with the same token counts.

## Bilocality inequality

The bilocal witness has inputs `(2, 1, 2)` and outputs `(2, 4, 2)`. Bob's output
encodes two bits as `b=2*b0+b1`. It tests `sqrt(abs(I))+sqrt(abs(J)) <= 1` under
independent classical AB and BC sources. It is a nonlinear test of a full behavior,
not a scalar Bell-game optimum. Exact rational input gives exact rational `I,J` and
an exact comparison with the threshold, even though the displayed square roots are
floating-point. A nonviolation does not establish bilocal compatibility. Neither
rational nor numerical evaluation supplies finite-sample statistical confidence.

`bilocal_deterministic_branches()` demonstrates the danger of implicit global
randomness: its two bilocal branches have `(I,J)=(1,0)` and `(0,1)`, while their equal
mixture violates the bilocal bound. That mixture is ordinary Bell-local. A local
polytope backend therefore cannot replace an independent-source compatibility test,
even though it correctly computes the network classical linear game value.


## Persisting witnesses

`nonlocalgames.networks.serialization` provides `network_to_dict/from_dict`,
`classical_to_dict/from_dict`, and `quantum_to_dict/from_dict`. Graph and witness
schemas are versioned separately. Classical probabilities and responses use rational
strings; quantum matrices use explicit `complex_float64` real/imaginary arrays.
Deserialization revalidates every source and response/POVM. Numerical quantum data
retain their tolerances and do not acquire exact-certificate status by serialization.

## Necessary conditions and an exact fragment

The network compatibility tools check conditions on a supplied full behavior table.
A violation excludes the stated network model; passing the conditions normally
leaves compatibility unresolved. Inputs must be independent of source variables.
There is no hidden global selector, communication, or conditioning on other parties'
outputs.

For groups of parties that share no incident source, their output marginals must
factor at each fixed input. This condition holds for both classical and quantum
independent-source witnesses. In the chain A--B--C it implies independence of the
unconditional A,C marginal. Conditioning on Bob's outcome can correlate A and C,
so postselected probabilities require a different task.

For a classical triangle with pair sources AB, BC, CA, every event satisfies
`p(a,b,c|x,y,z)**2 <= p_A(a|x)*p_B(b|y)*p_C(c|z)`. Local stochastic response functions
are allowed. The common-random-bit distribution `p(000)=p(111)=1/2` fails this test:
`1/4 > 1/8`. This necessary condition applies to both classical and quantum independent
pair sources; its quantum scope follows the cited network theorem. It is not
a general nonsignalling-network test.

There is a useful sufficient condition for one restricted case: a **no-input**
bilocal chain admits a classical model exactly when its A,C marginal factors.
Independent sources sample `a` and `c` from their marginals; Bob samples `b` using
`p(a,b,c)/(p_A(a)*p_C(c))`. Zero-probability rows use any normalized response.
Rational data give a rational constructive witness. This characterization also
covers the quantum no-input chain because source-disjoint independence is necessary
there and every classical construction has a quantum embedding. It does not extend
to general multi-input bilocal behaviors.

## Running exact compatibility checks

```python
from fractions import Fraction
from nonlocalgames import Scenario
from nonlocalgames.networks import SourceNetwork
from nonlocalgames.networks.behavior import RationalBehavior
from nonlocalgames.networks import (
    check_source_independence, triangle_finner, verify_compatibility_violation,
    violation_to_dict,
)
from nonlocalgames.networks.membership import reconstruct_bilocal_chain

scenario = Scenario((1, 1, 1), (2, 2, 2))
chain = SourceNetwork(3, ((0, 1), (1, 2)))
triangle = SourceNetwork(3, ((0, 1), (1, 2), (2, 0)))

# Independent end bits and their XOR at the middle party.
xor_behavior = RationalBehavior(scenario, tuple(
    Fraction(1, 4) if b == (a ^ c) else Fraction(0)
    for a, b, c in scenario.outputs
))
model = reconstruct_bilocal_chain(chain, xor_behavior)
assert model.status == "compatible"
assert model.witness.probabilities() == xor_behavior.probabilities

# A shared common bit requires correlations unavailable in these source graphs.
common_bit = RationalBehavior(scenario, tuple(
    Fraction(1, 2) if a == b == c else Fraction(0)
    for a, b, c in scenario.outputs
))
assert check_source_independence(chain, common_bit).status == "incompatible"
excluded = triangle_finner(triangle, common_bit)
assert excluded.status == "incompatible"
saved_violation = violation_to_dict(excluded.violation)
rechecked = verify_compatibility_violation(
    triangle, common_bit, excluded.method, saved_violation,
)
assert rechecked.status == "incompatible"
```

`check_source_independence(network, behavior)` checks all source-disjoint nonempty
party-subset pairs. Enumeration is guarded because the number of pairs grows
exponentially. Use `groups=((0,), (2,))` to select particular disjoint groups;
explicit groups that share a source are rejected. `triangle_finner` requires
exactly one pair source on every edge of a three-party triangle. Both methods
return `incompatible` with an exact event/marginal violation, or `not_ruled_out`.
A triangle has no source-disjoint party pairs, so the independence checker may
perform zero checks; that is correctly inconclusive rather than compatibility.

`violation_to_dict/from_dict` use exact rational strings in a versioned certificate.
`verify_compatibility_violation` rechecks the named input/event and applicability
conditions without searching over all checks. It rejects altered values, an invalid
relation, or a certificate that does not actually violate the necessary condition.
`behavior_to_dict/from_dict` in `networks.behavior_serialization` separately serialize the full
rational behavior used to verify that certificate.

`reconstruct_bilocal_chain(network, behavior)` returns `compatible` with a rational
`ClassicalNetworkStrategy`, or `incompatible` with an exact marginal violation.
It requires no inputs and the source edges AB and BC with party labels 0,1,2;
source and endpoint order may vary. It imposes no source-cardinality or quantum
dimension bounds. Its constructed source alphabets have sizes equal to the end
parties' output alphabets, and these additional resources must be checked separately
if a more restrictive task is intended.

The new compatibility checks require `RationalBehavior`, which accepts rational
strings, integers, or Fractions and validates exact normalization and nonsignalling.
It rejects floating data; no automatic rational reconstruction is performed.
The separate `bilocal_witness` evaluator also accepts floating data and retains a
numerical evidence label in that case. None of these routines provides finite-sample
confidence guarantees. Primary
sources are [Finner's inequality](https://doi.org/10.1214/aop/1176989534) and
[network correlation constraints](https://arxiv.org/abs/1901.08287); the report gives
explicit proofs for the implemented fragments.


## Typed compatibility tasks and classical inflation

`NetworkCompatibilityTask(behavior, resource)` separates distribution questions
from scalar `AnalysisTask` game values. Its resource must include a source graph.
Call `capabilities()` to inspect applicable checks, then `check(method=...)` with an
explicit choice. Necessary-condition passes never become membership claims.

The `spiral` method implements a fixed classical fanout inflation for no-input
triangles. It can exclude the W distribution, which passes the elementary Finner
checks. It is rejected for quantum resource tasks: classical copied-variable
constraints are not quantum fanout constraints. See [spiral inflation](spiral-inflation.md)
for its proof, certificates, numerical fallback states, and allocation budgets.

Compatibility tasks have separate persistence helpers in
`core.compatibility_serialization`: `compatibility_to_dict/from_dict`,
`dump_compatibility`, and `load_compatibility`. The CLI preserves the resource scope:

```sh
python -m nonlocalgames network --file output/network-task.json --capabilities
python -m nonlocalgames network --file output/network-task.json --method spiral --witness
python scripts/spiral_benchmark.py > output/spiral-benchmarks.json
```

The file must contain a serialized `NetworkCompatibilityTask`. The `--time-limit`
option is a per-LP HiGHS limit for spiral discovery. A timeout without an exactly
verified certificate returns unresolved evidence, never a classical or quantum
incompatibility claim. Certificate rechecking itself uses exact rational arithmetic.
