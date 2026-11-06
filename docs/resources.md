# Resource contracts and restricted analysis

A resource contract specifies which physical implementations are admissible.
A solver capability states what can be computed about that contract. These are
separate: declaring a restriction never authorizes an optimizer to drop it.

## Composable restrictions

`ResourceSpec` keeps the base `Resource` kind and adds the following fields.
Restrictions on states, dimensions, and measurements apply to
`Resource.QUANTUM_FINITE`; abstract closure and commuting tags do not acquire a
finite density-matrix interpretation.

| Field | Meaning |
|---|---|
| `local_dimensions` | One positive local dimension upper bound per party; `None` leaves it unspecified |
| `shared_randomness` | `ALLOWED` or `FORBIDDEN` external coordinated strategy-branch selection before inputs arrive |
| `state_constraint` | `ANY`, `PRODUCT` across all parties, or `PPT` across explicit cuts |
| `ppt_cuts` | Nonempty proper subsets of party indices; only used with `PPT` |
| `measurement_class` | Arbitrary `POVM` effects or `PROJECTIVE` measurements on the declared systems |
| `fixed_state` | A validated `FixedState` whose tensor factorization and matrix remain fixed |
| `network` | An independent-source `SourceNetwork`, or `None` for the ordinary Bell resource |

External randomness is distinct from correlations already encoded in a mixed
state. Forbidding an external branch variable does not require a pure state. When
shared randomness is allowed, every branch must individually satisfy the contract;
a fixed-state contract uses the same supplied state in every branch. External
branch labels are not counted against the quantum dimension bounds.

PPT cuts are canonicalized when the party count is known. Transposing a cut or its
complement imposes equivalent positivity constraints; redundant cuts are rejected.
PPT is not generally equivalent to separability. `PRODUCT` is a single tensor
product, not an arbitrary separable mixture. Projective measurements may have zero
projectors, so deterministic responses remain admissible in every positive dimension.
No projective dilation or extra ancilla is silently added to satisfy a restriction.

## Fixed-state search

```python
from nonlocalgames import AnalysisTask, Resource, ResourceSpec, SharedRandomness
from nonlocalgames.games import chsh
from nonlocalgames.quantum import FixedState, chsh_strategy

reference = chsh_strategy()
fixed = FixedState(reference.state, reference.dimensions)
task = AnalysisTask(chsh(), ResourceSpec(
    Resource.QUANTUM_FINITE,
    local_dimensions=(2, 2),
    shared_randomness=SharedRandomness.FORBIDDEN,
    fixed_state=fixed,
))
print(task.capabilities())
result = task.solve(method="seesaw", seed=7, restarts=2,
                    solver="SCS", solver_options={"eps": 1e-8})
assert result.guarantee == "numerical_witness"
print(result.value, task.evaluate(result.witness).diagnostics)
```

Here the search changes local POVMs only. Without a fixed state, the same method
also updates the state within the dimension bounds. This remains a bipartite
heuristic, and shared randomness does not turn its output into a global optimum.
The search retains an actual validated branch; it does not construct a convex
decomposition of arbitrary behaviors.

`FixedState` copies its matrix into immutable storage and records local dimensions
and an absolute numerical tolerance. Matrix acceptance, fixed-state equality,
product checks, and PPT checks are numerical statements. In particular, passing a
product-state tolerance test does not prove that the stored matrix is exactly a
tensor product. Such a matrix does not justify an exact classical-value certificate.

## PPT state restrictions and conditional optimization

```python
from nonlocalgames import AnalysisTask, Resource, ResourceSpec, StateConstraint
from nonlocalgames.games import chsh

ppt_task = AnalysisTask(chsh(), ResourceSpec(
    Resource.QUANTUM_FINITE,
    local_dimensions=(2, 2),
    state_constraint=StateConstraint.PPT,
    ppt_cuts=((0,),),
))
result = ppt_task.solve(method="seesaw", seed=7, restarts=2,
                        solver="SCS", solver_options={"eps": 1e-8})
```

PPT-constrained state updates are SDPs. A returned matrix is checked against all
specified partial-transpose cones. Numerical repair mixes with the maximally mixed
state when necessary; the actual repaired score is recomputed and repair diagnostics
are retained. This does not certify a rigorous bound.

For fixed supplied measurements, `task.solve(method="state_sdp", measurements=...)`
optimizes only the state. Direct `optimize_state(game, measurements, ppt_cuts=...)`
supports any finite party count, subject to its allocation guard. Without PPT cuts,
this is a largest-eigenvalue problem; with cuts, it is an SDP. Its exact mathematical
optimum would upper-bound states for those same measurements. It is **not** an upper
bound on the value after optimizing measurements as well. Raw block objectives and
scores of repaired witnesses are therefore kept separate.

## Exact cases and rejected lowerings

For an ordinary classical resource, allowing or forbidding external shared
randomness changes the behavior set but not any optimal linear score. Both cases
use exact deterministic optimization, with the resource policy recorded in the result.
For a symbolic product-state resource without a supplied numerical state, the same
exact reduction applies: product Born probabilities are local, and deterministic
outputs can be implemented using identity and zero projectors.

`task.capabilities()` lists available methods. Quantum methods are explicitly
selected. In particular, a restricted task cannot use the unrestricted NPA or XOR
backend as though it solved the restricted optimum. NPA would remain a possible
outer bound after an explicitly documented relaxation, but silently forgetting the
restriction is not part of this dispatch API. Projective-only measurement searches
are not implemented by the POVM see-saw.

## Persisting resource queries

`resource_to_dict` and `resource_from_dict` in
`nonlocalgames.core.resource_serialization` roundtrip resource contracts.
`task_to_dict`, `task_from_dict`, `dump_task`, and `load_task` include both the
verifier and its resource query. Resource schema
`nonlocalgames.resource.v2` includes the network graph; the reader also accepts
legacy v1 resources without a graph. Task and verifier schemas are versioned
separately. Fixed quantum states use explicit `complex_float64` numeric
encoding, including their dimensions and validation tolerance; this encoding
preserves numerical input data, not an exact algebraic state certificate.

## Independent-source networks

`SourceNetwork` records independent sources and their recipient parties.
Classical witnesses use exact rational source distributions and responses;
quantum witnesses keep explicit source density matrices and local POVMs.
Source independence is enforced structurally, and global shared randomness is
excluded. A network contract uses `Resource.LOCAL` or `Resource.QUANTUM_FINITE`
and explicitly sets `shared_randomness=SharedRandomness.FORBIDDEN`. A fixed global
state cannot substitute for source factors. Quantum dimension bounds and
product/PPT/projective restrictions apply to the supplied global party state and
local measurements after the source tensor factors have been assembled. Local
POVM dimensions equal the product of incident factor dimensions; a unary source
can represent an explicit local ancilla, with no implicit projective dilation. See
[network witnesses and compatibility](networks.md) for construction,
serialization, nonlinear checks, and the exact no-input bilocal fragment.

## Further backends

The current extension provides explicit source witnesses, bilocal and triangle
inequalities, source-disjoint marginal tests, and exact no-input chain reconstruction. General source-network optimization, inflation adapters, dimension-aware
outer relaxations, separability and entanglement-depth restrictions, trusted quantum
interfaces, and sequential instruments remain future capabilities. Relevant primary
sources are [dimension-constrained relaxations](https://arxiv.org/abs/1412.0924),
[the PPT criterion](https://arxiv.org/abs/quant-ph/9604005),
[its low-dimensional converse](https://arxiv.org/abs/quant-ph/9605038), and
[bilocal correlations](https://arxiv.org/abs/1112.4502).

## Serialized tasks from the command line

Write a task with `dump_task(task, "output/task.json")` after creating the output
directory. This preserves the verifier and all resource restrictions, including a
numerical fixed state or source graph. Inspect supported methods before solving:

```sh
python -m nonlocalgames task --file output/task.json --capabilities
python -m nonlocalgames task --file output/task.json --method seesaw \
  --solver SCS --seed 7 --restarts 2 --max-iterations 15 --witness
```

The second command applies only when the declared resource supports see-saw.
Classical or product-resource tasks can use `--method local`; unsupported methods
are rejected. Quantum tasks need explicit method selection. `state_sdp` is listed
as API-only because it additionally needs supplied measurement operators.

Output records the resource snapshot, solver settings, package versions, evidence
status, and optional witness. Ordinary quantum witnesses use
`quantum.serialization.strategy_to_dict/from_dict`; network witnesses retain their
source factors through the network schemas. Re-import a witness and call the
original task's `evaluate` method to check it against the original restrictions.
Serialization preserves numerical data and tolerances; it adds no proof guarantee.

Numerical resource loading rejects Boolean or string matrix entries and checks
budgets before allocating or eigendecomposing matrices. `resource_from_dict` and
`task_from_dict` accept `max_joint_dimension` and `max_matrix_entries`; `load_task`
also limits JSON bytes with `max_file_bytes`. Quantum-network decoding checks the
product source dimension and aggregate source/global/POVM entry count through
`max_total_dimension` and `max_matrix_entries`. Limits are configurable and failure
never silently truncates a state, drops a source, or changes a resource contract.
