# Resource contracts

A `ResourceSpec` describes admissible strategy branches separately from the
`Game`. Restrictions compose: the finite tensor-product model can have local
dimension upper bounds, a fixed state, a state-family constraint, a measurement
class, and an external shared-randomness policy.

| Field | Semantics |
|---|---|
| `kind` | Local, finite tensor quantum, closure, commuting quantum, or nonsignalling |
| `local_dimensions` | Upper bounds on local quantum system dimensions |
| `shared_randomness` | Whether an external classical variable may coordinate branch selection |
| `fixed_state` | A validated numerical state with a specified local tensor factorization |
| `state_constraint` | Any state, a product across all parties, or PPT across explicit cuts |
| `ppt_cuts` | Nonempty proper subsets of parties on which partial transpose is positive |
| `measurement_class` | All POVMs or projective measurements, including zero projectors |

External shared randomness is distinct from classical correlations already
present in a mixed state. With shared randomness allowed, every branch obeys
the dimension and state constraints; a classical branch label is not counted
against the declared quantum dimension. Fixed-state branches all use the same
state. Forbidding this selector does not force a quantum state to be product.
For the local model, forbidding it restricts behaviors to product responses.

Linear optimization is insensitive to convexification by shared randomness,
although behavior membership can change. With no fixed state, the PRODUCT
quantum resource has exactly the classical optimum: every resulting branch
has product response probabilities, and deterministic local strategies are
realizable using identity/zero effects in any positive local dimension. This
also holds for projective measurements. PRODUCT is not a separable mixed-state
membership test.

PPT is a positivity constraint, not a general separability characterization.
Projective dilation can consume dimension and is therefore never used to
silently replace a bounded-dimensional POVM contract. A product or PPT check
on a supplied floating state remains numerical evidence. In particular, a
numerical fixed state that passes a product tolerance check does not authorize
an exact value claim from that check alone.

`AnalysisTask.capabilities()` lists applicable method names. `solve(method=...)`
checks the resource before selecting a backend; explicit NPA/XOR calls through
this API reject dimension, fixed-state, and state-family restrictions they do
not enforce. The raw low-level backends continue to solve their documented
unrestricted mathematical programs. `evaluate(strategy)` checks a supplied
quantum branch against the complete resource contract before scoring it.

The resource and task codecs live in `core.resource_serialization`:
`resource_to_dict`, `resource_from_dict`, `task_to_dict`, `task_from_dict`,
`dump_task`, and `load_task`. Versioned schemas preserve game fractions exactly
and explicitly mark fixed-state arrays as numerical complex matrices with a
validation tolerance. They do not encode executable code, solver defaults,
local file references, or implied numerical proof guarantees.

Independent-source networks add an explicit `network=SourceNetwork(...)` field.
They require `shared_randomness="forbidden"`: mixing whole network branches using
an external global selector can destroy source independence. The current graph
contract applies to local and finite quantum resources. Source order and endpoint
order determine quantum tensor factors and are preserved in serialization.

A network task accepts only a corresponding factorized network witness in
`evaluate`; passing an unrestricted global quantum state is rejected. Global
product/PPT/measurement restrictions remain checked on the assembled party state
and local measurements. A global fixed state is not accepted as a substitute for
independent source states. The graph is preserved in resource schema v2; schema v1
loads as an ordinary resource without a graph.

For classical networks, every ordinary deterministic local strategy is available
by ignoring sources, while every network behavior is ordinary local. Consequently
all linear game values equal the ordinary classical value even though behavior
sets differ. Exact local optimization returns a source-ignoring network witness.
General quantum-network optimization is unsupported; the unrestricted product-state
subcase still has the exact local optimum. See [network analysis](networks.md) for
nonlinear tests and constructive compatibility methods.
