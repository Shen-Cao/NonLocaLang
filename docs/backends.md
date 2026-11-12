# Optional numerical and language backends

Python is the public construction and verification interface. Additional engines
are introduced when a defined research experiment needs their capabilities, with
an explicit translation contract back to the original game and resource model.
The base Python installation does not require Julia.

A future Julia adapter should use a dedicated project rather than the user's
global environment. Pin an upstream revision where no registered release exists;
record Julia, package, and solver versions in run output. Do not store machine
paths, license settings, global depot contents, or workstation defaults in source.

The subprocess boundary should use versioned JSON with rational strings, complete
resource assumptions, verifier identities, requested solver, and budgets. Invoke
Julia through an argument list with startup/history disabled, send logs to stderr,
and reserve stdout for machine-readable results. Preserve solver failure or missing
dependency information; do not install packages or switch solvers implicitly.

The [PCPOP project](https://github.com/abh1mishra/PCPOP.jl) is an integration candidate
for partially commutative polynomial optimization. Its exposed models, monomial
maps, and moment/localizing bases are useful translation objects. The
[PCPOP paper](https://arxiv.org/abs/2607.09339) describes exact algebra and SDP
rounding, but that is not a universal portable certificate contract. A bridge must
select a solver explicitly and treat numerical results as numerical until an
applicable independent verifier accepts their evidence.

Before adopting an adapter, require a small cross-check corpus, explicit supported
resource fragments, recoverable objective/constraint conventions, allocation and
runtime limits, and an artifact that can be rechecked without trusting a backend's
label. No Julia solver adapter is implemented at this stage.
