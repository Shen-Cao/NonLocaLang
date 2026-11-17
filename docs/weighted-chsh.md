# Resumable weighted CHSH experiment

At grid index `k` the question pair 11 has weight `p=k/steps` and each of
00, 01, and 10 has weight `q=(1-p)/3`. The winning parity is the CHSH
predicate. Each member asks whether the unrestricted finite quantum value
strictly exceeds the classical value. Both endpoints belong to the grid.

```sh
python -m nonlocalgames weighted-chsh run output/chsh-grid --max-new 3
python -m nonlocalgames weighted-chsh replay output/chsh-grid --members
python -m nonlocalgames weighted-chsh run output/chsh-grid
python -m nonlocalgames weighted-chsh replay output/chsh-grid --members
```

At the defaults the first command records indices 0 through 2 of 21. Repeating
`run` with the same scientific settings verifies the saved prefix and continues
at the next point. `--max-new` controls only the current invocation. Options
that change the experiment require a separate directory. For example:

```sh
python -m nonlocalgames weighted-chsh run output/chsh-coarse --no-screen --grid-denominator 2
```

## Reusing the numerical screen

```sh
python -m nonlocalgames weighted-chsh run output/chsh-template --reuse-screen --max-new 3
python -m nonlocalgames weighted-chsh run output/chsh-template --reuse-screen
python -m nonlocalgames weighted-chsh replay output/chsh-template --members
```

`--reuse-screen` gives each generation invocation one parameterized CVXPY
program for its fixed question and answer alphabets. A new set of exact
weighted XOR coefficients supplies the objective for each point. The session
sets `warm_start=False` and returns independent numerical results. Each
invocation owns its session, including when two processes publish to the same
run directory. Replay reconstructs the exact claims directly from saved evidence.

The template policy and effective solver options are part of the v2 run
identity. To compare policies, generate them in separate run directories.
Archives made by the default fresh-solve path keep the v1 manifest and remain
replayable. Both policies use the same rational witness and bound checkers.

## Evidence at each point

Exact local enumeration supplies a local optimum and a deterministic witness.
The finite quantum proposal uses a Bell pair. Alice measures at planar angles
0 and 2θ and Bob at θ and −θ. For `t=j/D`, where `D` is the grid denominator,
the cosine `c=(1-t²)/(1+t²)` and sine `2t/(1+t²)` are rational. The winning
score is one half plus one half times the bias

```text
3q cos θ − p cos(3θ) = 3(q+p)c − 4pc³.
```

This cubic is maximized on the finite grid by checking its endpoints and
the grid points bracketing the stationary maximum. Monotonicity of the grid
cosines permits binary search using exact comparisons. Every selected strategy
is verified through the ordinary exact Born-score checker.

For `p ≤ 1/10`, a candidate XOR upper certificate has diagonal
`(q/2, (q-p)/4, q/2, (q-p)/4)` in the score convention. For `p=1` the diagonal
is `(0, 1/4, 0, 1/4)`. The rational PSD checker decides feasibility. At
`p=1/10` the local and quantum values both equal 9/10. A nonpositive certified
lower bound on the gap alone is insufficient to resolve the strict claim.
With coarser angle grids some interior points remain unknown.

Numerical XOR screening is optional. The result records the requested solver's
status and an observed candidate gap. A missing solver or failed numerical
solve has a separate screen status. The verdict comes from rechecking exact
artifacts. `replay --members` exposes checked bounds and the numerical observation
for each recorded parameter.

The 21-point grid with `D=16` yields 17 proved strict advantages and four
disproofs. At `D=2` it yields 13 proofs, four disproofs and four unknowns.
Here the change in verdict comes from the available lower certificates.

## Checkpoints and reproducibility

`manifest.json` binds the parameter grid, claim, algorithm versions, witness
denominator, verification budgets, deterministic seed policy, requested and
effective solver settings, package source digest and runtime provenance. The
source digest includes package-relative paths and source bytes. A Git revision
is included when the executing package is in a checkout. A changed implementation
or configuration creates a different run identity.

`members/k.json` binds the parameter, verifier, claim, raw evidence and assessment
to that run. A completed temporary file in the same directory is published with
an atomic hard link. This requires a filesystem that supports hard links. If
two processes compute a member concurrently, the losing writer reads and checks
the published winner. The protocol handles process interruption; storage-device
durability depends on the filesystem. A missing suffix is pending work. A hole
inside the published prefix is an error.

Replay recomputes exact verdicts from stored evidence using local verification
limits. It validates the format and consistency of recorded screen observations;
reproducing a floating-point solve is a separate experiment. A recorded
generation failure carries an operational status and no proof. Use
`--budget path.json` with fields from `VerificationBudget` to choose limits.
Run manifests cannot enlarge the verifier's replay limits. Completed mathematical
unknowns, generation failures, and pending grid indices remain distinguishable.
