# Six-mode fermionic reference

Three laboratories independently split one fermion between a retained mode and
a sent mode. A passive relay redistributes the sent ports. Each laboratory then
receives a bit and measures its incoming and retained modes. The definitions
follow [Moradi Kalarde et al., v1, Fig. 2 and Appendix B](https://arxiv.org/abs/2606.12363v1).

```sh
mkdir -p output/fermion
python -m nonlocalgames fermion-reference target > output/fermion/target.json
python -m nonlocalgames fermion-reference check output/fermion/target.json
python -m nonlocalgames fermion-reference check output/fermion/target.json --statistics ordinary_tensor
```

The first comparison tests equality with the specified fermionic preparation,
route action, and measurements. The second compares the same data with one
ordinary tensor strategy with the corresponding local states and measurement
bases. Each receipt records both target IDs, the reference ID, the number of
entries compared, and the first disagreement when one exists. A mismatch
answers the question about that reference. Exclusion of every allowed qubit
strategy in the paper uses cross-wiring rigidity (Theorem B.3).

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

An independent test treats the state as three occupied one-particle orbitals
on six modes, applies local optical rotations and squares the corresponding
three-row determinants. It compares all `5 × 8 × 64 = 2560` entries against
the density calculation using numerical tolerances. Export, reference comparison
and target validation use exact coefficient pairs.
