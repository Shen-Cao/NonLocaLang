"""Exact marginal compilation for a fixed classical triangle spiral inflation."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import product
import json
from math import prod

from scipy.sparse import coo_matrix

from ..behavior import RationalBehavior, positive_budget
from ..behavior_serialization import behavior_to_dict, validate_behavior_scalars
from ..compatibility import validate_triangle
from ..serialization import network_to_dict

ROW_SCHEMA = "nonlocalgames.classical_spiral_rows.v1"
RESOURCE = "classical_fanout"
# Columns: (A1,B1,C1,A2,B2,C2). Each row block uses original A,B,C output order.
FAMILIES = ((0, 1, 2), (3, 4, 2), (0, 4, 5), (3, 1, 5), (3, 4, 5))


@dataclass(frozen=True)
class SpiralProgram:
    matrix: object
    rhs: tuple[Fraction, ...]
    output_sizes: tuple[int, ...]
    network_fingerprint: str
    behavior_fingerprint: str

    @property
    def sizes(self):
        return {"variables": self.matrix.shape[1], "equalities": self.matrix.shape[0],
                "nonzeros": self.matrix.nnz, "row_schema": ROW_SCHEMA}


def fingerprint(data):
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compile_spiral(network, behavior, *, max_variables=100_000, max_equalities=100_000,
                   max_nonzeros=500_000, max_scalar_bits=4096, max_scalar_chars=8192):
    """Compile Aq=b(p), q>=0 with five maximal expressible marginal blocks.

    The rows are pABC, pAC*pB, pAB*pC, pBC*pA, and pA*pB*pC, each in
    lexicographic A,B,C order. The first block normalizes the inflated joint
    table. A depends only on output sizes; b is polynomial in the input data.
    This classical copied-variable construction makes no quantum fanout claim.
    """
    validate_triangle(network)
    if not isinstance(behavior, RationalBehavior):
        raise TypeError("Expected an exactly validated RationalBehavior")
    if behavior.scenario.input_sizes != (1, 1, 1):
        raise ValueError("Spiral inflation currently requires three parties with no external inputs")
    budgets = (positive_budget(max_variables, "max_variables"),
               positive_budget(max_equalities, "max_equalities"),
               positive_budget(max_nonzeros, "max_nonzeros"))
    sizes = behavior.scenario.output_sizes
    joint = prod(sizes)
    counts = (joint**2, 5*joint, 5*joint**2)
    for size, limit, name in zip(counts, budgets, ("max_variables", "max_equalities", "max_nonzeros")):
        if size > limit:
            raise ValueError(f"Spiral inflation exceeds {name}: needs {size}, limit {limit}")
    validate_behavior_scalars(behavior, max_entries=joint, max_scalar_bits=max_scalar_bits,
                              max_scalar_chars=max_scalar_chars)
    profiles = tuple(product(*(range(n) for n in sizes)))
    marginals = [behavior.marginal((i,), (0, 0, 0)) for i in range(3)]
    ac = behavior.marginal((0, 2), (0, 0, 0))
    ab = behavior.marginal((0, 1), (0, 0, 0))
    bc = behavior.marginal((1, 2), (0, 0, 0))
    rhs = behavior.probabilities + tuple(ac[(a, c)]*marginals[1][(b,)] for a, b, c in profiles)
    rhs += tuple(ab[(a, b)]*marginals[2][(c,)] for a, b, c in profiles)
    rhs += tuple(bc[(b, c)]*marginals[0][(a,)] for a, b, c in profiles)
    rhs += tuple(prod((marginals[i][(a[i],)] for i in range(3)), start=Fraction(1)) for a in profiles)
    rows, columns = [], []
    for column, assignment in enumerate(product(*(range(n) for n in sizes + sizes))):
        for block, group in enumerate(FAMILIES):
            index = 0
            for party, node in enumerate(group):
                index = index*sizes[party] + assignment[node]
            rows.append(block*joint + index)
            columns.append(column)
    matrix = coo_matrix(([1]*counts[2], (rows, columns)), shape=(counts[1], counts[0]),
                        dtype="int8").tocsr()
    return SpiralProgram(matrix, rhs, sizes, fingerprint(network_to_dict(network)),
                          fingerprint(behavior_to_dict(behavior)))
