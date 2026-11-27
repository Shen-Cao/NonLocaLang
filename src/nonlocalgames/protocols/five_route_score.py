"""One fixed linear statistic of the five-route target, not a general model test."""
from dataclasses import dataclass
from fractions import Fraction
from math import prod

from ..core.numeric_data import positive_budget
from .fermion_reference import reference_arena
from .quadratic import Quadratic, scalar_to_pair
from .quadratic_behavior import QuadraticWiringTarget, _bound, _sum
from .quadratic_wire import target_id


# Sender at each receiver (A, B, C); not the destination at each source.
_ROUTES = (("ABC", (2, 0, 1)), ("ACB", (1, 2, 0)), ("AB", (1, 0, 2)),
           ("AC", (2, 1, 0)), ("BC", (0, 2, 1)))
_SIGNS = (0, 1, -1, 0)
_SWAP_BOUND = Fraction(3, 5)


@dataclass(frozen=True, slots=True)
class FiveRouteScore:
    """Unconditioned, fixed-weight contrasts in route order; no inferred statistics."""

    target_id: str
    abc: Quadratic
    acb: Quadratic
    ab: Quadratic
    ac: Quadratic
    bc: Quadratic

    @property
    def score(self):
        return (self.abc + self.acb - self.ab - self.ac - self.bc) / 5

    def to_dict(self):
        """Bind the score to the target; each strict comparison names its own model."""
        s = self.score
        # The phase cap is the positive root of 4s²-2s-1. The other root is
        # negative: squaring without the sign check can reverse the conclusion.
        exceeds_phase = s.sign() > 0 and (4*s*s - 2*s - 1).sign() > 0
        return {
            "schema": "nonlocalgames.five_route_score.v1",
            "target_id": self.target_id,
            "contrasts": {name: scalar_to_pair(getattr(self, name.lower())) for name, _ in _ROUTES},
            "score": scalar_to_pair(s),
            "comparisons": {
                "fixed_local_sign_swaps": {
                    "statistics": ["ordinary_tensor", "fermionic_even_pair"],
                    "sources": "Three independent balanced zero-phase reference sources.",
                    "readouts": "Fixed reference projectors; only 1+/1- swaps indexed by "
                                "(receiver, incoming sender, question).",
                    "upper_bound": scalar_to_pair(Quadratic(_SWAP_BOUND)),
                    "exceeds_upper_bound": (s - _SWAP_BOUND).sign() > 0,
                },
                "one_token_equatorial_phases": {
                    "statistics": ["ordinary_tensor", "fermionic_even_pair"],
                    "sources": "Three independent logical one-token states on kept/sent, "
                               "with no extra systems.",
                    "readouts": "Fixed token sectors and reference measurement shapes; only the "
                                "equatorial X phase and 1+/1- label swap may depend on "
                                "(receiver, incoming sender, question).",
                    # Not a Q(sqrt(2)) scalar pair and not part of the target codec.
                    "upper_bound_expression": "(1+sqrt(5))/4",
                    "exceeds_upper_bound": exceeds_phase,
                },
            },
        }


def _contrast(branch, route, bits):
    active = tuple(i for i, sender in enumerate(route) if i != sender)
    # Both settings of Alice contribute when active. Idle inputs are zero;
    # their outputs are marginalized, even when the idle label is 0 or 2 tokens.
    weight = (Quadratic(2) if 0 not in active else
              Quadratic(0, 2 if len(active) == 3 else 1))
    alice_questions = (0, 1) if 0 in active else (0,)
    terms = []
    for x in alice_questions:
        questions = (x, int(1 in active), int(2 in active))
        for labels, probability in branch.marginal(active, questions).items():
            sign = prod(_SIGNS[a] for a in labels)
            if sign:
                terms.append(weight * (probability if sign > 0 else -probability))
    return _sum(terms, bits)


def five_route_score(target: QuadraticWiringTarget, *, max_scalar_bits=4096) -> FiveRouteScore:
    """Score the validated reference arena, irrespective of how the tables were produced.

    No empirical sector probabilities enter the weights. The extra scalar budget
    bounds the running contrast coefficients and the returned score coefficients.
    """
    if not isinstance(target, QuadraticWiringTarget):
        raise TypeError("Expected a quadratic wiring target")
    if target.arena != reference_arena():
        raise ValueError("The score requires the reference five routes and question/answer alphabets")
    bits = positive_budget(max_scalar_bits, "max_scalar_bits")
    branches = dict(zip(target.arena.routes, target.behaviors, strict=True))
    contrasts = tuple(_contrast(branches[route], route, bits) for _, route in _ROUTES)
    result = FiveRouteScore(target_id(target), *contrasts)
    value = result.score
    _bound(bits, value.a, value.b)
    return result
