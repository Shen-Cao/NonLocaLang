"""Small benchmark verifiers; conventions are specified in each constructor."""
from fractions import Fraction

from .core import Game, Scenario


def chsh():
    """Uniform inputs x,y in {0,1}; accept iff a XOR b = x*y."""
    return Game.from_rule(Scenario((2, 2), (2, 2)),
                          lambda x, a: (a[0] ^ a[1]) == x[0] * x[1], name="CHSH")


def ghz():
    """Three players, even-parity inputs; output parity equals half the input sum."""
    scenario = Scenario((2, 2, 2), (2, 2, 2))
    return Game.from_rule(scenario, lambda x, a: sum(a) % 2 == (sum(x) // 2) % 2,
                          distribution={x: Fraction(1, 4) for x in scenario.inputs
                                        if sum(x) % 2 == 0}, name="GHZ")


def magic_square():
    """Alice returns an even row, Bob an odd column, and their intersection agrees."""
    rows = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
    cols = ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1))
    return Game.from_rule(Scenario((3, 3), (4, 4)),
                          lambda x, a: rows[a[0]][x[1]] == cols[a[1]][x[0]],
                          name="Magic square")


def gyni():
    """Three-party guess-your-neighbour input game with uniform even-parity promise."""
    scenario = Scenario((2, 2, 2), (2, 2, 2))
    return Game.from_rule(scenario, lambda x, a: a == (x[1], x[2], x[0]),
                          distribution={x: Fraction(1, 4) for x in scenario.inputs
                                        if sum(x) % 2 == 0}, name="GYNI")


def biased_chsh(p):
    """Weight input 11 by p, sharing the remaining probability among 00,01,10."""
    from .core.game import rational
    from .core.scalar_wire import scalar_string
    p = rational(p)
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0,1]")
    return Game.from_rule(Scenario((2, 2), (2, 2)),
                          lambda x, a: (a[0] ^ a[1]) == x[0] * x[1],
                          distribution={(0, 0): (1-p)/3, (0, 1): (1-p)/3,
                                        (1, 0): (1-p)/3, (1, 1): p},
                          name=f"Biased CHSH ({scalar_string(p)})")


GAMES = {"chsh": chsh, "ghz": ghz, "magic-square": magic_square, "gyni": gyni}
