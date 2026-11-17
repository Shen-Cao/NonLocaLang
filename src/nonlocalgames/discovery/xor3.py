"""Uniform binary XOR predicates with three questions per party and their local orbits."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice, permutations, product
from numbers import Integral

from ..core.game import Game, Scenario
from ..equivalence.maps import Relabelling

XOR3_FAMILY = "nonlocalgames.uniform_xor3.v1"
XOR3_ACTION = "nonlocalgames.xor3_local_bijections.v1"
_SCENARIO = Scenario((3, 3), (2, 2))
_UNIFORM = (Fraction(1, 9),) * 9
_QUESTIONS = tuple(permutations(range(3)))
_PARTIES = ((0, 1), (1, 0))
_CANDIDATES = len(_PARTIES) * len(_QUESTIONS)**2


def xor3_game(mask: int) -> Game:
    """Build the uniform game. Bit 8 is the winning parity at question (0,0)."""
    if isinstance(mask, bool) or not isinstance(mask, Integral) or not 0 <= mask < 512:
        raise ValueError("XOR3 mask must be an integer between 0 and 511")
    mask = int(mask)
    payoff = []
    for shift in reversed(range(9)):
        odd = (mask >> shift) & 1
        payoff.extend((1-odd, odd, odd, 1-odd))
    return Game(_SCENARIO, _UNIFORM, tuple(payoff), name="Uniform 3x3 XOR")


def xor3_mask(game: Game) -> int:
    """Recover the row-major mask from a verifier in this precise family."""
    if not isinstance(game, Game):
        raise TypeError("Expected a Game")
    if game.scenario != _SCENARIO or game.distribution != _UNIFORM:
        raise ValueError("XOR3 requires three binary-output questions per party with uniform sampling")
    mask = 0
    for start in range(0, 36, 4):
        block = game.payoff[start:start+4]
        if block == (1, 0, 0, 1):
            bit = 0
        elif block == (0, 1, 1, 0):
            bit = 1
        else:
            raise ValueError("Exactly one answer parity must win at every XOR3 question pair")
        mask = 2*mask + bit
    return mask


@dataclass(frozen=True)
class Xor3Canonicalization:
    """The transporter maps the supplied verifier to its representative."""
    complete: bool
    representative: int | None
    orbit_id: str | None
    transporter: Relabelling | None
    examined: int


def _normalize_flips(table, rows, cols):
    """Return the least mask for these input maps and a choice of output flips."""
    row_flips = tuple(table[r][cols[0]] ^ table[rows[0]][cols[0]] for r in rows)
    col_flips = tuple(table[rows[0]][c] for c in cols)
    mask = 0
    for r, rflip in zip(rows, row_flips):
        for c, cflip in zip(cols, col_flips):
            mask = 2*mask + (table[r][c] ^ rflip ^ cflip)
    return mask, (row_flips, col_flips)


def canonicalize_xor3(game: Game, *, max_candidates: int = _CANDIDATES) -> Xor3Canonicalization:
    """Minimize the row-major winning-parity mask over local bijections and party swap.

    With the question and party order fixed, local answer flips add one bit per
    row and column modulo two. The least mask has first row zero, achievable by
    column flips. Each later row can then independently obtain a leading zero.
    These conditions determine the entire flipped table. Fixing the first row's
    flip at zero resolves the twofold redundancy in the flip maps.

    It suffices to examine the 72 possible party and question orders. Equal
    minima retain the first order in lexicographic (party, row, column) order.
    A truncated search returns only its completion flag and examined count.
    """
    if (isinstance(max_candidates, bool) or not isinstance(max_candidates, Integral)
            or max_candidates < 0):
        raise ValueError("max_candidates must be a nonnegative integer")
    mask = xor3_mask(game)
    grid = tuple(tuple((mask >> (8-3*i-j)) & 1 for j in range(3)) for i in range(3))
    transpose = tuple(zip(*grid))
    best = None
    examined = min(int(max_candidates), _CANDIDATES)
    for parties, rows, cols in islice(product(_PARTIES, _QUESTIONS, _QUESTIONS), examined):
        table = grid if parties == (0, 1) else transpose
        candidate, flips = _normalize_flips(table, rows, cols)
        if best is None or candidate < best:
            best = candidate
            best_parties, best_inputs, best_flips = parties, (rows, cols), flips
    if examined < _CANDIDATES:
        return Xor3Canonicalization(False, None, None, None, examined)
    output_maps = tuple(tuple((flip, 1-flip) for flip in axis) for axis in best_flips)
    transporter = Relabelling(_SCENARIO, best_parties, best_inputs, output_maps)
    orbit_id = f"{XOR3_FAMILY}/{XOR3_ACTION}/{best:09b}"
    return Xor3Canonicalization(True, best, orbit_id, transporter, examined)
