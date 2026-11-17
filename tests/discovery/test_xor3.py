from collections import Counter
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames.discovery.xor3 import (XOR3_ACTION, XOR3_FAMILY, canonicalize_xor3,
                                           xor3_game, xor3_mask)
from nonlocalgames.equivalence import Relabelling, relabel_game
from nonlocalgames.games import chsh


def _neighbors(grid):
    # Transposition also conjugates each row operation into its column analogue.
    yield tuple(zip(*grid))
    for i in range(2):
        rows = list(grid)
        rows[i], rows[i+1] = rows[i+1], rows[i]
        yield tuple(rows)
    yield (tuple(1-v for v in grid[0]), *grid[1:])


def _orbits_by_generators():
    """Traverse connected components with elementary moves and no gauge fixing."""
    remaining = set(range(512))
    while remaining:
        smallest = min(remaining)
        bits = tuple(map(int, f"{smallest:09b}"))
        grid = (bits[:3], bits[3:6], bits[6:])
        orbit, pending = {grid}, [grid]
        while pending:
            for neighbor in _neighbors(pending.pop()):
                if neighbor not in orbit:
                    orbit.add(neighbor)
                    pending.append(neighbor)
        masks = {int("".join(str(v) for row in g for v in row), 2) for g in orbit}
        assert masks <= remaining
        remaining -= masks
        yield smallest, masks


def test_all_masks_match_independent_orbits_and_exact_verifier_transport():
    sizes = Counter()
    for representative, members in _orbits_by_generators():
        canonical = xor3_game(representative)
        for mask in members:
            source = xor3_game(mask)
            assert xor3_mask(source) == mask
            result = canonicalize_xor3(source)
            assert result.complete and result.examined == 72
            assert result.representative == representative
            assert result.orbit_id == f"{XOR3_FAMILY}/{XOR3_ACTION}/{representative:09b}"
            assert relabel_game(source, result.transporter) == canonical
            assert relabel_game(canonical, result.transporter.inverse()) == source
            sizes[representative] += 1
        assert canonicalize_xor3(canonical).transporter == Relabelling(canonical.scenario)
    assert sizes == {0: 32, 1: 288, 10: 192}


def test_binary_encoding_places_most_significant_bit_at_first_question():
    mask = np.int64(256)
    game = xor3_game(mask)
    assert game.probability((2, 1)) == Fraction(1, 9)
    for x in game.inputs:
        for a in game.outputs:
            assert game.reward(x, a) == ((a[0] ^ a[1]) == int(x == (0, 0)))
    assert xor3_mask(replace(game, name="Alternative label")) == mask


@pytest.mark.parametrize("invalid", [True, -1, 512, "001100011", 3.0])
def test_membership_rejects_invalid_mask(invalid):
    with pytest.raises(ValueError, match="mask"):
        xor3_game(invalid)


def test_membership_requires_full_interface_uniform_sampler_and_boolean_xor():
    g = xor3_game(0)
    invalid = [chsh(), replace(g, distribution=(1,)+(0,)*8)]
    for block in ((1, 1, 1, 1), (0, 0, 0, 0), (0, 1, 0, 1),
                  (Fraction(1, 2), 0, 0, Fraction(1, 2)), (2, -1, -1, 2)):
        invalid.append(replace(g, payoff=g.payoff[:-4]+block))
    for game in invalid:
        with pytest.raises(ValueError):
            canonicalize_xor3(game)


@pytest.mark.parametrize("limit", [0, 1, 71])
def test_incomplete_search_consumes_only_budgeted_orders_and_issues_no_identity(monkeypatch, limit):
    import nonlocalgames.discovery.xor3 as module
    normalize = module._normalize_flips
    calls = 0

    def counted(*args):
        nonlocal calls
        calls += 1
        assert calls <= limit
        return normalize(*args)

    monkeypatch.setattr(module, '_normalize_flips', counted)
    result = canonicalize_xor3(xor3_game(0), max_candidates=limit)
    assert calls == result.examined == limit
    assert result.complete is False
    assert (result.representative, result.orbit_id, result.transporter) == (None, None, None)


def test_budget_validation_and_large_explicit_limit():
    game = xor3_game(23)
    for limit in (False, -1, 4.5):
        with pytest.raises(ValueError, match="max_candidates"):
            canonicalize_xor3(game, max_candidates=limit)
    assert canonicalize_xor3(game, max_candidates=10**100) == canonicalize_xor3(game)
