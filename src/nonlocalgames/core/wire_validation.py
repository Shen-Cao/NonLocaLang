"""Semantic scalar and count preflights shared by portable data readers."""
from .numeric_data import positive_budget
from .scalar_wire import parse_rational


def bounded_product(values, limit, name):
    limit = positive_budget(limit, name)
    result = 1
    for value in values:
        if value > limit // result:
            raise ValueError(f'Data expansion exceeds {name}')
        result *= value
    return result


def validate_game_numbers(game, *, max_scalar_bits=4096, max_scalar_chars=8192):
    """Bound supplied rational components before coefficient arithmetic or hashing."""
    from .game import Game
    if not isinstance(game, Game):
        raise TypeError('Expected Game')
    positive_budget(max_scalar_bits, 'max_scalar_bits')
    positive_budget(max_scalar_chars, 'max_scalar_chars')
    for values in (game.distribution, game.payoff):
        for value in values:
            parse_rational(value, max_bits=max_scalar_bits, max_chars=max_scalar_chars)
