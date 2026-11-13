"""Exact verifier, behavior, and deterministic-witness transport through checked maps."""
from itertools import chain
from math import prod

from ..core.game import Game
from .maps import Relabelling, RelabellingBudgetExceeded, _positive_budget


def _matching(source, relabelling):
    if not isinstance(relabelling, Relabelling):
        raise TypeError('Expected a checked Relabelling')
    if source != relabelling.source:
        raise ValueError('Data interface differs from the relabelling source')


def _table_budget(scenario, max_entries):
    limit = _positive_budget(max_entries, 'max_entries')
    entries = 1
    for size in chain(scenario.input_sizes, scenario.output_sizes):
        if size > limit//entries:
            raise RelabellingBudgetExceeded('Relabelled table exceeds max_entries')
        entries *= size
    return limit


def relabel_game(game, relabelling, *, max_entries=1_000_000, name=None):
    """Transport mu and V separately, including unused questions and signed rewards.

    This changes verifier coordinates only. It does not authorize reusing a
    resource-specific result without transporting that resource specification.
    """
    if not isinstance(game, Game):
        raise TypeError('Expected a finite Game')
    _matching(game.scenario, relabelling)
    _table_budget(game.scenario, max_entries)
    if name is not None and not isinstance(name, str):
        raise TypeError('Game name override must be a string')
    distribution, payoff = [], []
    outputs = relabelling.target.outputs
    for new_input in relabelling.target.inputs:
        old_input = relabelling.source_input(new_input)
        distribution.append(game.probability(old_input))
        payoff.extend(game.reward(old_input, relabelling.source_output(new_input, new_output))
                      for new_output in outputs)
    return Game(relabelling.target, distribution, payoff, game.name if name is None else name)


def relabel_behavior(behavior, relabelling, *, max_entries=1_000_000):
    """Transport every conditional row and recheck exact normalization/nonsignalling."""
    from ..networks.behavior import RationalBehavior
    if not isinstance(behavior, RationalBehavior):
        raise TypeError('Expected a RationalBehavior')
    _matching(behavior.scenario, relabelling)
    limit = _table_budget(behavior.scenario, max_entries)
    width = prod(behavior.scenario.output_sizes)
    outputs = relabelling.target.outputs
    probabilities = []
    for new_input in relabelling.target.inputs:
        old_input = relabelling.source_input(new_input)
        row = behavior.scenario.input_index(old_input)*width
        probabilities.extend(behavior.probabilities[row + behavior.scenario.output_index(
            relabelling.source_output(new_input, new_output))] for new_output in outputs)
    return RationalBehavior(relabelling.target, probabilities, max_entries=limit)


def relabel_deterministic_strategy(strategy, relabelling):
    """Transport local responses by inverse output labels at the new local input."""
    from ..classical.strategy import DeterministicStrategy
    if not isinstance(strategy, DeterministicStrategy):
        raise TypeError('Expected a DeterministicStrategy')
    _matching(strategy.scenario, relabelling)
    responses = tuple(tuple(relabelling.output_maps[party][question].index(
        strategy.responses[old][relabelling.input_maps[party][question]])
        for question in range(relabelling.target.input_sizes[party]))
        for party, old in enumerate(relabelling.party_permutation))
    return DeterministicStrategy(relabelling.target, responses)
