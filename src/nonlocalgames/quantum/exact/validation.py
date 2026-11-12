"""Exact state/POVM conditions and bounded game data for rational witnesses."""
from fractions import Fraction
from itertools import islice

from ...core.game import Game, Scenario
from .budgets import ExactBudgetExceeded, bounded_dimension, coerce_budget
from .matrix import RationalMatrix, verify_psd_complex
from .scalars import exact_scalar


def validate_scenario(scenario, budget):
    if not isinstance(scenario, Scenario):
        raise TypeError('Expected Scenario')
    if scenario.parties > budget.max_parties:
        raise ExactBudgetExceeded('Scenario exceeds max_parties')
    if scenario.entries > budget.max_events:
        raise ExactBudgetExceeded('Scenario exceeds max_events')


def validate_game_scalars(game, budget=None):
    """Check all original verifier scalars before hashing, then its exact coefficients.

    Zero-probability questions do not exempt stored payoff data from bit limits.
    The returned coefficients can be reused by exact scoring.
    """
    budget = coerce_budget(budget)
    if not isinstance(game, Game):
        raise TypeError('Expected Game')
    validate_scenario(game.scenario, budget)
    distribution = tuple(exact_scalar(value, max_bits=budget.max_scalar_bits) for value in game.distribution)
    payoff = tuple(exact_scalar(value, max_bits=budget.max_scalar_bits) for value in game.payoff)
    if any(p < 0 for p in distribution) or sum(distribution, Fraction(0)) != 1:
        raise ValueError('Exact verifier distribution must be normalized and nonnegative')
    width = len(payoff)//len(distribution)
    return tuple(exact_scalar(distribution[i//width]*reward, max_bits=budget.max_scalar_bits)
                 for i, reward in enumerate(payoff))


def collect_measurements(measurements, state, budget):
    """Freeze bounded matrices and infer their party-local dimensions."""
    collected, dimensions, entries = [], [], state.dimension**2
    joint_dimension = 1
    for party_index, party in enumerate(islice(iter(measurements), budget.max_parties+1)):
        if party_index == budget.max_parties:
            raise ExactBudgetExceeded('Strategy exceeds max_parties')
        local, dimension, outputs = [], None, None
        for povm in party:
            effects = []
            for effect in povm:
                if not isinstance(effect, RationalMatrix):
                    raise TypeError('Exact measurement effects must be RationalMatrix objects')
                effect.check_budget(budget)
                if dimension is None:
                    dimension = effect.dimension
                    joint_dimension *= dimension
                    if joint_dimension > budget.max_joint_dimension:
                        raise ExactBudgetExceeded('Tensor factors exceed max_joint_dimension')
                if effect.dimension != dimension:
                    raise ValueError('All effects of one party must have the same dimension')
                entries += dimension**2
                if entries > budget.max_matrix_entries:
                    raise ExactBudgetExceeded('State and POVMs exceed max_matrix_entries')
                effects.append(effect)
            if not effects:
                raise ValueError('Every POVM needs at least one outcome')
            if outputs is None:
                outputs = len(effects)
            if len(effects) != outputs:
                raise ValueError('A party must use a fixed output alphabet across inputs')
            local.append(tuple(effects))
        if not local:
            raise ValueError('Every party needs at least one input')
        collected.append(tuple(local))
        dimensions.append(dimension)
    if not collected:
        raise ValueError('At least one party is required')
    return tuple(collected), tuple(dimensions), entries, joint_dimension


def declared_dimensions(dimensions, actual, budget):
    if dimensions is None:
        return actual
    values = tuple(islice(iter(dimensions), budget.max_parties+1))
    if len(values) > budget.max_parties:
        raise ExactBudgetExceeded('Declared dimensions exceed max_parties')
    values = tuple(bounded_dimension(d, budget) for d in values)
    if values != actual:
        raise ValueError('Declared dimensions differ from the local measurement factors')
    return actual


def validate_strategy_data(state, measurements, dimensions, budget):
    """Verify trace, positive state/effects, and exact POVM completeness."""
    state.check_budget(budget)
    if state.trace(budget=budget) != (Fraction(1), Fraction(0)):
        raise ValueError('Exact state trace must be one')
    state_psd = verify_psd_complex(state, budget=budget)
    checked = {state: state_psd.rank}
    for dimension, party in zip(dimensions, measurements):
        for povm in party:
            for effect in povm:
                effect.check_budget(budget)
                if effect not in checked:
                    checked[effect] = verify_psd_complex(effect, budget=budget).rank
            for i in range(dimension):
                for j in range(dimension):
                    real, imag = Fraction(0), Fraction(0)
                    for effect in povm:
                        real = exact_scalar(real+effect.real[i][j], max_bits=budget.max_scalar_bits)
                        imag = exact_scalar(imag+effect.imag[i][j], max_bits=budget.max_scalar_bits)
                    if real != int(i == j) or imag != 0:
                        raise ValueError('Exact POVM effects must sum to identity')
    return {'state_rank': state_psd.rank//2, 'realification_state_rank': state_psd.rank,
            'distinct_positive_matrices': len(checked), 'certified': True}
