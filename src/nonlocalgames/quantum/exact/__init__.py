"""Rational finite quantum witnesses with solver-independent feasibility checks."""
from .budgets import ExactBudget, ExactBudgetExceeded
from .matrix import ExactPSDViolation, RationalMatrix, realification, verify_psd_complex
from .strategy import ExactQuantumStrategy
from .validation import validate_game_scalars
from .serialization import strategy_from_dict, strategy_to_dict
from .references import exact_chsh_strategy, exact_ghz_strategy

__all__ = ['ExactBudget', 'ExactBudgetExceeded', 'ExactPSDViolation', 'RationalMatrix',
           'realification', 'verify_psd_complex', 'ExactQuantumStrategy', 'validate_game_scalars',
           'strategy_from_dict', 'strategy_to_dict', 'exact_chsh_strategy', 'exact_ghz_strategy']
