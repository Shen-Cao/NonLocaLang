"""Finite-dimensional witnesses and exact symbolic CHSH identities."""

from .strategy import QuantumStrategy, chsh_strategy
from .sos import verify_chsh_sos
from .seesaw import SeesawResult, seesaw
from .references import ghz_strategy, magic_square_strategy
from .states import FixedState, partial_trace, partial_transpose, state_diagnostics, product_state_residual
from .restricted import StateOptimizationResult, optimize_state

__all__ = ["QuantumStrategy", "chsh_strategy", "ghz_strategy", "magic_square_strategy",
           "verify_chsh_sos", "SeesawResult", "seesaw", "FixedState", "partial_trace",
           "partial_transpose", "state_diagnostics", "product_state_residual",
           "StateOptimizationResult", "optimize_state"]
