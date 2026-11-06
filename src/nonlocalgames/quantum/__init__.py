"""Finite-dimensional witnesses and exact symbolic CHSH identities."""

from .strategy import QuantumStrategy, chsh_strategy
from .sos import verify_chsh_sos
from .seesaw import SeesawResult, seesaw
from .references import ghz_strategy, magic_square_strategy

__all__ = ["QuantumStrategy", "chsh_strategy", "ghz_strategy", "magic_square_strategy",
           "verify_chsh_sos", "SeesawResult", "seesaw"]
