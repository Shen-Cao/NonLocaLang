"""Data for a finite verifier, one local strategy, and their exact transcript."""
from dataclasses import dataclass
from fractions import Fraction

from ...quantum.exact import ExactBudget, RationalMatrix


Scalar = int | Fraction | str


@dataclass(frozen=True, slots=True)
class Continuation:
    """Verifier delivers this classical question after the first answer.

    accept[z] is an acceptance effect on the verifier's *retained* reference,
    selected after the local final answer z. These effects need not form a POVM.
    """
    question: int
    accept: tuple[RationalMatrix, ...]


@dataclass(frozen=True, slots=True)
class PrivateCase:
    """The index of this private draw is not sent; only question and Q are sent."""
    probability: Scalar
    question: int
    state: RationalMatrix  # joint density in reference R, transmitted system Q order
    after_first: tuple[Continuation, ...]  # indexed by the first classical answer


@dataclass(frozen=True, slots=True)
class TwoRoundProtocol:
    """Trusted verifier: initial correlated preparation, adaptive question, final test."""
    reference_dimension: int
    message_dimension: int
    cases: tuple[PrivateCase, ...]


@dataclass(frozen=True, slots=True)
class KrausTerm:
    """A completely positive summand X -> weight * operator X operator†."""
    weight: Scalar
    operator: RationalMatrix  # Q to memory, same dimension in this dialect


@dataclass(frozen=True, slots=True)
class FirstMove:
    """Indexed only by the delivered classical question, never by the private case."""
    question: int
    outcomes: tuple[tuple[KrausTerm, ...], ...]  # one CP map per classical answer


@dataclass(frozen=True, slots=True)
class LocalHistory:
    """Precisely what the prover knows classically at its second move."""
    first_question: int
    first_answer: int
    second_question: int


@dataclass(frozen=True, slots=True)
class FinalMove:
    """One final POVM for all branches with this delivered local history."""
    history: LocalHistory
    effects: tuple[RationalMatrix, ...]  # POVM on the local conditioned memory


@dataclass(frozen=True, slots=True)
class LocalStrategy:
    """Finite declarative response tables for the one prover; no callbacks."""
    first: tuple[FirstMove, ...]
    final: tuple[FinalMove, ...]


@dataclass(frozen=True, slots=True)
class ReplayLimits:
    """Use exact.max_events for transcript leaves, max_matrix_entries for input slots.

    max_dense_work is a size guard in dense matrix work units, not a time limit.
    """
    exact: ExactBudget = ExactBudget(max_events=256)
    max_kraus_terms: int = 128
    max_dense_work: int = 2_000_000

    def __post_init__(self):
        if not isinstance(self.exact, ExactBudget):
            raise TypeError("exact must be an ExactBudget")
        for name in ("max_kraus_terms", "max_dense_work"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class Transcript:
    """A verifier-side leaf; probabilities are unconditional, including its prior."""
    private_case: int
    history: LocalHistory
    final_answer: int
    probability: Fraction
    accepted_mass: Fraction


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Exact acceptance probability of this one supplied strategy (not an optimum)."""
    acceptance_probability: Fraction
    transcripts: tuple[Transcript, ...]
