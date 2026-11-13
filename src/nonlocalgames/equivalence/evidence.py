"""Transport supported raw evidence by constructing and checking fresh certificates."""
from dataclasses import asdict, dataclass

from ..claims import EvidenceArtifact, EvidenceKind, VerificationBudget, make_evidence, verify_artifact
from ..core.game import Game
from ..core.identity import canonical_json
from ..core.scalar_wire import parse_rational, scalar_string
from .finite import _matching, _table_budget, relabel_deterministic_strategy, relabel_game
from .maps import Relabelling
from .serialization import relabelling_id, relabelling_to_dict


@dataclass(frozen=True)
class EvidenceTransportReceipt:
    """Provenance only; public claim assessment still requires raw evidence."""
    relabelling_json: str
    relabelling_id: str
    source_game_id: str
    target_game_id: str
    source_resource_id: str
    target_resource_id: str
    source_evidence_id: str
    target_evidence_id: str
    rule: str = 'local_bijection_and_fresh_certificate_check.v1'


@dataclass(frozen=True)
class EvidenceTransport:
    artifact: EvidenceArtifact
    receipt: EvidenceTransportReceipt


def transport_receipt_to_dict(receipt):
    if not isinstance(receipt, EvidenceTransportReceipt):
        raise TypeError('Expected EvidenceTransportReceipt')
    result = {'schema': 'nonlocalgames.evidence_transport_receipt.v1', **asdict(receipt)}
    canonical_json(result)
    return result


def _local_payload(game, artifact, relabelling, check):
    from ..classical import DeterministicStrategy
    source = DeterministicStrategy(game.scenario, artifact.payload['strategy'])
    target = relabel_deterministic_strategy(source, relabelling)
    payload = {'strategy': target.responses}
    if artifact.kind == EvidenceKind.LOCAL_ENUMERATION:
        payload['value'] = scalar_string(check.lower)
    return payload


def _quantum_payload(game, target, artifact, relabelling, budget):
    from ..claims.verify import _quantum_policy
    from ..certificates.quantum import certify_quantum_strategy
    from ..quantum.exact import strategy_from_dict
    from .quantum import relabel_exact_strategy
    policy = _quantum_policy(game, budget, artifact.artifact_id)
    source = strategy_from_dict(artifact.payload['strategy'], budget=policy)
    strategy = relabel_exact_strategy(source, relabelling, budget=policy)
    return certify_quantum_strategy(target, strategy, budget=policy,
                                     max_bytes=budget.max_artifact_bytes).witness


def _xor_payload(target, artifact, relabelling, budget):
    from ..certificates.xor import verify_xor_dual
    diagonal = tuple(parse_rational(value, max_bits=budget.max_wire_scalar_bits,
                                    max_chars=budget.max_wire_scalar_chars)
                     for value in artifact.payload['diagonal'])
    offsets = (0, relabelling.source.input_sizes[0])
    transported = tuple(diagonal[offsets[old]+question]
                        for old, questions in zip(relabelling.party_permutation, relabelling.input_maps)
                        for question in questions)
    return verify_xor_dual(target, transported, max_matrix_size=budget.max_matrix_size,
                           max_scalar_bits=budget.max_wire_scalar_bits,
                           max_scalar_chars=budget.max_wire_scalar_chars).witness


def relabel_evidence(game, artifact, relabelling, *, budget=None, max_entries=1_000_000):
    """Recheck a source, construct its transported witness, then recheck a new artifact.

    Local enumeration is explicitly replayed on both verifiers. No optimizer is
    called for deterministic, exact quantum, or XOR-dual transport. NS equality
    duals and opaque numerical hints are unsupported; no fingerprint-only rewrite
    or generic proof-receipt inference is performed.
    """
    if not isinstance(relabelling, Relabelling):
        raise TypeError('Expected Relabelling')
    if not isinstance(artifact, EvidenceArtifact):
        raise TypeError('Transport requires a raw EvidenceArtifact')
    budget = VerificationBudget() if budget is None else budget
    if not isinstance(budget, VerificationBudget):
        raise TypeError('Expected VerificationBudget')
    supported = {EvidenceKind.LOCAL_STRATEGY, EvidenceKind.LOCAL_ENUMERATION,
                 EvidenceKind.QUANTUM_STRATEGY, EvidenceKind.XOR_DUAL}
    if artifact.kind not in supported:
        raise NotImplementedError(f'Raw {artifact.kind.value} evidence transport is unsupported')
    if not isinstance(game, Game):
        raise TypeError('Expected a finite Game')
    _matching(game.scenario, relabelling)
    _table_budget(game.scenario, max_entries)
    source_check = verify_artifact(game, artifact, budget=budget)
    target = relabel_game(game, relabelling, max_entries=max_entries)
    if artifact.kind in (EvidenceKind.LOCAL_STRATEGY, EvidenceKind.LOCAL_ENUMERATION):
        payload = _local_payload(game, artifact, relabelling, source_check)
    elif artifact.kind == EvidenceKind.QUANTUM_STRATEGY:
        payload = _quantum_payload(game, target, artifact, relabelling, budget)
    else:
        payload = _xor_payload(target, artifact, relabelling, budget)
    # Admission above restricts these proof kinds to standard unrestricted resources.
    from ..core.resources import ResourceSpec
    result = make_evidence(target, ResourceSpec(source_check.resource), artifact.kind, payload,
                            max_bytes=budget.max_artifact_bytes)
    target_check = verify_artifact(target, result, budget=budget)
    for name in ('lower', 'upper', 'attained_score'):
        if getattr(source_check, name) != getattr(target_check, name):
            raise RuntimeError('Fresh evidence check changed a relabelled score or bound')
    if source_check.actual_dimensions is not None:
        expected = tuple(source_check.actual_dimensions[i] for i in relabelling.party_permutation)
        if target_check.actual_dimensions != expected:
            raise RuntimeError('Fresh evidence check changed transported physical dimensions')
    receipt = EvidenceTransportReceipt(
        canonical_json(relabelling_to_dict(relabelling)), relabelling_id(relabelling),
        artifact.game_id, result.game_id, artifact.resource_id, result.resource_id,
        artifact.artifact_id, result.artifact_id)
    return EvidenceTransport(result, receipt)
