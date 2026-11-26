"""Immutable data envelopes; valid identity metadata is not proof admission."""
from dataclasses import InitVar, dataclass
from enum import StrEnum
import hashlib
from numbers import Integral

from ..core.identity import (JSONBudgetExceeded, RESOURCE_ID_PREFIX, VERIFIER_ID_PREFIX,
                             canonical_json, verifier_id)
from ..core.json_file import decode_json

EVIDENCE_SCHEMA = "nonlocalgames.evidence.v1"
ARTIFACT_ID_PREFIX = EVIDENCE_SCHEMA + ":"
DEFAULT_MAX_BYTES = 8_000_000


class EvidenceKind(StrEnum):
    LOCAL_STRATEGY = "local_strategy"
    LOCAL_ENUMERATION = "local_enumeration"
    NONSIGNALLING_LP = "nonsignalling_lp"
    XOR_DUAL = "xor_dual"
    XOR_PRIMAL = "xor_primal"
    NUMERICAL_HINT = "numerical_hint"
    QUANTUM_STRATEGY = "quantum_strategy"


def _budget(max_bytes):
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, Integral) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    return int(max_bytes)


def _identifier(value, prefix):
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("Invalid versioned evidence identifier prefix")
    digest = value[len(prefix):]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Evidence identifiers require lowercase SHA256 digests")


def _resource_digest(resource_json):
    return RESOURCE_ID_PREFIX + hashlib.sha256(resource_json.encode("utf-8")).hexdigest()


def _artifact_digest(body):
    return ARTIFACT_ID_PREFIX + hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceArtifact:
    """Untrusted evidence data with immutable canonical JSON snapshots.

    Construction validates only the envelope and resource representation. The
    payload's asserted guarantee, value, or method is not authenticated here.
    Proof admission requires replay by the relevant verifier on the bound game.
    ``game_id`` has checked syntax; resolving it requires an external Game.
    """
    kind: EvidenceKind
    game_id: str
    resource_json: str
    payload_json: str
    max_bytes: InitVar[int] = DEFAULT_MAX_BYTES

    def __post_init__(self, max_bytes):
        limit = _budget(max_bytes)
        object.__setattr__(self, "kind", EvidenceKind(self.kind))
        _identifier(self.game_id, VERIFIER_ID_PREFIX)
        if not isinstance(self.resource_json, str) or not isinstance(self.payload_json, str):
            raise TypeError("Evidence resource_json and payload_json must be JSON strings")
        if len(self.resource_json.encode("utf-8")) + len(self.payload_json.encode("utf-8")) > limit:
            raise JSONBudgetExceeded("Evidence JSON exceeds max_bytes")
        raw_resource, raw_payload = decode_json(self.resource_json), decode_json(self.payload_json)
        resource_json = canonical_json(raw_resource, max_bytes=limit)
        payload_json = canonical_json(raw_payload, max_bytes=limit)
        object.__setattr__(self, "resource_json", resource_json)
        object.__setattr__(self, "payload_json", payload_json)
        body = self._body()
        canonical_json({**body, "artifact_id": _artifact_digest(body)}, max_bytes=limit)
        from ..core.resource_serialization import resource_from_dict, resource_to_dict
        resource = resource_from_dict(raw_resource)
        if canonical_json(resource_to_dict(resource)) != resource_json:
            raise ValueError("Evidence resource must use the canonical current resource codec")

    def _body(self):
        return {"schema": EVIDENCE_SCHEMA, "kind": self.kind.value, "game_id": self.game_id,
                "resource": decode_json(self.resource_json), "resource_id": self.resource_id,
                "payload": decode_json(self.payload_json)}

    @property
    def resource(self):
        from ..core.resource_serialization import resource_from_dict
        return resource_from_dict(decode_json(self.resource_json))

    @property
    def payload(self):
        return decode_json(self.payload_json)

    @property
    def resource_id(self):
        return _resource_digest(self.resource_json)

    @property
    def artifact_id(self):
        return _artifact_digest(self._body())

    @property
    def nbytes(self):
        body = self._body()
        return len(canonical_json({**body, "artifact_id": _artifact_digest(body)}).encode("utf-8"))


def make_evidence(game, resource, kind, payload, *, max_bytes=DEFAULT_MAX_BYTES):
    """Snapshot explicit JSON data; no candidate values are rationalized or certified."""
    from ..core.resources import ResourceSpec
    from ..core.resource_serialization import resource_to_dict
    limit = _budget(max_bytes)
    if not isinstance(resource, ResourceSpec):
        raise TypeError("Expected a ResourceSpec")
    return EvidenceArtifact(EvidenceKind(kind), verifier_id(game),
                            canonical_json(resource_to_dict(resource), max_bytes=limit),
                            canonical_json(payload, max_bytes=limit), max_bytes=limit)


def evidence_to_dict(artifact, *, max_bytes=DEFAULT_MAX_BYTES):
    if not isinstance(artifact, EvidenceArtifact):
        raise TypeError("Expected an EvidenceArtifact")
    limit = _budget(max_bytes)
    if len(artifact.resource_json.encode("utf-8")) + len(artifact.payload_json.encode("utf-8")) > limit:
        raise JSONBudgetExceeded("Evidence JSON exceeds max_bytes")
    body = artifact._body()
    result = {**body, "artifact_id": _artifact_digest(body)}
    canonical_json(result, max_bytes=limit)
    return result


def evidence_from_dict(data, *, max_bytes=DEFAULT_MAX_BYTES):
    """Check strict envelope fields and content IDs; the resulting artifact is still untrusted."""
    limit = _budget(max_bytes)
    canonical_json(data, max_bytes=limit)
    fields = {"schema", "kind", "game_id", "resource", "resource_id", "payload", "artifact_id"}
    if not isinstance(data, dict) or set(data) != fields or data["schema"] != EVIDENCE_SCHEMA:
        raise ValueError("Invalid evidence envelope fields or schema")
    _identifier(data["game_id"], VERIFIER_ID_PREFIX)
    _identifier(data["resource_id"], RESOURCE_ID_PREFIX)
    _identifier(data["artifact_id"], ARTIFACT_ID_PREFIX)
    resource_json = canonical_json(data["resource"], max_bytes=limit)
    if data["resource_id"] != _resource_digest(resource_json):
        raise ValueError("Evidence resource_id differs from the encoded resource")
    body = {key: value for key, value in data.items() if key != "artifact_id"}
    if data["artifact_id"] != _artifact_digest(body):
        raise ValueError("Evidence artifact_id differs from the envelope content")
    return EvidenceArtifact(EvidenceKind(data["kind"]), data["game_id"], resource_json,
                            canonical_json(data["payload"], max_bytes=limit), max_bytes=limit)
