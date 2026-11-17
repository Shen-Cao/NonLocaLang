"""Generate and independently replay all 512 uniform 3×3 XOR gap claims."""
from collections import Counter

from ..claims import GapClaim, assess_gap, evidence_from_dict, evidence_to_dict
from ..core.identity import canonical_json, verifier_id
from ..core.resources import ResourceSpec
from ..core.scalar_wire import scalar_string
from ..equivalence import relabel_evidence, relabel_game, relabelling_to_dict
from ._xor3_archive import read_archive, write_archive
from .xor3 import XOR3_ACTION, XOR3_FAMILY, canonicalize_xor3, xor3_game
from .xor3_evidence import representative_evidence

_ORBIT_SIZES = {0: 32, 1: 288, 10: 192}
_GAP = GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local'))
_HEADER = {'schema': 'nonlocalgames.xor3_corpus.v1', 'family': XOR3_FAMILY, 'action': XOR3_ACTION,
           'query': {'kind': 'gap_greater', 'higher': 'quantum_finite', 'lower': 'local',
                     'threshold': '0'}}


def _checked_member(mask, representatives):
    game = xor3_game(mask)
    canonical = canonicalize_xor3(game)
    if not canonical.complete or canonical.representative not in representatives:
        raise ValueError(f'Incomplete or unexpected canonicalization at mask {mask}')
    rep = canonical.representative
    rep_game = xor3_game(rep)
    mapping = canonical.transporter.inverse()
    if relabel_game(rep_game, mapping) != game:
        raise ValueError(f'Transporter does not recover mask {mask}')
    artifacts = tuple(relabel_evidence(rep_game, artifact, mapping).artifact
                      for artifact in representatives[rep])
    result = assess_gap(game, _GAP, artifacts)
    expected_truth = 'disproved' if rep == 0 else 'proved'
    if result.execution != 'completed' or result.truth != expected_truth:
        raise ValueError(f'Gap verification failed at mask {mask}: {result.execution}, {result.truth}')
    quantum, local = result.intervals
    if local.lower is None or local.lower != local.upper:
        raise ValueError(f'Missing exact local value at mask {mask}')
    return {'mask': mask, 'verifier_id': result.game_id, 'representative': rep,
            'orbit_id': canonical.orbit_id, 'to_representative': relabelling_to_dict(canonical.transporter),
            'target_evidence_ids': [a.artifact_id for a in artifacts], 'claim_id': result.claim_id,
            'truth': result.truth.value, 'local_value': scalar_string(local.lower),
            'quantum_lower': scalar_string(quantum.lower),
            'quantum_upper': None if quantum.upper is None else scalar_string(quantum.upper)}


def _summary(members):
    sizes = Counter(row['representative'] for row in members)
    verdicts = Counter(row['truth'] for row in members)
    if (sizes != _ORBIT_SIZES or verdicts != {'proved': 480, 'disproved': 32}
            or len({row['verifier_id'] for row in members}) != 512):
        raise ValueError('XOR3 orbit census or verdict counts failed verification')
    return {'schema': 'nonlocalgames.xor3_corpus_summary.v1', 'family': XOR3_FAMILY,
            'action': XOR3_ACTION, 'members': len(members),
            'orbit_sizes': {f'{mask:09b}': sizes[mask] for mask in _ORBIT_SIZES},
            'verdicts': dict(verdicts)}


def write_xor3_corpus(path):
    """Generate representative evidence, verify every member, and atomically write the archive."""
    representatives = {mask: representative_evidence(mask) for mask in _ORBIT_SIZES}
    members = [_checked_member(mask, representatives) for mask in range(512)]
    summary = _summary(members)
    records = [{'mask': mask, 'evidence': [evidence_to_dict(a) for a in artifacts]}
               for mask, artifacts in representatives.items()]
    write_archive(path, {**_HEADER, 'representatives': records, 'members': members})
    return summary


def _check_coverage(rows, masks, name):
    if not isinstance(rows, list) or len(rows) != len(masks):
        raise ValueError(f'XOR3 {name} coverage is incomplete or duplicated')
    for row, mask in zip(rows, masks):
        if not isinstance(row, dict) or type(row.get('mask')) is not int or row['mask'] != mask:
            raise ValueError(f'XOR3 {name} must cover each mask once in ascending order')


def _decode_representatives(rows):
    result = {}
    for row in rows:
        mask = row['mask']
        if set(row) != {'mask', 'evidence'} or not isinstance(row['evidence'], list) or len(row['evidence']) != 2:
            raise ValueError(f'Invalid representative evidence bundle for mask {mask}')
        artifacts = tuple(evidence_from_dict(item) for item in row['evidence'])
        required = ('local_enumeration', 'xor_dual' if mask == 0 else 'quantum_strategy')
        if tuple(a.kind.value for a in artifacts) != required:
            raise ValueError(f'Unexpected representative evidence kinds for mask {mask}')
        game_id = verifier_id(xor3_game(mask))
        if any(a.game_id != game_id for a in artifacts):
            raise ValueError(f'Wrong representative evidence verifier at mask {mask}')
        result[mask] = artifacts
    return result


def replay_xor3_corpus(path, *, max_file_bytes=8_000_000):
    """Rebuild all verifiers and recheck archived raw proofs against each target claim."""
    archive = read_archive(path, max_file_bytes)
    if (not isinstance(archive, dict) or set(archive) != {*_HEADER, 'representatives', 'members'}
            or any(archive[key] != value for key, value in _HEADER.items())):
        raise ValueError('Invalid XOR3 archive fields, family, action or query')
    _check_coverage(archive['representatives'], _ORBIT_SIZES, 'representatives')
    _check_coverage(archive['members'], range(512), 'members')
    representatives = _decode_representatives(archive['representatives'])
    checked = []
    for mask, supplied in enumerate(archive['members']):
        member = _checked_member(mask, representatives)
        if canonical_json(member) != canonical_json(supplied):
            raise ValueError(f'XOR3 member record differs from the recomputed result at mask {mask}')
        checked.append(member)
    return _summary(checked)
