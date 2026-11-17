"""Resumable weighted CHSH discovery with independently replayable member claims."""
from collections import Counter
from numbers import Integral
from pathlib import Path

from ._checkpoint import MANIFEST_BYTES, MEMBER_BYTES, load_checkpoint, prefix_size, publish
from ._weighted_record import generate_member, verify_member
from .run_manifest import create_manifest, replay_budget, validate_manifest


def _load_manifest(directory):
    return validate_manifest(load_checkpoint(directory / 'manifest.json', MANIFEST_BYTES))


def _prepare_directory(directory, desired):
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / 'manifest.json'
    members = directory / 'members'
    if not header.exists():
        if prefix_size(members, desired['steps']) and not header.exists():
            raise ValueError('Member checkpoints exist without their manifest')
        publish(header, desired, MANIFEST_BYTES)
    manifest = _load_manifest(directory)
    if manifest['run_id'] != desired['run_id']:
        raise ValueError('Run manifest differs from the requested settings or producer provenance')
    members.mkdir(exist_ok=True)
    return manifest


def _summary(manifest, reviewed, include_members=False):
    expected = manifest['steps'] + 1
    verdicts = Counter(row['truth'] for row in reviewed)
    summary = {
        'schema': 'nonlocalgames.weighted_chsh_summary.v1', 'run_id': manifest['run_id'],
        'expected': expected, 'recorded': len(reviewed),
        'pending': list(range(len(reviewed), expected)), 'coverage_complete': len(reviewed) == expected,
        'verdicts': {status: verdicts[status] for status in ('proved', 'disproved', 'unknown')},
        'execution': dict(Counter(row['execution'] for row in reviewed)),
        'screen_statuses': dict(Counter(row['screen_status'] for row in reviewed)),
        'generation_failures': sum(row['generation_error'] is not None for row in reviewed),
    }
    if include_members:
        summary['results'] = reviewed
    return summary


def run_weighted_chsh(directory, *, steps=20, grid_denominator=16, screen=True, solver='SCS',
                      eps=1e-8, verification_budget=None, max_new=None):
    """Verify an existing prefix and publish up to max_new additional member checkpoints.

    Concurrent writers may repeat discovery for the same parameter. The first
    complete checkpoint wins. Other writers verify and use that saved record.
    A local filesystem supporting atomic hard-link publication is required.
    """
    if max_new is not None and (isinstance(max_new, bool) or not isinstance(max_new, Integral) or max_new < 0):
        raise ValueError('max_new must be a nonnegative integer')
    desired = create_manifest(steps=steps, grid_denominator=grid_denominator, screen=screen,
                              solver=solver, eps=eps, verification_budget=verification_budget)
    directory = Path(directory)
    manifest = _prepare_directory(directory, desired)
    budget = replay_budget(manifest, verification_budget)
    members = directory / 'members'
    existing = prefix_size(members, manifest['steps'])
    reviewed, added = [], 0
    for k in range(manifest['steps'] + 1):
        path = members / f'{k}.json'
        if k < existing or path.exists():
            record = load_checkpoint(path, MEMBER_BYTES)
        elif max_new is not None and added >= max_new:
            break
        else:
            record = generate_member(manifest, k, budget)
            computed = verify_member(manifest, k, record, budget)
            if publish(path, record, MEMBER_BYTES):
                reviewed.append(computed)
                added += 1
                continue
            record = load_checkpoint(path, MEMBER_BYTES)
        reviewed.append(verify_member(manifest, k, record, budget))
    return _summary(manifest, reviewed)


def replay_weighted_chsh(directory, *, budget=None, include_members=False):
    """Replay the saved prefix and report the remaining grid indices using local verification limits."""
    if not isinstance(include_members, bool):
        raise TypeError('include_members must be a Boolean')
    directory = Path(directory)
    manifest = _load_manifest(directory)
    limit = replay_budget(manifest, budget)
    members = directory / 'members'
    prefix = prefix_size(members, manifest['steps'])
    reviewed = [verify_member(manifest, k, load_checkpoint(members / f'{k}.json', MEMBER_BYTES), limit)
                for k in range(prefix)]
    return _summary(manifest, reviewed, include_members)
