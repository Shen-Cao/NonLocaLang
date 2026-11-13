"""Inspect precise verifier properties and rechecked attainment claims."""
import argparse
import json
from pathlib import Path

from .claims import VerificationBudget, evidence_from_dict
from .core.numeric_data import positive_budget
from .core.resource_serialization import resource_from_dict
from .core.resources import ResourceSpec
from .core.serialization import from_dict as game_from_dict
from .games import GAMES
from .properties import (PropertyClaim, PropertyKind, assess_property,
                         property_assessment_to_dict, property_claim_to_dict)
from .properties.types import STRUCTURAL_KINDS


def _load(path, limit):
    positive_budget(limit, 'max_file_bytes')
    with Path(path).open('rb') as stream:
        data = stream.read(limit+1)
    if len(data) > limit:
        raise ValueError('Input JSON exceeds max_file_bytes')
    return json.loads(data)


def main(argv=None):
    from .cli import environment
    parser = argparse.ArgumentParser(prog='nonlocalgames property', description=__doc__)
    parser.add_argument('game', choices=tuple(GAMES)+('json',))
    parser.add_argument('--file', help='verifier JSON when game=json')
    parser.add_argument('--kind', choices=tuple(k.value for k in PropertyKind), required=True)
    parser.add_argument('--resource', help='explicit standard resource kind for perfect_strategy')
    parser.add_argument('--resource-file', help='complete versioned resource specification')
    parser.add_argument('--off-support', choices=('all', 'sampled'))
    parser.add_argument('--evidence', help='JSON list of raw evidence envelopes; no result receipts')
    parser.add_argument('--max-property-entries', type=int, default=1_000_000)
    parser.add_argument('--max-artifacts', type=int, default=100)
    parser.add_argument('--max-file-bytes', type=int, default=64_000_000)
    args = parser.parse_args(argv)
    if (args.game == 'json') != (args.file is not None):
        parser.error('--file is required only with game=json')
    if args.resource is not None and args.resource_file is not None:
        parser.error('Choose --resource or --resource-file, not both')
    if PropertyKind(args.kind) in STRUCTURAL_KINDS and args.evidence is not None:
        parser.error('Structural queries inspect the verifier and do not accept evidence files')
    try:
        positive_budget(args.max_file_bytes, 'max_file_bytes')
        budget = VerificationBudget(max_property_entries=args.max_property_entries,
                                     max_artifacts=args.max_artifacts)
        game = game_from_dict(_load(args.file, args.max_file_bytes)) if args.game == 'json' else GAMES[args.game]()
        resource = None
        if args.resource is not None:
            resource = ResourceSpec(args.resource)
        elif args.resource_file is not None:
            resource = resource_from_dict(_load(args.resource_file, args.max_file_bytes))
        claim = PropertyClaim(args.kind, resource, off_support=args.off_support)
        artifacts = ()
        if args.evidence is not None:
            data = _load(args.evidence, args.max_file_bytes)
            if not isinstance(data, list):
                raise ValueError('Evidence file must be a list of raw artifact envelopes')
            if len(data) > budget.max_artifacts:
                raise ValueError('Evidence file exceeds max_artifacts')
            artifacts = tuple(evidence_from_dict(item, max_bytes=budget.max_artifact_bytes) for item in data)
        result = assess_property(game, claim, artifacts, budget=budget)
        record = {'schema': 'nonlocalgames.property_result.v1', 'environment': environment(),
                  'definition': property_claim_to_dict(game, claim),
                  'assessment': property_assessment_to_dict(result)}
        encoded = json.dumps(record, indent=2, allow_nan=False)
    except (OSError, TypeError, ValueError, ZeroDivisionError, NotImplementedError, RuntimeError) as error:
        parser.error(str(error))
    print(encoded)
