"""Explicit compatibility checks for serialized independent-source tasks."""
import argparse
import json
import math

from .core.compatibility_serialization import load_compatibility
from .core.resource_serialization import resource_to_dict


CHECKS = ('source_independence', 'triangle_finner', 'bilocal_chain', 'spiral')


def check_record(result, method, include_witness=False):
    """Record common evidence fields without copying omitted dense certificates."""
    record = {'method': method, 'status': result.status, 'guarantee': result.guarantee}
    for field in ('resource', 'diagnostics', 'checks', 'assumptions', 'applicable_resources'):
        if hasattr(result, field):
            record[field] = getattr(result, field)
    if include_witness:
        certificate = getattr(result, 'certificate', None)
        if certificate is not None:
            from .networks.inflation import spiral_certificate_to_dict
            record['certificate'] = spiral_certificate_to_dict(certificate)
        violation = getattr(result, 'violation', None)
        if violation is not None:
            from .networks import violation_to_dict
            record['violation'] = violation_to_dict(violation)
        witness = getattr(result, 'witness', None)
        if witness is not None:
            from .networks.serialization import classical_to_dict
            record['witness'] = classical_to_dict(witness)
    return record


def main(argv=None):
    from .cli import environment, json_default
    parser = argparse.ArgumentParser(prog='nonlocalgames network', description=__doc__)
    parser.add_argument('--file', required=True, help='serialized NetworkCompatibilityTask JSON')
    parser.add_argument('--capabilities', action='store_true')
    parser.add_argument('--method', choices=CHECKS)
    parser.add_argument('--witness', action='store_true', help='include exact certificate or constructive witness')
    parser.add_argument('--max-denominator', type=int, default=1_000_000,
                        help='spiral rational reconstruction denominator limit')
    parser.add_argument('--time-limit', type=float, help='spiral HiGHS time limit per LP solve, in seconds')
    args = parser.parse_args(argv)
    if args.capabilities and (args.method is not None or args.witness):
        parser.error('--capabilities does not solve; omit --method and --witness')
    if not args.capabilities and args.method is None:
        parser.error('Select --method explicitly or inspect --capabilities')
    if args.max_denominator < 1:
        parser.error('--max-denominator must be positive')
    if args.time_limit is not None and (not math.isfinite(args.time_limit) or args.time_limit < 0):
        parser.error('--time-limit must be nonnegative and finite')
    if args.time_limit is not None and args.method != 'spiral':
        parser.error('--time-limit applies only to spiral')
    try:
        task = load_compatibility(args.file)
        record = {'schema': 'nonlocalgames.network_check_result.v1',
                  'resource': resource_to_dict(task.resource), 'capabilities': task.capabilities(),
                  'environment': environment(), 'requested_options': {}}
        if not args.capabilities:
            options = {}
            if args.method == 'spiral':
                options['max_denominator'] = args.max_denominator
                if args.time_limit is not None:
                    options['solver_options'] = {'time_limit': args.time_limit}
            result = task.check(method=args.method, **options)
            record['requested_options'] = {'method': args.method, **options}
            record['result'] = check_record(result, args.method, args.witness)
        encoded = json.dumps(record, default=json_default, indent=2, allow_nan=False)
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError) as error:
        parser.error(str(error))
    print(encoded)
