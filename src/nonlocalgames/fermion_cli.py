"""Generate or compare exact five-route fermionic or ordinary reference distributions."""
import argparse
import json
from pathlib import Path

from .core.json_file import read_json
from .protocols.fermion_reference import REFERENCE_STATISTICS, reference_target
from .protocols.quadratic_wire import target_from_dict, target_to_dict
from .protocols.reference_check import compare_reference


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nonlocalgames fermion-reference", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    target = commands.add_parser("target", help="print the exact family of conditional probabilities")
    check = commands.add_parser(
        "check", help="compare every coefficient against the specified reference",
        description="Compare an exact target. Exit 0 means the comparison ran, not that it matched; "
                    "read matches_reference in the JSON receipt.")
    check.add_argument("file", type=Path)
    check.add_argument("--max-file-bytes", type=int, default=8_000_000)
    for command in (target, check):
        command.add_argument("--statistics", choices=REFERENCE_STATISTICS, default="fermionic")
    args = parser.parse_args(argv)
    try:
        if args.command == "target":
            record = target_to_dict(reference_target(statistics=args.statistics))
        else:
            data = read_json(args.file, max_file_bytes=args.max_file_bytes)
            supplied = target_from_dict(data, max_bytes=args.max_file_bytes)
            record = compare_reference(supplied, statistics=args.statistics)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(record, ensure_ascii=True, allow_nan=False,
                     indent=2 if args.command == "check" else None))
