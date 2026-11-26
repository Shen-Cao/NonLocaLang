"""Generate, compare or score exact five-route reference distributions."""
import argparse
import json
from pathlib import Path

from .core.json_file import read_json
from .protocols.fermion_reference import REFERENCE_STATISTICS, reference_arena, reference_target
from .protocols.five_route_score import five_route_score
from .protocols.quadratic_wire import target_from_dict, target_to_dict
from .protocols.reference_check import compare_reference
from .protocols.serialization import arena_from_dict
from .protocols.wiring import WiringBudgetExceeded


def _score_target_from_dict(data, *, max_bytes, max_scalar_bits):
    # Reject unrelated arenas before exact probability decoding. The full
    # target codec still checks all branches, including cells the score ignores.
    expected = reference_arena()
    limits = dict(max_bytes=max_bytes, max_parties=expected.scenario.parties,
                  max_routes=len(expected.routes),
                  max_total_entries=len(expected.routes)*expected.scenario.entries)
    mismatch = "The score requires the reference five routes and question/answer alphabets"
    if not isinstance(data, dict) or "arena" not in data:
        raise ValueError(mismatch)
    try:
        arena = arena_from_dict(data["arena"], **limits)
    except WiringBudgetExceeded as error:
        raise ValueError(mismatch) from error
    if arena != expected:
        raise ValueError(mismatch)
    return target_from_dict(data, max_scalar_bits=max_scalar_bits, **limits)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nonlocalgames fermion-reference", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    target = commands.add_parser("target", help="print the exact family of conditional probabilities")
    check = commands.add_parser(
        "check", help="compare every coefficient against the specified reference",
        description="Compare an exact target. Exit 0 means the comparison ran, not that it matched; "
                    "read matches_reference in the JSON receipt.")
    score = commands.add_parser("score", help="score a target with fixed, unconditioned weights",
                                description="Report an exact linear score and comparisons with two "
                                            "restricted control families, not a general exclusion.")
    for command in (check, score):
        command.add_argument("file", type=Path)
        command.add_argument("--max-file-bytes", type=int, default=8_000_000)
    score.add_argument("--max-scalar-bits", type=int, default=4096,
                       help="bound input, contrast and score coefficient sizes (default: 4096)")
    for command in (target, check):
        command.add_argument("--statistics", choices=REFERENCE_STATISTICS, default="fermionic")
    args = parser.parse_args(argv)
    try:
        if args.command == "target":
            record = target_to_dict(reference_target(statistics=args.statistics))
        else:
            try:
                data = read_json(args.file, max_file_bytes=args.max_file_bytes)
            except (OverflowError, MemoryError) as error:
                raise ValueError("The --max-file-bytes limit is too large to read safely") from error
            if args.command == "score":
                supplied = _score_target_from_dict(data, max_bytes=args.max_file_bytes,
                                                   max_scalar_bits=args.max_scalar_bits)
                record = five_route_score(supplied, max_scalar_bits=args.max_scalar_bits).to_dict()
            else:
                supplied = target_from_dict(data, max_bytes=args.max_file_bytes)
                record = compare_reference(supplied, statistics=args.statistics)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(record, ensure_ascii=True, allow_nan=False,
                     indent=None if args.command == "target" else 2))
