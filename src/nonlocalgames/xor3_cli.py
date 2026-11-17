"""Classify a uniform 3×3 XOR predicate or replay the full finite corpus."""
import argparse
import json
from pathlib import Path

from .discovery import XOR3_ACTION, XOR3_FAMILY, canonicalize_xor3, xor3_game
from .equivalence.serialization import relabelling_to_dict


def _mask(bits):
    if len(bits) != 9 or any(bit not in "01" for bit in bits):
        raise argparse.ArgumentTypeError("expected nine binary digits in row-major order")
    return int(bits, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nonlocalgames xor3", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    classify = commands.add_parser("canonicalize", help="find a representative and transporter")
    classify.add_argument("bits", type=_mask, help="nine winning parity bits, e.g. 000001010")
    classify.add_argument("--max-candidates", type=int, default=72)
    build = commands.add_parser("build", help="generate and verify all 512 members")
    build.add_argument("archive", type=Path)
    replay = commands.add_parser("replay", help="check a saved corpus using its raw evidence")
    replay.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "canonicalize":
            result = canonicalize_xor3(xor3_game(args.bits), max_candidates=args.max_candidates)
            record = {"schema": "nonlocalgames.xor3_classification.v1", "family": XOR3_FAMILY,
                      "action": XOR3_ACTION, "mask": args.bits, "complete": result.complete,
                      "examined": result.examined, "representative": result.representative,
                      "orbit_id": result.orbit_id, "transporter": None}
            if result.complete:
                record["transporter"] = relabelling_to_dict(result.transporter)
        else:
            from .discovery.xor3_corpus import replay_xor3_corpus, write_xor3_corpus
            operation = write_xor3_corpus if args.command == "build" else replay_xor3_corpus
            record = operation(args.archive)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2, allow_nan=False))
