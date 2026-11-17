"""Generate and check rational target families for hidden classical wirings."""
import argparse
import json
from pathlib import Path

from .core.json_file import read_json
from .protocols.classical_wire import classical_from_dict, classical_to_dict
from .protocols.compatibility import assess_wiring, wiring_assessment_to_dict
from .protocols.examples import bit_exchange_strategy, bit_exchange_target, inconsistent_alice_target
from .protocols.serialization import target_from_dict, target_to_dict


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nonlocalgames wiring", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("example", help="print a portable example target or strategy")
    example.add_argument("name", choices=("bit-exchange", "bit-exchange-strategy", "inconsistent-alice"))
    check = commands.add_parser("check", help="assess joint compatibility across all routes")
    check.add_argument("target", type=Path)
    check.add_argument("--strategy", type=Path, help="joint preparations and shared local responses")
    check.add_argument("--max-file-bytes", type=int, default=8_000_000)
    check.add_argument("--max-work", type=int, default=10_000_000)
    args = parser.parse_args(argv)
    try:
        if args.command == "example":
            if args.name == "bit-exchange-strategy":
                record = classical_to_dict(bit_exchange_strategy())
            else:
                target = bit_exchange_target() if args.name == "bit-exchange" else inconsistent_alice_target()
                record = target_to_dict(target)
        else:
            target = target_from_dict(read_json(args.target, max_file_bytes=args.max_file_bytes),
                                      max_bytes=args.max_file_bytes)
            strategy = None
            if args.strategy is not None:
                data = read_json(args.strategy, max_file_bytes=args.max_file_bytes)
                strategy = classical_from_dict(target.arena, data)
            record = wiring_assessment_to_dict(assess_wiring(target, strategy=strategy, max_work=args.max_work))
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2, allow_nan=False))
