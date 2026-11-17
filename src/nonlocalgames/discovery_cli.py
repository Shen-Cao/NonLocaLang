"""Run or independently replay a finite weighted CHSH experiment."""
import argparse
import json
from pathlib import Path

from .claims import VerificationBudget
from .core.json_file import read_json


def _budget(path):
    if path is None:
        return None
    data = read_json(path, max_file_bytes=16_384)
    if not isinstance(data, dict):
        raise ValueError("Verification budget must be a JSON object")
    return VerificationBudget(**data)


def main(argv=None):
    from .discovery.weighted_run import replay_weighted_chsh, run_weighted_chsh

    parser = argparse.ArgumentParser(prog="nonlocalgames weighted-chsh", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="create a run or resume one with matching parameters")
    run.add_argument("directory", type=Path)
    run.add_argument("--steps", type=int, default=20, help="sample p=k/steps, including both endpoints")
    run.add_argument("--grid-denominator", type=int, default=16, help="denominator of the rational witness grid")
    run.add_argument("--no-screen", dest="screen", action="store_false", help="skip the numerical XOR screen")
    run.add_argument("--solver", default="SCS", help="requested numerical screen solver")
    run.add_argument("--eps", type=float, default=1e-8, help="SCS accuracy setting")
    run.add_argument("--max-new", type=int, help="stop after this many new member records")
    run.add_argument("--budget", type=Path, help="JSON object of VerificationBudget overrides")
    replay = commands.add_parser("replay", help="recompute verdicts from the checkpointed raw evidence")
    replay.add_argument("directory", type=Path)
    replay.add_argument("--budget", type=Path, help="local verification policy for archived evidence")
    replay.add_argument("--members", action="store_true", help="include a checked result for each recorded point")
    args = parser.parse_args(argv)
    try:
        budget = _budget(args.budget)
        if args.command == "run":
            result = run_weighted_chsh(args.directory, steps=args.steps, grid_denominator=args.grid_denominator,
                                       screen=args.screen, solver=args.solver, eps=args.eps, max_new=args.max_new,
                                       verification_budget=budget)
        else:
            result = replay_weighted_chsh(args.directory, budget=budget, include_members=args.members)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
