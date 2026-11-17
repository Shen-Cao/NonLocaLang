"""Analyze serialized verifier/resource tasks without weakening their restrictions."""
import argparse
import json
import math

from .core.resource_serialization import load_task, resource_to_dict

CLI_METHODS = ("local", "ns", "ns_exact", "npa", "xor", "xor_certified", "xor_lower_certified", "seesaw")
API_ONLY = {"state_sdp": "Requires supplied measurement operators; use AnalysisTask.solve in Python."}


def witness_to_dict(witness):
    """Preserve source factors for network witnesses and complex quantum data."""
    from .networks.classical import ClassicalNetworkStrategy
    from .networks.quantum import QuantumNetworkStrategy
    from .networks.serialization import classical_to_dict, quantum_to_dict
    from .quantum.serialization import strategy_to_dict
    from .quantum.strategy import QuantumStrategy
    if isinstance(witness, ClassicalNetworkStrategy):
        return classical_to_dict(witness)
    if isinstance(witness, QuantumNetworkStrategy):
        return quantum_to_dict(witness)
    if isinstance(witness, QuantumStrategy):
        return strategy_to_dict(witness)
    from .quantum.exact import ExactQuantumStrategy, strategy_to_dict as exact_to_dict
    if isinstance(witness, ExactQuantumStrategy):
        return exact_to_dict(witness)
    return witness


def _positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _seed(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main(argv=None):
    """Entry point for ``nonlocalgames task``; argv excludes the leading task word."""
    parser = argparse.ArgumentParser(prog="nonlocalgames task", description=__doc__)
    parser.add_argument("--file", required=True, help="serialized AnalysisTask JSON")
    parser.add_argument("--capabilities", action="store_true", help="show methods without running an optimizer")
    parser.add_argument("--method", help="one of " + ", ".join(CLI_METHODS) + "; quantum methods require selection")
    parser.add_argument("--level", choices=("1", "1+AB"), default="1+AB", help="NPA level")
    parser.add_argument("--solver", default="SCS", help="CVXPY solver for quantum optimization")
    parser.add_argument("--tolerance", type=float, default=1e-7,
                        help="SCS eps; also see-saw witness validation tolerance")
    parser.add_argument("--seed", type=_seed, default=0, help="see-saw initialization seed")
    parser.add_argument("--restarts", type=_positive_integer, default=3, help="see-saw restarts")
    parser.add_argument("--max-iterations", type=_positive_integer, default=30, help="see-saw sweep limit")
    parser.add_argument("--witness", action="store_true", help="include a serializable result witness")
    args = parser.parse_args(argv)
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be positive and finite")
    if args.method in API_ONLY:
        parser.error(f"{args.method} is API-only. {API_ONLY[args.method]}")
    if args.method is not None and args.method not in CLI_METHODS:
        parser.error("--method must be one of " + ", ".join(CLI_METHODS))
    if args.capabilities and (args.method is not None or args.witness):
        parser.error("--capabilities does not solve; omit --method and --witness")
    from .cli import environment, json_default, result_record
    from cvxpy.error import SolverError
    try:
        task = load_task(args.file)
        available = task.capabilities()
        record = {"schema": "nonlocalgames.task_result.v1", "game": task.game.name,
                  "resource": resource_to_dict(task.resource), "capabilities": available,
                  "cli_methods": tuple(method for method in available if method in CLI_METHODS),
                  "api_only_methods": {method: API_ONLY[method] for method in available if method in API_ONLY},
                  "environment": environment(), "requested_options": {}}
        if not args.capabilities:
            options = {}
            if args.method in ("npa", "xor", "xor_certified", "xor_lower_certified"):
                options["solver"] = args.solver
                if args.solver.upper() == "SCS":
                    options["eps"] = args.tolerance
                if args.method == "npa":
                    options["level"] = args.level
            elif args.method == "seesaw":
                options = {"solver": args.solver, "seed": args.seed, "restarts": args.restarts,
                           "max_iterations": args.max_iterations, "validation_tolerance": args.tolerance,
                           "solver_options": {"eps": args.tolerance} if args.solver.upper() == "SCS" else {}}
            result = task.solve(method=args.method, **options)
            record["requested_options"] = {"method": args.method, **options}
            record["result"] = result_record(result, include_witness=False)
            if args.witness:
                record["result"]["witness"] = witness_to_dict(result.witness)
        encoded = json.dumps(record, default=json_default, indent=2, allow_nan=False)
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError, SolverError) as error:
        parser.error(str(error))
    print(encoded)
