"""Command-line entry point for reproducible small-game experiments."""
import argparse
from dataclasses import fields
from fractions import Fraction
import importlib.metadata
import json
import math
import platform
import sys

from .analysis import (certify_nonsignalling, classical_value, nonsignalling_value,
                       npa_bound, xor_quantum_value, certify_xor_upper_bound)
from .core.serialization import load
from .core.scalar_wire import scalar_string
from .games import GAMES


def environment():
    packages = {}
    for name in ("numpy", "scipy", "cvxpy", "sympy", "scs"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {"python": platform.python_version(), "packages": packages}


def json_default(value):
    if isinstance(value, Fraction):
        return scalar_string(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def result_record(result, include_witness=False):
    return {field.name: getattr(result, field.name) for field in fields(result)
            if include_witness or field.name != "witness"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "claim":
        from .claim_cli import main as claim_main
        return claim_main(argv[1:])
    if argv and argv[0] == "weighted-chsh":
        from .discovery_cli import main as discovery_main
        return discovery_main(argv[1:])
    if argv and argv[0] == "wiring":
        from .wiring_cli import main as wiring_main
        return wiring_main(argv[1:])
    if argv and argv[0] == "xor3":
        from .xor3_cli import main as xor3_main
        return xor3_main(argv[1:])
    if argv and argv[0] == "property":
        from .property_cli import main as property_main
        return property_main(argv[1:])
    if argv and argv[0] == "network":
        from .network_cli import main as network_main
        return network_main(argv[1:])
    if argv and argv[0] == "task":
        from .task_cli import main as task_main
        return task_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Subcommands: task, network, claim, property, xor3, wiring, weighted-chsh. Use COMMAND --help for options.")
    parser.add_argument("game", choices=tuple(GAMES) + ("json",))
    parser.add_argument("--file", help="versioned verifier JSON, required for game=json")
    parser.add_argument("--method", choices=("local", "ns", "ns-exact", "npa1", "npa1ab", "xor", "xor-certified"),
                        default="local")
    parser.add_argument("--solver", default="SCS", help="CVXPY solver for SDP methods")
    parser.add_argument("--tolerance", type=float, default=1e-7, help="SCS eps (only SCS)")
    parser.add_argument("--witness", action="store_true")
    args = parser.parse_args(argv)
    if args.game == "json" and not args.file:
        parser.error("--file is required with game=json")
    if args.game != "json" and args.file:
        parser.error("--file requires game=json")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be positive and finite")
    options = {"solver": args.solver}
    if args.solver.upper() == "SCS":
        options["eps"] = args.tolerance
    from cvxpy.error import SolverError
    try:
        game = load(args.file) if args.game == "json" else GAMES[args.game]()
        if args.method == "local":
            result = classical_value(game)
        elif args.method == "ns-exact":
            result = certify_nonsignalling(game)
        elif args.method == "ns":
            result = nonsignalling_value(game)
        elif args.method == "xor":
            result = xor_quantum_value(game, **options)
        elif args.method == "xor-certified":
            result = certify_xor_upper_bound(game, **options)
        else:
            result = npa_bound(game, level="1" if args.method == "npa1" else "1+AB", **options)
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError, SolverError) as error:
        parser.error(str(error))
    print(json.dumps({"game": game.name, "result": result_record(result, args.witness),
                      "environment": environment(), "requested_options": options if args.method
                      not in ("local", "ns", "ns-exact") else {}}, default=json_default, indent=2,
                     allow_nan=False))
