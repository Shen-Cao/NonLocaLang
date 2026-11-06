"""Screen rationally weighted CHSH games for numerical separation candidates."""
import argparse
from fractions import Fraction
import json

from nonlocalgames.analysis import classical_value, xor_quantum_value
from nonlocalgames.cli import environment, json_default, result_record
from nonlocalgames.games import biased_chsh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--solver", default="SCS")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    options = {"solver": args.solver}
    if args.solver.upper() == "SCS":
        options["eps"] = 1e-8
    records = []
    for k in range(args.steps + 1):
        p = Fraction(k, args.steps)
        game = biased_chsh(p)
        local = classical_value(game)
        quantum = xor_quantum_value(game, **options)
        records.append({"p": p, "local": local.value,
                        "quantum": result_record(quantum),
                        "candidate_gap": quantum.value - float(local.value)})
    print(json.dumps({"environment": environment(), "solver_options": options,
                      "interpretation": "Numerical screening only; gaps are not certified separations.",
                      "candidates": sorted(records, key=lambda r: r["candidate_gap"], reverse=True)},
                     default=json_default, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
