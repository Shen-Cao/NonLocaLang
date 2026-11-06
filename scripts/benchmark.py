"""Print reproducible baseline results; redirect JSON into ignored output/."""
import argparse
import json
from time import perf_counter

from nonlocalgames.analysis import (certify_nonsignalling, classical_value, nonsignalling_value,
                                    npa_bound, xor_quantum_value)
from nonlocalgames.cli import environment, json_default, result_record
from nonlocalgames.games import GAMES
from nonlocalgames.quantum import (chsh_strategy, ghz_strategy, magic_square_strategy,
                                  verify_chsh_sos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--include-magic-square-sdp", action="store_true")
    args = parser.parse_args()
    options = {"solver": args.solver}
    if args.solver.upper() == "SCS":
        options["eps"] = 1e-7
    records = []
    for name, constructor in GAMES.items():
        game = constructor()
        methods = [("local", lambda: classical_value(game)),
                   ("ns", lambda: nonsignalling_value(game)),
                   ("ns-exact", lambda: certify_nonsignalling(game))]
        if name == "chsh" or (name == "magic-square" and args.include_magic_square_sdp):
            methods.append(("npa1ab", lambda: npa_bound(game, **options)))
        if name == "chsh":
            methods.append(("xor", lambda: xor_quantum_value(game, **options)))
        for method, run in methods:
            start = perf_counter()
            result = run()
            records.append({"game": name, "method": method,
                            "elapsed_seconds": perf_counter() - start,
                            "result": result_record(result)})
    certificate = verify_chsh_sos()
    print(json.dumps({"environment": environment(), "solver_options": options,
                      "results": records, "reference_strategy_scores": {
                          "chsh": chsh_strategy().score(GAMES["chsh"]()),
                          "ghz": ghz_strategy().score(GAMES["ghz"]()),
                          "magic-square": magic_square_strategy().score(GAMES["magic-square"]())},
                      "chsh_symbolic_certificate_verified": certificate.verified},
                     default=json_default, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
