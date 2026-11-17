"""Compare fresh and parameterized numerical XOR screens on p=k/20, k=0,...,20."""
import argparse
from fractions import Fraction
import json
from math import isfinite
from statistics import median
from time import perf_counter

from nonlocalgames.analysis import xor_quantum_value
from nonlocalgames.analysis.xor_template import XorScreenSession
from nonlocalgames.discovery.provenance import runtime_provenance
from nonlocalgames.games import biased_chsh


def measure(method, games, solver, options):
    start = perf_counter()
    if method == 'parameterized':
        session = XorScreenSession(games[0].scenario, solver=solver, **options)
        build_seconds = perf_counter() - start
        solve = session.solve
    else:
        build_seconds = None  # Each fresh call includes its own problem construction.
        def solve(game):
            return xor_quantum_value(game, solver=solver, **options)

    calls = []
    for game in games:
        begin = perf_counter()
        result = solve(game)
        elapsed = perf_counter() - begin
        if result.diagnostics['solver'] != solver or result.diagnostics['certified']:
            raise RuntimeError('Screen returned an unexpected solver or evidence level')
        calls.append({'elapsed_seconds': elapsed, 'compilation_seconds': result.diagnostics['compilation_time'],
                      'solver_seconds': result.diagnostics['solve_time'], 'status': result.status,
                      'value': result.value})
    return {'elapsed_seconds': perf_counter() - start, 'session_build_seconds': build_seconds,
            'first_call_seconds': calls[0]['elapsed_seconds'],
            'following_calls_seconds': sum(c['elapsed_seconds'] for c in calls[1:]), 'calls': calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', type=str.upper, choices=('SCS', 'CLARABEL'), default='SCS')
    parser.add_argument('--accuracy', type=float, default=1e-8, help='requested solver tolerance')
    parser.add_argument('--repeats', type=int, default=2,
                        help='balanced blocks, each timing both orders of the two methods')
    args = parser.parse_args()
    if args.repeats < 1 or not isfinite(args.accuracy) or args.accuracy <= 0:
        parser.error('repeats and solver accuracy must be positive and finite')
    keys = ('eps',) if args.solver == 'SCS' else ('tol_gap_abs', 'tol_gap_rel', 'tol_feas')
    options = dict.fromkeys(keys, args.accuracy)
    options['warm_start'] = False
    games = tuple(biased_chsh(Fraction(k, 20)) for k in range(21))
    methods = ('fresh', 'parameterized')
    # Initialize the runtime. Measured trials start with a new session each time.
    for method in methods:
        measure(method, games[10:11], args.solver, options)

    samples = []
    for _ in range(args.repeats):
        for order in (methods, methods[::-1]):
            pair = {method: measure(method, games, args.solver, options) for method in order}
            differences = [abs(f['value'] - p['value'])
                           for f, p in zip(pair['fresh']['calls'], pair['parameterized']['calls'])]
            samples.append({'order': order, **pair, 'max_abs_value_difference': max(differences),
                            'max_difference_index': differences.index(max(differences))})
    timing_fields = ('elapsed_seconds', 'first_call_seconds', 'following_calls_seconds')
    summary = {method: {field: median(sample[method][field] for sample in samples)
                        for field in timing_fields} for method in methods}
    summary['parameterized']['session_build_seconds'] = median(
        sample['parameterized']['session_build_seconds'] for sample in samples)
    print(json.dumps({'schema': 'nonlocalgames.xor_screen_benchmark.v1',
                      'provenance': runtime_provenance(args.solver), 'solver': args.solver,
                      'solver_options': options, 'balanced_blocks': args.repeats,
                      'probabilities': [str(Fraction(k, 20)) for k in range(21)],
                      'median_timings': summary, 'samples': samples,
                      'max_abs_value_difference': max(s['max_abs_value_difference'] for s in samples)},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
