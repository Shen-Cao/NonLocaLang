"""Compare explicit resource restrictions on CHSH; emit generated JSON evidence."""
import argparse
import json

from nonlocalgames import AnalysisTask, ResourceSpec
from nonlocalgames.cli import environment, json_default, result_record
from nonlocalgames.core.resource_serialization import resource_to_dict
from nonlocalgames.games import chsh
from nonlocalgames.quantum import FixedState, chsh_strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', default='SCS')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    solver_options = {'eps': 1e-8} if args.solver.upper() == 'SCS' else {}
    reference = chsh_strategy()
    cases = [
        ('classical_shared', ResourceSpec('local'), None),
        ('classical_independent', ResourceSpec('local', shared_randomness='forbidden'), None),
        ('quantum_product', ResourceSpec('quantum_finite', (2, 2), state_constraint='product'), None),
        ('quantum_qubits', ResourceSpec('quantum_finite', (2, 2)), 'seesaw'),
        ('fixed_bell_pair', ResourceSpec('quantum_finite', fixed_state=FixedState(reference.state, (2, 2))), 'seesaw'),
        ('ppt_qubits', ResourceSpec('quantum_finite', (2, 2), state_constraint='ppt', ppt_cuts=((0,),)), 'seesaw'),
        ('ppt_fixed_measurements', ResourceSpec('quantum_finite', (2, 2), state_constraint='ppt', ppt_cuts=((0,),)), 'state_sdp'),
    ]
    records = []
    for name, resource, method in cases:
        options = {}
        if method is not None:
            options.update(solver=args.solver, solver_options=solver_options)
        if method == 'seesaw':
            options.update(seed=args.seed, restarts=2, max_iterations=15)
        if method == 'state_sdp':
            options['measurements'] = reference.measurements
        result = AnalysisTask(chsh(), resource).solve(method=method, **options)
        records.append({'name': name, 'resource': resource_to_dict(resource),
                        'result': result_record(result)})
    print(json.dumps({'environment': environment(), 'seed': args.seed,
                      'solver': args.solver, 'solver_options': solver_options, 'results': records},
                     default=json_default, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
