"""Compare elementary and inflation checks with portable exact certificates."""
from fractions import Fraction
import json

from nonlocalgames import NetworkCompatibilityTask, ResourceSpec, Scenario
from nonlocalgames.cli import environment, json_default
from nonlocalgames.core.compatibility_serialization import compatibility_to_dict
from nonlocalgames.network_cli import check_record
from nonlocalgames.networks import RationalBehavior, triangle_xor_strategy
from nonlocalgames.networks.inflation import verify_spiral_certificate


def main():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    triangle = triangle_xor_strategy()
    w = tuple(Fraction(1, 3) if sum(a) == 1 else 0 for a in scenario.outputs)
    noisy = tuple(Fraction(9, 10)*p + Fraction(1, 80) for p in w)
    resources = ResourceSpec('local', network=triangle.network, shared_randomness='forbidden')
    cases = [('W', w), ('W with uniform mixture weight 1/10', noisy),
             ('independent triangle XOR', triangle.probabilities())]
    records = []
    for name, probabilities in cases:
        task = NetworkCompatibilityTask(RationalBehavior(scenario, probabilities), resources)
        result = task.check(method='spiral')
        record = {'name': name, 'task': compatibility_to_dict(task),
                  'finner': check_record(task.check(method='triangle_finner'), 'triangle_finner'),
                  'spiral': check_record(result, 'spiral', include_witness=True)}
        if result.certificate is not None:
            checked = verify_spiral_certificate(resources.network, task.behavior, result.certificate)
            record['independent_recheck'] = {'status': checked.status, 'guarantee': checked.guarantee}
        records.append(record)
    print(json.dumps({'environment': environment(), 'cases': records,
                      'scope': 'Classical triangle checks. A convex noise mixture is an explicit target distribution, not a source-preserving operation.'},
                     default=json_default, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
