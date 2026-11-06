"""Construct independent-source witnesses and test a nonlinear bilocal constraint."""
from dataclasses import asdict
from fractions import Fraction
import json

from nonlocalgames.cli import environment, json_default
from nonlocalgames.networks import (bilocal_deterministic_branches,
                                    bilocal_entanglement_swapping, bilocal_witness,
                                    triangle_xor_strategy, BILOCAL_SCENARIO)
from nonlocalgames.networks.serialization import classical_to_dict, quantum_to_dict


def main():
    triangle = triangle_xor_strategy()
    swap = bilocal_entanglement_swapping()
    branches = bilocal_deterministic_branches()
    mixture = tuple((a+b)/2 for a, b in zip(*(s.probabilities() for s in branches)))
    record = {
        'environment': environment(),
        'triangle_xor': {'witness': classical_to_dict(triangle),
                         'probabilities': triangle.probabilities()},
        'quantum_bilocal': {'witness': quantum_to_dict(swap),
                           'inequality': asdict(bilocal_witness(swap.probabilities(BILOCAL_SCENARIO)))},
        'global_selector_counterexample': {
            'weights': (Fraction(1, 2), Fraction(1, 2)),
            'branches': [classical_to_dict(s) for s in branches],
            'branch_checks': [asdict(bilocal_witness(s.probabilities())) for s in branches],
            'mixture_check': asdict(bilocal_witness(mixture)),
            'interpretation': 'Mixing these independent-source branches uses a forbidden global selector.'},
    }
    print(json.dumps(record, default=json_default, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
