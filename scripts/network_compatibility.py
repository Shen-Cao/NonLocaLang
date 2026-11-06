"""Generate exact network compatibility certificates and a reconstructed source model."""
from dataclasses import asdict
from fractions import Fraction
import json

from nonlocalgames import Scenario
from nonlocalgames.cli import environment, json_default
from nonlocalgames.networks import (RationalBehavior, SourceNetwork, behavior_to_dict,
                                    check_source_independence, reconstruct_bilocal_chain,
                                    triangle_finner, triangle_xor_strategy,
                                    verify_compatibility_violation, violation_to_dict)
from nonlocalgames.networks.serialization import classical_to_dict


def main():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    common = RationalBehavior(scenario, tuple(Fraction(1, 2) if len(set(a)) == 1 else 0
                                              for a in scenario.outputs))
    chain = SourceNetwork(3, ((0, 1), (1, 2)))
    triangle = triangle_xor_strategy()
    excluded = []
    for network, checker in ((chain, check_source_independence), (triangle.network, triangle_finner)):
        result = checker(network, common)
        certificate = violation_to_dict(result.violation)
        rechecked = verify_compatibility_violation(network, common, result.method, certificate)
        excluded.append({'result': asdict(result), 'certificate': certificate,
                         'rechecked_status': rechecked.status})
    xor_behavior = RationalBehavior(scenario, triangle.probabilities())
    reconstructed = reconstruct_bilocal_chain(chain, xor_behavior)
    if reconstructed.witness.probabilities() != xor_behavior.probabilities:
        raise RuntimeError('Reconstructed chain witness failed exact behavior comparison')
    print(json.dumps({
        'environment': environment(), 'common_bit': behavior_to_dict(common),
        'incompatibility_certificates': excluded,
        'xor_behavior': behavior_to_dict(xor_behavior),
        'triangle_check': asdict(triangle_finner(triangle.network, xor_behavior)),
        'chain_reconstruction': {'status': reconstructed.status, 'guarantee': reconstructed.guarantee,
                                 'witness': classical_to_dict(reconstructed.witness)},
        'interpretation': 'The XOR behavior has both triangle and chain realizations; topology need not be unique.'
    }, default=json_default, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
