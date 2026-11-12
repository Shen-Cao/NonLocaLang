"""Demonstrate checked bound inference without promoting numerical hints to proofs."""
import json
from fractions import Fraction

from nonlocalgames import Game, ResourceSpec, Scenario
from nonlocalgames.analysis import classical_value, certify_nonsignalling, certify_xor_upper_bound
from nonlocalgames.claims import (EvidenceKind, GapClaim, ValueClaim, assess_gap, assess_value,
    assessment_to_dict, claim_to_dict, evidence_to_dict, make_evidence)
from nonlocalgames.cli import environment
from nonlocalgames.core.serialization import to_dict
from nonlocalgames.games import chsh, gyni


def artifacts(game, xor=False):
    local = classical_value(game)
    ns = certify_nonsignalling(game)
    result = [make_evidence(game, ResourceSpec('local'), EvidenceKind.LOCAL_ENUMERATION,
                           {'strategy': local.witness, 'value': str(local.value)}),
              make_evidence(game, ResourceSpec('nonsignalling'), EvidenceKind.NONSIGNALLING_LP,
                           {key: list(map(str, ns.witness[key])) for key in ('primal', 'dual')})]
    if xor:
        result.append(make_evidence(game, ResourceSpec('quantum_commuting'), EvidenceKind.XOR_DUAL,
                                    certify_xor_upper_bound(game, solver='SCS', eps=1e-8).witness))
    return result


def main():
    constant = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: 1, name='constant acceptance')
    cases = []
    for game in (chsh(), gyni(), constant):
        proofs = artifacts(game, xor=game.name == 'CHSH')
        proofs.append(make_evidence(game, ResourceSpec('quantum_finite', (2,)*game.scenario.parties),
            EvidenceKind.NUMERICAL_HINT, {'method': 'untrusted candidate', 'value': 1.0, 'guarantee': 'exact'}))
        claims = [GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local')),
                  GapClaim(ResourceSpec('nonsignalling'), ResourceSpec('local')),
                  ValueClaim(ResourceSpec('quantum_finite'), 'at_most', Fraction(1))]
        cases.append({'game': to_dict(game), 'evidence': [evidence_to_dict(e) for e in proofs],
            'claims': [{'definition': claim_to_dict(game, c), 'assessment': assessment_to_dict(
                assess_gap(game, c, proofs) if isinstance(c, GapClaim) else assess_value(game, c, proofs))}
                for c in claims]})
    print(json.dumps({'schema': 'nonlocalgames.claim_demo.v1', 'environment': environment(), 'cases': cases},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
