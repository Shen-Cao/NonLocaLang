"""Recheck structural and attainment properties on the benchmark corpus."""
import argparse
import json
from pathlib import Path

from nonlocalgames import ResourceSpec
from nonlocalgames.analysis import (classical_value, certify_nonsignalling,
    certify_quantum_lower_bound, certify_quantum_strategy, certify_xor_upper_bound)
from nonlocalgames.claims import EvidenceKind, evidence_to_dict, make_evidence
from nonlocalgames.cli import environment
from nonlocalgames.core.serialization import to_dict
from nonlocalgames.games import GAMES
from nonlocalgames.properties import (PropertyClaim, assess_property,
    property_assessment_to_dict, property_claim_to_dict)
from nonlocalgames.quantum import magic_square_strategy
from nonlocalgames.quantum.exact import exact_chsh_strategy, exact_ghz_strategy


def evidence(game, name):
    local, ns = classical_value(game), certify_nonsignalling(game)
    artifacts = [make_evidence(game, ResourceSpec('local'), EvidenceKind.LOCAL_ENUMERATION,
                               {'strategy': local.witness, 'value': str(local.value)}),
                 make_evidence(game, ResourceSpec('nonsignalling'), EvidenceKind.NONSIGNALLING_LP,
                               {k: list(map(str, ns.witness[k])) for k in ('primal', 'dual')})]
    quantum = None
    if name == 'chsh':
        quantum = certify_quantum_strategy(game, exact_chsh_strategy())
        upper = certify_xor_upper_bound(game, solver='SCS', eps=1e-8)
        artifacts.append(make_evidence(game, ResourceSpec('quantum_commuting'), EvidenceKind.XOR_DUAL, upper.witness))
    elif name == 'ghz':
        quantum = certify_quantum_strategy(game, exact_ghz_strategy())
    elif name == 'magic-square':
        quantum = certify_quantum_lower_bound(game, magic_square_strategy())
    if quantum is not None:
        artifacts.append(make_evidence(game, ResourceSpec('quantum_finite'), EvidenceKind.QUANTUM_STRATEGY, quantum.witness))
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, help='optional directory for raw evidence lists')
    args = parser.parse_args()
    if args.evidence_dir is not None:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, constructor in GAMES.items():
        game, properties = constructor(), []
        artifacts = evidence(game, name)
        wire = [evidence_to_dict(a) for a in artifacts]
        if args.evidence_dir is not None:
            (args.evidence_dir / (name+'.json')).write_text(json.dumps(wire, indent=2)+'\n')
        for kind in ('acceptance', 'boolean_predicate', 'xor_predicate', 'xor_score'):
            claim = PropertyClaim(kind, off_support='sampled')
            properties.append({'definition': property_claim_to_dict(game, claim),
                'assessment': property_assessment_to_dict(assess_property(game, claim))})
        for claim in (PropertyClaim('perfect_strategy', ResourceSpec('local')),
                      PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')),
                      PropertyClaim('pseudotelepathy')):
            properties.append({'definition': property_claim_to_dict(game, claim),
                'assessment': property_assessment_to_dict(assess_property(game, claim, artifacts))})
        records.append({'name': name, 'game': to_dict(game), 'evidence': wire, 'properties': properties})
    print(json.dumps({'schema': 'nonlocalgames.property_demo.v1', 'environment': environment(),
                      'cases': records}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
