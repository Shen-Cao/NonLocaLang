"""Prove benchmark quantum advantages from exact realizations and checked bounds."""
import json

from nonlocalgames import ResourceSpec
from nonlocalgames.analysis import classical_value, certify_nonsignalling, certify_xor_upper_bound
from nonlocalgames.certificates.quantum import (certify_quantum_lower_bound,
    certify_quantum_strategy, verify_quantum_lower_certificate)
from nonlocalgames.claims import (EvidenceKind, GapClaim, assess_gap, assessment_to_dict,
                                  claim_to_dict, evidence_to_dict, make_evidence)
from nonlocalgames.cli import environment, json_default
from nonlocalgames.core.serialization import to_dict
from nonlocalgames.games import chsh, ghz, magic_square
from nonlocalgames.quantum import chsh_strategy, magic_square_strategy
from nonlocalgames.quantum.exact import exact_chsh_strategy, exact_ghz_strategy


def main():
    cases = [(chsh(), certify_quantum_strategy(chsh(), exact_chsh_strategy())),
             (ghz(), certify_quantum_strategy(ghz(), exact_ghz_strategy())),
             (magic_square(), certify_quantum_lower_bound(magic_square(), magic_square_strategy()))]
    records = []
    for game, lower in cases:
        local = classical_value(game)
        ns = certify_nonsignalling(game)
        artifacts = [
            make_evidence(game, ResourceSpec('local'), EvidenceKind.LOCAL_ENUMERATION,
                           {'value': str(local.value), 'strategy': local.witness}),
            make_evidence(game, ResourceSpec('quantum_finite'), EvidenceKind.QUANTUM_STRATEGY, lower.witness),
            make_evidence(game, ResourceSpec('nonsignalling'), EvidenceKind.NONSIGNALLING_LP,
                           {key: list(map(str, ns.witness[key])) for key in ('primal', 'dual')})]
        if game.name == 'CHSH':
            upper = certify_xor_upper_bound(game, solver='SCS', eps=1e-8)
            artifacts.append(make_evidence(game, ResourceSpec('quantum_commuting'),
                                           EvidenceKind.XOR_DUAL, upper.witness))
        claim = GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local'))
        checked = verify_quantum_lower_certificate(game, lower.witness)
        records.append({'game': to_dict(game), 'evidence': [evidence_to_dict(a) for a in artifacts],
            'claim': claim_to_dict(game, claim), 'assessment': assessment_to_dict(assess_gap(game, claim, artifacts)),
            'independent_lower_recheck': {'value': checked.value, 'guarantee': checked.guarantee}})
    rounded = certify_quantum_lower_bound(chsh(), chsh_strategy())
    print(json.dumps({'schema': 'nonlocalgames.quantum_certificate_demo.v1', 'environment': environment(),
                      'cases': records, 'rounded_chsh': {'value': rounded.value,
                      'certificate': rounded.witness, 'diagnostics': rounded.diagnostics}},
                     default=json_default, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
