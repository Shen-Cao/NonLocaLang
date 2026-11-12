from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import numpy as np

from nonlocalgames import Game, ResourceSpec
from nonlocalgames.core.identity import resource_id, verifier_digest, verifier_id
from nonlocalgames.core.numeric_data import matrix_from_dict
from nonlocalgames.games import chsh
from nonlocalgames.quantum import FixedState


def test_verifier_identity_preserves_the_legacy_exact_certificate_context():
    game = chsh()
    legacy = {'input_sizes': game.scenario.input_sizes, 'output_sizes': game.scenario.output_sizes,
              'distribution': tuple(map(str, game.distribution)), 'payoff': tuple(map(str, game.payoff))}
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert verifier_digest(game) == expected
    assert verifier_id(replace(game, name='cosmetic only')) == verifier_id(game)
    changed = Game(game.scenario, (Fraction(1, 2), Fraction(1, 6), Fraction(1, 6), Fraction(1, 6)),
                   game.payoff)
    assert verifier_id(changed) != verifier_id(game)


def test_resource_identity_keeps_restrictions_and_numeric_zero_representation():
    assert resource_id(ResourceSpec('local')) != resource_id(ResourceSpec('local', shared_randomness='forbidden'))
    positive = np.eye(2, dtype=complex)/2
    negative = positive.conjugate()
    assert np.signbit(negative.imag).all()
    a, b = FixedState(positive, (2,)), FixedState(negative, (2,))
    assert resource_id(ResourceSpec('quantum_finite', fixed_state=a)) != resource_id(ResourceSpec('quantum_finite', fixed_state=b))
    decoded = matrix_from_dict({'real': negative.real.tolist(), 'imag': negative.imag.tolist()}, 2)
    assert decoded.tobytes() == negative.tobytes()
