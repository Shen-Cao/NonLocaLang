"""Independent nonbinary Bell benchmark using an explicit complex realization."""

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import classical_value, npa_bound, nonsignalling_value
from nonlocalgames.quantum import QuantumStrategy


def _cglmp3():
    # I3 from https://arxiv.org/abs/quant-ph/0106024, Eq. (5).
    positive = {(0, 0): 0, (1, 0): -1, (1, 1): 0, (0, 1): 0}
    negative = {(0, 0): -1, (1, 0): 0, (1, 1): -1, (0, 1): 1}
    return Game.from_rule(
        Scenario((2, 2), (3, 3)),
        lambda x, a: 4 * (int((a[0] - a[1] - positive[x]) % 3 == 0)
                         - int((a[0] - a[1] - negative[x]) % 3 == 0)),
        name="CGLMP3 Bell score",
    )


def _cglmp3_strategy():
    # Nonmaximal qutrit state and Fourier settings: https://arxiv.org/abs/quant-ph/0111143.
    gamma = (np.sqrt(11) - np.sqrt(3)) / 2
    state_vector = np.diag([1, gamma, 1]).reshape(-1) / np.sqrt(2 + gamma**2)
    measurements = []
    for sign, shifts in ((1, (0, 0.5)), (-1, (0.25, -0.25))):
        party = []
        for shift in shifts:
            vectors = [np.exp(2j * np.pi * np.arange(3) * (sign * a + shift) / 3) / np.sqrt(3)
                       for a in range(3)]
            party.append(tuple(np.outer(vector, vector.conj()) for vector in vectors))
        measurements.append(tuple(party))
    return QuantumStrategy(np.outer(state_vector, state_vector.conj()), tuple(measurements))


def test_genuine_three_outcome_quantum_violation_and_moment_completeness():
    game, strategy = _cglmp3(), _cglmp3_strategy()
    expected = 1 + np.sqrt(11 / 3)
    assert classical_value(game).value == 2
    assert strategy.score(game) == pytest.approx(expected, abs=1e-12)
    assert nonsignalling_value(game).value == pytest.approx(4, abs=1e-8)
    q1 = npa_bound(game, level="1", solver="SCS", eps=1e-7)
    qab = npa_bound(game, level="1+AB", solver="SCS", eps=1e-7)
    assert qab.value == pytest.approx(expected, abs=1e-5)
    assert q1.value >= qab.value - 1e-5
    # Completeness as a Gram-column relation, independently of generated moment
    # substitution equations, includes every joint word and every matrix row.
    for result in (q1, qab):
        words = result.witness["words"]
        matrix = result.witness["moment_matrix"]
        index = {word: i for i, word in enumerate(words)}
        for party in (0, 1):
            for x in range(2):
                columns = [index[((party, x, a),)] for a in range(3)]
                np.testing.assert_allclose(matrix[:, columns].sum(axis=1), matrix[:, 0], atol=2e-6)
                if result is qab:
                    for y in range(2):
                        for b in range(3):
                            other = (1 - party, y, b)
                            joint = [tuple(sorted(((party, x, a), other))) for a in range(3)]
                            np.testing.assert_allclose(
                                matrix[:, [index[word] for word in joint]].sum(axis=1),
                                matrix[:, index[(other,)]], atol=2e-6,
                            )
