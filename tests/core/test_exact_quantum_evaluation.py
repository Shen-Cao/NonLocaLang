from fractions import Fraction

import pytest

from nonlocalgames import AnalysisTask, ResourceSpec
from nonlocalgames.games import chsh
from nonlocalgames.quantum import FixedState, chsh_strategy
from nonlocalgames.quantum.exact import exact_chsh_strategy


def test_exact_point_evaluation_checks_dimensions_and_keeps_witness_scope():
    strategy = exact_chsh_strategy()
    for randomness in ('allowed', 'forbidden'):
        task = AnalysisTask(chsh(), ResourceSpec('quantum_finite', (3, 3), shared_randomness=randomness))
        result = task.evaluate(strategy)
        assert result.guarantee == 'exact_witness'
        assert result.value == Fraction(3880899, 4546756)
        assert result.witness is strategy
    with pytest.raises(ValueError, match='dimension bounds'):
        AnalysisTask(chsh(), ResourceSpec('quantum_finite', (1, 1))).evaluate(strategy)


def test_exact_point_does_not_erase_unverified_restrictions():
    resources = [ResourceSpec('quantum_finite', measurement_class='projective'),
                 ResourceSpec('quantum_finite', state_constraint='ppt', ppt_cuts=((0,),)),
                 ResourceSpec('quantum_finite', fixed_state=FixedState(chsh_strategy().state, (2, 2))),
                 ResourceSpec('local')]
    for resource in resources:
        with pytest.raises(NotImplementedError):
            AnalysisTask(chsh(), resource).evaluate(exact_chsh_strategy())


def test_exact_point_is_revalidated_before_a_plausible_score_is_accepted():
    from nonlocalgames import Game, Scenario
    from nonlocalgames.quantum.exact import ExactQuantumStrategy, RationalMatrix
    valid = RationalMatrix(((Fraction(1, 2), 0), (0, Fraction(1, 2))))
    point = ExactQuantumStrategy(valid, (((valid, valid),),))
    # A corrupted trace-one state can still give normalized positive probabilities
    # for this uninformative POVM; evaluating only that score would miss nonpositivity.
    object.__setattr__(point, 'state', RationalMatrix(((1, 1), (1, 0))))
    game = Game.from_rule(Scenario((1,), (2,)), lambda x, a: 1)
    with pytest.raises(ValueError, match='not PSD'):
        AnalysisTask(game, ResourceSpec('quantum_finite')).evaluate(point)


def test_exact_strategy_serialization_helper_preserves_rational_data():
    from nonlocalgames.task_cli import witness_to_dict
    from nonlocalgames.quantum.exact import strategy_from_dict
    source = exact_chsh_strategy()
    restored = strategy_from_dict(witness_to_dict(source))
    assert restored == source
    assert AnalysisTask(chsh(), ResourceSpec('quantum_finite')).evaluate(restored).value == source.score(chsh())
