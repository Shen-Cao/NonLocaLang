"""Runtime evidence labels preserve wire strings without authenticating claimed proofs."""
from dataclasses import asdict, replace
from fractions import Fraction
import json

import numpy as np
import pytest

from nonlocalgames.analysis import AnalysisResult
from nonlocalgames.core.evidence_labels import EXACT_SCORE_GUARANTEES, VALUE_RESULT_GUARANTEES, Guarantee


def _result(value=Fraction(1, 2), guarantee='exact', **overrides):
    fields = dict(value=value, resource='local', method='supplied_record',
                  guarantee=guarantee, status='verified')
    fields.update(overrides)
    return AnalysisResult(**fields)


def test_existing_evidence_vocabulary_is_preserved_as_json_strings():
    expected = {
        'exact', 'exact_witness', 'certified_upper_bound', 'certified_lower_bound',
        'numerical_optimum', 'numerical_relaxation', 'numerical_witness',
        'numerical_feasible', 'numerical_block_optimum', 'exact_necessary_condition',
        'exact_membership', 'exact_rational_witness', 'exact_relaxation_feasibility',
        'exact_classical_incompatibility', 'numerical_relaxation_feasibility',
        'uncertified_solver_output',
    }
    assert set(Guarantee) == expected
    for value in expected:
        label = Guarantee(value)
        assert label == value and str(label) == value
        assert json.loads(json.dumps(label)) == value
        if label not in VALUE_RESULT_GUARANTEES:
            with pytest.raises(ValueError, match='scalar AnalysisResult'):
                _result(guarantee=value)
            continue
        result = _result(guarantee=value)
        assert result.guarantee is label
        serialized = json.loads(json.dumps(asdict(result), default=str))
        assert serialized['guarantee'] == value


@pytest.mark.parametrize('guarantee', tuple(EXACT_SCORE_GUARANTEES))
def test_exact_score_labels_reject_floating_or_nonreal_values(guarantee):
    for value in (0.5, np.float64(.5), True, np.bool_(True), '1/2', complex(.5), None):
        with pytest.raises(TypeError, match='exact score|certified bound'):
            _result(value, guarantee)
    for value in (Fraction(1, 2), 0, -3, 10**1000, np.int64(4)):
        assert _result(value, guarantee).value == value


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), np.float64('nan')])
def test_numerical_labels_reject_nonfinite_scores(value):
    with pytest.raises(ValueError, match='finite'):
        _result(value, 'numerical_optimum')


@pytest.mark.parametrize('value', [True, np.bool_(False), complex(1), '0.5', None])
def test_numerical_labels_reject_nonreal_and_boolean_scores(value):
    with pytest.raises(TypeError, match='finite real'):
        _result(value, 'numerical_witness')


def test_finite_numeric_scores_keep_original_scalar_representation():
    for value in (.5, np.float32(.5), Fraction(1, 3), 10**1000):
        result = _result(value, 'numerical_optimum')
        assert result.value is value
    with pytest.raises(ValueError):
        replace(_result(), guarantee='invented_proof')


@pytest.mark.parametrize('field', ['resource', 'method', 'status'])
def test_context_labels_must_be_nonempty_strings(field):
    for value in ('', ' \n '):
        with pytest.raises(ValueError, match='nonempty string'):
            _result(**{field: value})
    with pytest.raises(TypeError, match='nonempty string'):
        _result(**{field: 3})


def test_an_exact_label_does_not_authenticate_a_forged_mathematical_claim():
    forged = _result(Fraction(100), 'exact')
    assert forged.guarantee is Guarantee.EXACT
    assert forged.witness is None
    assert forged.diagnostics == {}
    assert 'remains untrusted' in AnalysisResult.__doc__
    assert 'does not check a certificate' in Guarantee.__doc__
