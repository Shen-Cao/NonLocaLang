"""Large derived exact values serialize without changing Python process-wide limits."""
from fractions import Fraction
import json
import sys

import pytest

from nonlocalgames import Game, ResourceSpec, Scenario
from nonlocalgames.certificates.xor import verify_xor_dual
from nonlocalgames.claims.types import CertifiedInterval
from nonlocalgames.cli import json_default, result_record
from nonlocalgames.core.identity import verifier_id
from nonlocalgames.core.resources import Resource
from nonlocalgames.core.scalar_wire import parse_rational, scalar_bits, scalar_string
from nonlocalgames.games import chsh
from nonlocalgames.properties import (PropertyAssessment, PropertyClaim, property_assessment_to_dict,
                                      property_id)


def _forbid_process_limit_changes(monkeypatch):
    original = sys.get_int_max_str_digits()
    monkeypatch.setattr(sys, 'set_int_max_str_digits',
                        lambda *a: pytest.fail('serialization must not alter process-wide integer limits'))
    return original


def test_derived_xor_pivots_roundtrip_through_cli_json_without_global_changes(monkeypatch):
    original = _forbid_process_limit_changes(monkeypatch)
    q, r = (1 << 3800)+1, (1 << 3801)+1
    game = Game.from_rule(Scenario((2, 1), (2, 2)),
                          lambda x, a: Fraction(1, r) if a[0] == a[1] else 0,
                          distribution={(0, 0): Fraction(1, q), (1, 0): Fraction(q-1, q)})
    result = verify_xor_dual(game, [1, 1, 1])
    pivot = result.diagnostics['psd_pivots'][-1]
    assert scalar_bits(pivot) > 14_300
    decoded = json.loads(json.dumps(result_record(result), default=json_default, allow_nan=False))
    text = decoded['diagnostics']['psd_pivots'][-1]
    assert text == scalar_string(pivot)
    assert parse_rational(text, max_bits=20_000, max_chars=20_000, canonical=True) == pivot
    assert sys.get_int_max_str_digits() == original


def test_property_receipt_formats_large_rational_interval_without_global_changes(monkeypatch):
    original = _forbid_process_limit_changes(monkeypatch)
    game = chsh()
    claim = PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite'))
    lower = Fraction(1, (1 << 20_000)+1)
    # A receipt is an output data record, not a proof admitted from its labels.
    receipt = PropertyAssessment(property_id(game, claim), verifier_id(game), 'unknown', 'completed',
        intervals=(CertifiedInterval(Resource.QUANTUM_FINITE, lower, Fraction(1)),))
    encoded = property_assessment_to_dict(receipt)
    text = encoded['intervals'][0]['lower']
    assert len(text) > 4300
    assert parse_rational(text, max_bits=20_001, max_chars=20_000, canonical=True) == lower
    assert encoded['intervals'][0]['upper'] == '1'
    assert json.loads(json.dumps(encoded, allow_nan=False))['truth'] == 'unknown'
    assert sys.get_int_max_str_digits() == original


def test_ordinary_fraction_output_bytes_are_unchanged():
    assert json.dumps({'value': Fraction(1, 3)}, default=json_default) == '{"value": "1/3"}'
