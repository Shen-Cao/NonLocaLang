"""Public PSD input is bounded; prepared exact arithmetic never reparses strings."""
from fractions import Fraction

import pytest

from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded


def test_public_psd_boundary_parses_bounded_data_and_private_kernel_accepts_only_prepared_fractions(monkeypatch):
    import nonlocalgames.core.scalar_wire as wire
    from nonlocalgames.certificates.linalg import _verify_psd_prepared, verify_psd
    def forbidden(*args, **kwargs):
        pytest.fail('compact oversized exponent must not reach integer conversion or power')
    monkeypatch.setattr(wire, '_decimal_integer', forbidden)
    monkeypatch.setattr(wire, '_power10', forbidden)
    with pytest.raises(ScalarBudgetExceeded):
        verify_psd([['1e1000000000']])
    for value in ('1e1000000000', 1.0, 1):
        with pytest.raises(TypeError):
            _verify_psd_prepared(((value,),))
    derived = Fraction(1, (1 << 16000)-1)
    assert _verify_psd_prepared(((derived,),)).rank == 1
    with pytest.raises(ScalarBudgetExceeded):
        verify_psd(((derived,),))
    assert verify_psd(((derived,),), max_scalar_bits=16000).rank == 1


def test_prepared_kernel_rejects_bounded_numpy_integer_components():
    import numpy as np
    from nonlocalgames.certificates.linalg import _verify_psd_prepared, verify_psd
    fraction = Fraction(np.int64(1), np.int64(2))
    with pytest.raises(TypeError):
        _verify_psd_prepared(((fraction,),))
    assert verify_psd(((fraction,),)).rank == 1


def test_public_psd_validates_scalar_limit_configuration_even_for_empty_matrices():
    from nonlocalgames.certificates.linalg import verify_psd
    assert verify_psd([]).rank == 0
    with pytest.raises(ValueError):
        verify_psd([], max_scalar_bits=0)
    with pytest.raises(ValueError):
        verify_psd([], max_scalar_chars=True)


def test_psd_shape_preflight_precedes_scalar_parsing(monkeypatch):
    import nonlocalgames.certificates.linalg as linalg
    monkeypatch.setattr(linalg, 'parse_rational', lambda *args, **kwargs: pytest.fail('shape must be checked first'))
    def rows():
        yield ('1e1000000000',)
        yield ('1e1000000000',)
        pytest.fail('matrix row-count guard must stop consumption')
    with pytest.raises(ValueError, match='max_size'):
        linalg.verify_psd(rows(), max_size=1)
    with pytest.raises(ValueError, match='square'):
        linalg.verify_psd((('1e1000000000', '0'),))


def test_large_prepared_intermediates_still_get_exact_negative_psd_verdicts():
    from nonlocalgames.certificates.linalg import _verify_psd_prepared, verify_psd
    tiny = Fraction(1, (1 << 16000)-1)
    matrix = ((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1)-tiny))
    with pytest.raises(ValueError, match='not PSD') as failure:
        _verify_psd_prepared(matrix)
    assert not isinstance(failure.value, ScalarBudgetExceeded)
    with pytest.raises(ScalarBudgetExceeded):
        verify_psd(matrix)
    with pytest.raises(ValueError, match='not PSD'):
        verify_psd(matrix, max_scalar_bits=16000)
