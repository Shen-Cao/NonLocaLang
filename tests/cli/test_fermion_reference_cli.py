import json

import pytest

from nonlocalgames.cli import main
from nonlocalgames.protocols.quadratic_wire import target_from_dict, target_id


def test_full_exact_reference_exports_and_compares_in_both_conventions(tmp_path, capsys):
    main(["fermion-reference", "target"])
    raw = capsys.readouterr().out
    target = target_from_dict(json.loads(raw))
    assert sum(p.b != 0 for b in target.behaviors for p in b.probabilities) == 352
    path = tmp_path / "fermion.json"
    path.write_text(raw)
    for statistics, count in (("fermionic", 0), ("ordinary_tensor", 64)):
        main(["fermion-reference", "check", str(path), "--statistics", statistics])
        result = json.loads(capsys.readouterr().out)
        assert result["reference"]["statistics"] == statistics
        assert result["checked_entries"] == 2560 and result["mismatches"] == count
        assert result["target_id"] == target_id(target)
        assert result["matches_reference"] == (count == 0)
        assert (result["first_mismatch"] is None) == (count == 0)
    with pytest.raises(SystemExit):
        main(["fermion-reference", "check", str(path), "--max-file-bytes", "3"])


def test_even_pair_reference_identifies_physical_model_separately_from_the_target(tmp_path, capsys):
    path = tmp_path / "double.json"
    main(["fermion-reference", "target", "--statistics", "fermionic_even_pair"])
    path.write_text(capsys.readouterr().out)
    receipts = []
    for kind in ("fermionic_even_pair", "ordinary_tensor", "fermionic"):
        main(["fermion-reference", "check", str(path), "--statistics", kind])
        receipts.append(json.loads(capsys.readouterr().out))
    double, tensor, single = receipts
    assert double["reference"]["modes_per_port"] == 2
    assert double["reference"]["schema"] == "nonlocalgames.even_pair_reference.v1"
    assert double["reference_id"] != tensor["reference_id"]
    assert double["expected_target_id"] == tensor["expected_target_id"]
    assert double["checked_entries"] == 2560 and double["mismatches"] == tensor["mismatches"] == 0
    assert single["mismatches"] == 64
    # Existing reference definitions, not just the output distributions, are stable.
    prefix = "nonlocalgames.six_mode_reference.v1:"
    assert single["reference_id"] == prefix + "c85a0021f46f362ef0b2e931a0a94e318c5abfade8ef57a59861373021c8a11d"
    assert tensor["reference_id"] == prefix + "051f8367aadc7bcb2e28e05b742439ac522d110bb711d77f6cfe97ee13c3ff88"
