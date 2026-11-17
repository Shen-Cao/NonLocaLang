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
