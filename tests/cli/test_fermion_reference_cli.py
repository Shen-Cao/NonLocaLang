import json
import sys

import pytest

from nonlocalgames import Scenario
from nonlocalgames.cli import main
from nonlocalgames.protocols import quadratic_wire
from nonlocalgames.protocols.fermion_reference import reference_arena
from nonlocalgames.protocols.quadratic_wire import TARGET_SCHEMA, target_from_dict, target_id
from nonlocalgames.protocols.serialization import arena_to_dict
from nonlocalgames.protocols.wiring import WiringArena


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


def test_score_replays_the_existing_target_wire_not_a_supplied_receipt(tmp_path, capsys):
    path = tmp_path / "family.json"
    main(["fermion-reference", "target", "--statistics", "fermionic_even_pair"])
    data = json.loads(capsys.readouterr().out)
    data["branches"].reverse()  # names, not wire positions, determine the weights
    data["arena"]["routes"].reverse()
    path.write_text(json.dumps(data))
    main(["fermion-reference", "score", str(path)])
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "nonlocalgames.five_route_score.v1"
    assert record["target_id"] == target_id(target_from_dict(data))
    assert record["score"] == ["-1/5", "0"]
    assert record["contrasts"] == dict.fromkeys(("ABC", "ACB", "AB", "AC", "BC"), ["1", "0"])
    controls = record["comparisons"]
    assert controls["fixed_local_sign_swaps"]["upper_bound"] == ["3/5", "0"]
    assert controls["one_token_equatorial_phases"]["upper_bound_expression"] == "(1+sqrt(5))/4"
    for control in controls.values():
        assert control["statistics"] == ["ordinary_tensor", "fermionic_even_pair"]
        assert control["sources"] and control["readouts"] and not control["exceeds_upper_bound"]
    for flags in (["--max-file-bytes", "3"], ["--max-scalar-bits", "1"]):
        with pytest.raises(SystemExit) as err:
            main(["fermion-reference", "score", str(path), *flags])
        assert err.value.code == 2 and not capsys.readouterr().out

    # The score never uses the all-zero input. It must still reject an
    # unnormalized row there rather than reading a partial set of cells.
    data["branches"][0]["probabilities"][0] = ["1", "0"]
    for invalid in (data, record):
        path.write_text(json.dumps(invalid))
        with pytest.raises(SystemExit) as err:
            main(["fermion-reference", "score", str(path)])
        assert err.value.code == 2 and not capsys.readouterr().out


@pytest.mark.parametrize("variant", ["oversized", "wrong_alphabets", "wrong_routes"])
def test_score_rejects_unrelated_arenas_before_decoding_probabilities(
        variant, tmp_path, capsys, monkeypatch):
    ref = reference_arena()
    if variant == "oversized":
        arena = WiringArena(Scenario((1,), (65_536,)), ((0,),))
    elif variant == "wrong_alphabets":
        arena = WiringArena(Scenario((1, 2, 2), (4, 4, 4)), ref.routes)
    else:
        arena = WiringArena(ref.scenario, ((0, 1, 2), *ref.routes[1:]))
    s = arena.scenario
    row = [["1", "0"]] + [["0", "0"]] * (len(s.outputs) - 1)
    data = {"schema": TARGET_SCHEMA, "encoding": "a_plus_b_sqrt2_positive",
            "arena": arena_to_dict(arena),
            "branches": [{"route": list(r), "probabilities": row * len(s.inputs)}
                         for r in arena.routes]}
    path = tmp_path / "wrong-arena.json"
    path.write_text(json.dumps(data))

    def must_not_decode(*args, **kwargs):
        pytest.fail("an unrelated arena reached probability decoding")

    monkeypatch.setattr(quadratic_wire, "scalar_from_pair", must_not_decode)
    with pytest.raises(SystemExit) as err:
        main(["fermion-reference", "score", str(path)])
    capture = capsys.readouterr()
    assert err.value.code == 2 and not capture.out
    assert "reference five routes and question/answer alphabets" in capture.err


@pytest.mark.parametrize("command", ["check", "score"])
def test_unrepresentable_file_byte_limit_is_a_usage_error(command, tmp_path, capsys):
    path = tmp_path / "valid.json"
    main(["fermion-reference", "target"])
    path.write_text(capsys.readouterr().out)
    with pytest.raises(SystemExit) as err:
        main(["fermion-reference", command, str(path), "--max-file-bytes", str(sys.maxsize)])
    capture = capsys.readouterr()
    assert err.value.code == 2 and not capture.out
    assert "--max-file-bytes limit is too large" in capture.err
