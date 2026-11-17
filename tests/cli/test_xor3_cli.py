import json

import pytest

from nonlocalgames.cli import main
from nonlocalgames.discovery.xor3 import xor3_game
from nonlocalgames.equivalence import relabel_game, relabelling_from_dict


def test_cli_classifies_binary_mask_and_exposes_budget_completion(capsys):
    main(["xor3", "canonicalize", "100000000"])
    complete = json.loads(capsys.readouterr().out)
    assert complete["representative"] == 1
    assert relabel_game(xor3_game(256), relabelling_from_dict(complete["transporter"])) == xor3_game(1)
    main(["xor3", "canonicalize", "100000000", "--max-candidates", "71"])
    partial = json.loads(capsys.readouterr().out)
    assert not partial["complete"] and partial["examined"] == 71
    assert partial["representative"] is partial["orbit_id"] is partial["transporter"] is None


def test_cli_rejects_incomplete_mask(capsys):
    with pytest.raises(SystemExit) as failure:
        main(["xor3", "canonicalize", "10"])
    assert failure.value.code == 2 and "nine binary digits" in capsys.readouterr().err
