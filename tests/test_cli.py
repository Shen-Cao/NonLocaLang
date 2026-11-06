import json
from nonlocalgames.cli import main


def test_cli_exact_output(capsys):
    main(["chsh"])
    result = json.loads(capsys.readouterr().out)
    assert result["result"]["value"] == "3/4"
    assert result["result"]["guarantee"] == "exact"
    assert "witness" not in result["result"]
    assert result["environment"]["packages"]["numpy"]


def test_cli_handles_missing_file_and_solver(capsys):
    import pytest
    for args in (["json", "--file", "/nonexistent/nonlocalgames.json"],
                 ["chsh", "--method", "xor", "--solver", "MISSING_SOLVER"],
                 ["chsh", "--tolerance", "nan"]):
        with pytest.raises(SystemExit) as failure:
            main(args)
        assert failure.value.code == 2
        assert "error:" in capsys.readouterr().err


def test_cli_exact_ns(capsys):
    main(["gyni", "--method", "ns-exact"])
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["guarantee"] == "exact"
    assert result["value"] == "1/3"
