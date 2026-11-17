import json

from nonlocalgames.cli import main


def test_wiring_examples_prove_disprove_and_remain_unknown(tmp_path, capsys):
    def example(name):
        main(["wiring", "example", name])
        path = tmp_path / (name + ".json")
        path.write_text(capsys.readouterr().out, encoding="utf-8")
        return str(path)

    target, strategy = example("bit-exchange"), example("bit-exchange-strategy")
    for args, verdict in (([target, "--strategy", strategy], "proved"), ([target], "unknown"),
                          ([example("inconsistent-alice")], "disproved")):
        main(["wiring", "check", *args])
        record = json.loads(capsys.readouterr().out)
        assert record["truth"] == verdict and record["execution"] == "completed"
        if verdict == "proved":
            assert record["strategy_id"]
        if verdict == "disproved":
            assert record["counterexample"]["party"] == 0
