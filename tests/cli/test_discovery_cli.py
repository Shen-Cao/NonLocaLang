import json

from nonlocalgames.cli import main


def test_public_commands_resume_and_review_weighted_grid(tmp_path, capsys):
    directory = str(tmp_path / "run")
    args = ["weighted-chsh", "run", directory, "--no-screen", "--steps", "2"]
    main([*args, "--max-new", "1"])
    assert json.loads(capsys.readouterr().out)["pending"] == [1, 2]
    main(args)
    assert json.loads(capsys.readouterr().out)["coverage_complete"]
    main(["weighted-chsh", "replay", directory, "--members"])
    review = json.loads(capsys.readouterr().out)
    assert [r["p"] for r in review["results"]] == ["0", "1/2", "1"]
    assert [r["truth"] for r in review["results"]] == ["disproved", "proved", "disproved"]
    assert all(r["screen_status"] == "disabled" for r in review["results"])
