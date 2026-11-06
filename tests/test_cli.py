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


def test_summary_does_not_copy_omitted_witness():
    from nonlocalgames.analysis import AnalysisResult
    from nonlocalgames.cli import result_record

    class NoCopyWitness:
        def __deepcopy__(self, memo):
            raise AssertionError('omitted witnesses must not be copied')

    result = AnalysisResult(0, 'local', 'test', 'exact', 'optimal', witness=NoCopyWitness())
    assert 'witness' not in result_record(result)


def test_main_routes_serialized_resource_tasks(tmp_path, capsys):
    from nonlocalgames import AnalysisTask, ResourceSpec
    from nonlocalgames.core.resource_serialization import dump_task
    from nonlocalgames.games import chsh
    path = tmp_path / 'task.json'
    dump_task(AnalysisTask(chsh(), ResourceSpec('local', shared_randomness='forbidden')), path)
    main(['task', '--file', str(path), '--method', 'local'])
    result = json.loads(capsys.readouterr().out)
    assert result['result']['value'] == '3/4'
    assert result['resource']['shared_randomness'] == 'forbidden'


def test_cli_certified_xor_bound_can_be_reverified(capsys):
    from fractions import Fraction
    from nonlocalgames.analysis import verify_xor_certificate
    from nonlocalgames.games import chsh
    main(['chsh', '--method', 'xor-certified', '--witness'])
    record = json.loads(capsys.readouterr().out)['result']
    checked = verify_xor_certificate(chsh(), record['witness'])
    assert record['guarantee'] == 'certified_upper_bound'
    assert checked.value == Fraction(record['value'])
    assert (checked.value - Fraction(1, 2))**2 >= Fraction(1, 8)
