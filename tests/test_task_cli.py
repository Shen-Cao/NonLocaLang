"""Task commands must preserve resource contracts and witness schemas."""
import json

import numpy as np
import pytest

from nonlocalgames import AnalysisTask, ResourceSpec
from nonlocalgames.core.resource_serialization import dump_task
from nonlocalgames.games import chsh
from nonlocalgames.networks import SourceNetwork
from nonlocalgames.networks.serialization import classical_from_dict, quantum_from_dict
from nonlocalgames.quantum.serialization import strategy_from_dict
from nonlocalgames.task_cli import main


def _file(tmp_path, resource):
    task = AnalysisTask(chsh(), resource)
    path = tmp_path / "task.json"
    dump_task(task, path)
    return task, str(path)


def test_capabilities_is_read_only_and_explains_api_only_methods(tmp_path, capsys, monkeypatch):
    _, path = _file(tmp_path, ResourceSpec("quantum_finite", (2, 2)))
    monkeypatch.setattr(AnalysisTask, "solve", lambda *args, **kwargs: pytest.fail("must not optimize"))
    main(["--file", path, "--capabilities"])
    record = json.loads(capsys.readouterr().out)
    assert record["resource"]["local_dimensions"] == [2, 2]
    assert record["capabilities"] == ["seesaw", "state_sdp"]
    assert record["cli_methods"] == ["seesaw"]
    assert "measurement" in record["api_only_methods"]["state_sdp"]
    assert "result" not in record and not record["requested_options"]


def test_exact_local_task_and_nonsignalling_certificate(tmp_path, capsys):
    _, path = _file(tmp_path, ResourceSpec("local", shared_randomness="forbidden"))
    main(["--file", path, "--witness"])
    record = json.loads(capsys.readouterr().out)
    assert record["result"]["value"] == "3/4"
    assert record["result"]["guarantee"] == "exact"
    assert record["resource"]["shared_randomness"] == "forbidden"
    assert len(record["result"]["witness"]) == 2
    _, path = _file(tmp_path, ResourceSpec("nonsignalling"))
    main(["--file", path, "--method", "ns_exact", "--witness"])
    record = json.loads(capsys.readouterr().out)
    assert record["result"]["value"] == "1"
    assert record["result"]["guarantee"] == "exact"
    assert all(isinstance(p, str) for p in record["result"]["witness"]["primal"])


def test_task_seesaw_emits_recheckable_quantum_witness(tmp_path, capsys):
    task, path = _file(tmp_path, ResourceSpec("quantum_finite", (2, 2)))
    main(["--file", path, "--method", "seesaw", "--restarts", "1", "--max-iterations", "8",
          "--seed", "7", "--tolerance", "1e-8", "--witness"])
    record = json.loads(capsys.readouterr().out)
    witness = strategy_from_dict(record["result"]["witness"])
    checked = task.evaluate(witness)
    assert checked.value == record["result"]["value"]
    assert checked.value == pytest.approx((2 + np.sqrt(2))/4, abs=1e-6)
    assert record["requested_options"]["seed"] == 7
    assert record["requested_options"]["solver_options"] == {"eps": 1e-8}
    assert record["result"]["guarantee"] == "numerical_witness"


@pytest.mark.parametrize("kind,constraint,decoder", [("local", "any", classical_from_dict),
                                                    ("quantum_finite", "product", quantum_from_dict)])
def test_network_witness_preserves_source_provenance(tmp_path, capsys, kind, constraint, decoder):
    task, path = _file(tmp_path, ResourceSpec(kind, network=SourceNetwork(2, ((1, 0),)),
                                             state_constraint=constraint, shared_randomness="forbidden"))
    main(["--file", path, "--witness"])
    record = json.loads(capsys.readouterr().out)
    witness = decoder(record["result"]["witness"])
    assert witness.network == task.resource.network
    assert float(task.evaluate(witness).value) == 0.75
    assert "network" in record["result"]["witness"]


def test_task_rejects_silent_restriction_drop_and_unsupported_cli_method(tmp_path, capsys):
    _, path = _file(tmp_path, ResourceSpec("quantum_finite", (2, 2)))
    for options in ([], ["--method", "xor"], ["--method", "state_sdp"],
                    ["--capabilities", "--witness"], ["--method", "invalid"]):
        with pytest.raises(SystemExit) as error:
            main(["--file", path, *options])
        assert error.value.code == 2
        stderr = capsys.readouterr().err
        assert "error:" in stderr
        if options == ["--method", "state_sdp"]:
            assert "measurement" in stderr


def test_task_input_and_solver_errors_are_graceful(tmp_path, capsys):
    _, path = _file(tmp_path, ResourceSpec("quantum_finite"))
    for args in (["--file", "/nonexistent/nonlocalgames-task.json"],
                 ["--file", path, "--method", "xor", "--solver", "MISSING_SOLVER"],
                 ["--file", path, "--tolerance", "nan"],
                 ["--file", path, "--restarts", "0"],
                 ["--file", path, "--seed", "-1"]):
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
        assert "error:" in capsys.readouterr().err


def test_task_routes_through_main_command_and_omits_witness_by_default(tmp_path, capsys):
    from nonlocalgames.cli import main as application_main
    _, path = _file(tmp_path, ResourceSpec("local"))
    application_main(["task", "--file", path])
    record = json.loads(capsys.readouterr().out)
    assert record["result"]["value"] == "3/4"
    assert "witness" not in record["result"]
