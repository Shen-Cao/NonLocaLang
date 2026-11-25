import json

import pytest

from nonlocalgames import AnalysisTask, NetworkCompatibilityTask, ResourceSpec, Scenario
from nonlocalgames.core.compatibility_serialization import compatibility_to_dict, load_compatibility
from nonlocalgames.core.json_file import read_json
from nonlocalgames.core.resource_serialization import load_task, task_to_dict
from nonlocalgames.core.serialization import load, to_dict
from nonlocalgames.games import chsh
from nonlocalgames.networks import RationalBehavior, SourceNetwork


@pytest.mark.parametrize('encoding', ['utf-8-sig', 'utf-16', 'utf-16-be', 'utf-16-le',
                                      'utf-32', 'utf-32-be', 'utf-32-le'])
def test_encoding_compatibility_is_opt_in_and_budget_counts_original_bytes(tmp_path, encoding):
    path = tmp_path / 'data.json'
    data = '{"name":"réseau"}'.encode(encoding)
    path.write_bytes(data)
    assert read_json(path, max_file_bytes=len(data), detect_encoding=True) == {'name': 'réseau'}
    with pytest.raises(ValueError, match='max_file_bytes'):
        read_json(path, max_file_bytes=len(data)-1, detect_encoding=True)
    with pytest.raises(ValueError):
        read_json(path)


@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity'])
def test_nonstandard_constants_are_rejected_in_both_file_modes(tmp_path, token):
    path = tmp_path / 'data.json'
    path.write_text('{"samples":[' + token + ']}', encoding='utf-8')
    for detect in (False, True):
        with pytest.raises(ValueError, match='Invalid JSON constant'):
            read_json(path, detect_encoding=detect)


def test_decoder_recursion_failure_is_a_value_error(tmp_path, monkeypatch):
    # Decoder recursion thresholds vary across Python builds.
    def reject_nesting(*args, **kwargs):
        raise RecursionError

    monkeypatch.setattr(json, 'loads', reject_nesting)
    path = tmp_path / 'nested.json'
    path.write_text('[[]]', encoding='utf-8')
    with pytest.raises(ValueError, match='nesting'):
        read_json(path)


def test_public_file_codecs_keep_encodings_and_limits_but_reject_duplicate_fields(tmp_path):
    game = chsh()
    task = AnalysisTask(game, ResourceSpec('local'))
    graph = SourceNetwork(2, ((0, 1),))
    network = NetworkCompatibilityTask(RationalBehavior(Scenario((1, 1), (1, 1)), (1,)),
        ResourceSpec('local', network=graph, shared_randomness='forbidden'))
    path = tmp_path / 'document.json'
    for obj, encode, read in ((game, to_dict, load), (task, task_to_dict, load_task),
                              (network, compatibility_to_dict, load_compatibility)):
        text = json.dumps(encode(obj))
        wire = text.encode('utf-16')
        path.write_bytes(wire)
        assert read(path, max_file_bytes=len(wire)) == obj
        with pytest.raises(ValueError, match='max_file_bytes'):
            read(path, max_file_bytes=len(wire)-1)

        # Each document has a top-level schema and nested scenario alphabet.
        for key in ('schema', 'input_sizes'):
            quoted = json.dumps(key) + ':'
            path.write_bytes(text.replace(quoted, quoted + 'null,' + quoted, 1).encode('utf-16'))
            with pytest.raises(ValueError, match='Duplicate JSON key'):
                read(path)
