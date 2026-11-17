from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import shutil
import subprocess

import pytest

from nonlocalgames.claims import VerificationBudget
from nonlocalgames.core.identity import canonical_json
from nonlocalgames.discovery import provenance
from nonlocalgames.discovery.run_manifest import RUN_SCHEMA, create_manifest, replay_budget, validate_manifest


def _rehash(data):
    definition = {k: v for k, v in data.items() if k != "run_id"}
    data["run_id"] = RUN_SCHEMA + ":" + sha256(canonical_json(definition).encode()).hexdigest()
    return data


def test_archived_manifest_binds_settings_and_replays_with_other_runtime(monkeypatch):
    saved = create_manifest(steps=10, grid_denominator=12, solver="scs", eps=1e-7)
    assert saved["solver"] == "SCS" and saved["solver_options"] == {"eps": 1e-7}
    assert saved["provenance"]["solver_distribution"] == "scs"
    assert saved["provenance"]["solver_version"]
    assert validate_manifest(json.loads(json.dumps(saved))) == saved
    for key, new in [("steps", 20), ("grid_denominator", 14), ("eps", 1e-5)]:
        changed = dict(saved, **{key: new})
        with pytest.raises(ValueError):
            validate_manifest(changed)
    import nonlocalgames.discovery.run_manifest as manifests
    different = {**saved["provenance"], "source_sha256": "a"*64, "git_revision": None}
    monkeypatch.setattr(manifests, "runtime_provenance", lambda solver: different)
    assert validate_manifest(saved) == saved
    current = create_manifest(steps=10, grid_denominator=12, solver="SCS", eps=1e-7)
    assert current["run_id"] != saved["run_id"]


def test_rehashed_manifest_cannot_redefine_grid_query_or_effective_options():
    original = create_manifest()
    for key, new in [("schema", "new"), ("steps", True), ("screen", 1), ("seed", 0),
                     ("solver_options", {}), ("grid", "p=1"), ("query", {"threshold": "1"})]:
        with pytest.raises(ValueError):
            validate_manifest(_rehash(dict(original, **{key: new})))
    changed = deepcopy(original)
    changed["verification_budget"].pop("max_artifacts")
    with pytest.raises(ValueError):
        validate_manifest(_rehash(changed))
    changed = deepcopy(original)
    changed["provenance"]["source_sha256"] = "short"
    with pytest.raises(ValueError, match="digest"):
        validate_manifest(_rehash(changed))


def test_screen_configuration_is_explicit_even_for_missing_packages():
    disabled = create_manifest(screen=False, solver="SCS")
    assert disabled["screen_method"] is None and disabled["solver_options"] == {}
    assert disabled["provenance"]["solver_version"] is None
    other = create_manifest(solver="nonexistent")
    assert other["solver_options"] == {} and other["solver"] == "NONEXISTENT"
    assert other["provenance"]["solver_version"] == "unavailable"
    assert validate_manifest(other) == other


def test_manifest_cannot_silently_widen_or_loosen_caller_verification_policy():
    default = VerificationBudget()
    saved = replace(default, max_strategies=default.max_strategies*10, max_quantum_entries=64)
    manifest = create_manifest(verification_budget=saved)
    assert replay_budget(manifest) == replace(default, max_quantum_entries=64)
    local = replace(default, max_strategies=default.max_strategies*20, max_quantum_entries=32)
    assert replay_budget(manifest, local) == replace(saved, max_quantum_entries=32)


@pytest.mark.parametrize("options", [{"steps": 0}, {"steps": 10_001}, {"grid_denominator": 1_000_001},
                                     {"solver": "/opt/foo"}, {"eps": float("inf")}, {"screen": 1}])
def test_invalid_manifest_settings_remain_outside_family(options):
    with pytest.raises(ValueError):
        create_manifest(**options)


def test_package_source_digest_uses_relative_filenames_and_contents(tmp_path):
    a, b = tmp_path / "original", tmp_path / "installed"
    for directory in (a, b):
        (directory / "sub").mkdir(parents=True)
        (directory / "__init__.py").write_bytes(b"x=1\n")
        (directory / "sub" / "ops.py").write_bytes(b"y=2\n")
    (a / "compiled.pyc").write_bytes(b"untracked bytecode")
    assert provenance._source_hash(a) == provenance._source_hash(b)
    old = (b / "sub" / "ops.py")
    old.write_bytes(b"y=3\n")
    assert provenance._source_hash(a) != provenance._source_hash(b)
    old.write_bytes(b"y=2\n")
    old.rename(b / "renamed.py")
    assert provenance._source_hash(a) != provenance._source_hash(b)


@pytest.mark.skipif(not shutil.which("git"), reason="Git metadata is optional")
def test_git_revision_resolves_in_tracked_package_not_invocation_directory(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "--quiet")
    pkg = tmp_path / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("version = 1")
    git("add", "--", "src/pkg/__init__.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "Fixture")
    head = git("rev-parse", "HEAD")
    untracked = tmp_path / "untracked"
    untracked.mkdir()
    (untracked / "__init__.py").write_text("version = 1")
    monkeypatch.chdir(untracked)
    monkeypatch.setenv("GIT_DIR", str(untracked / ".missing"))
    assert provenance._git_revision(pkg) == head
    assert provenance._git_revision(untracked) is None
