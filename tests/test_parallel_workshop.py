"""Concurrent launches retain validation, independent settings and provenance."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import threading

from fastapi import HTTPException
import pytest

from arena.workshop import app as server
from arena.workshop.challenges import read_challenge
from microduck_challenges import workshop_runtime
from mjlab_microduck import challenge, provenance


def patch_module():
    path = Path(__file__).parents[1] / "scripts/enable_parallel_training.py"
    spec = importlib.util.spec_from_file_location("parallel_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_is_idempotent_and_retains_launch_lock():
    patch = patch_module()
    root = Path(server.__file__).parent
    app = (root / "app.py").read_text().replace(patch.ALLOW, patch.GUARD)
    training = (root / "training.py").read_text().replace(patch.SNAPSHOT, patch.SPAWN).replace(patch.FROZEN_ENV, patch.CHILD_ENV)
    changed = patch.transform_sources(app, training)
    assert patch.GUARD not in changed[0]
    assert 'ws.lock.acquire(blocking=False)' in changed[0]
    assert 'if body.name in taken:' in changed[0]
    assert 'env=launch_env' in changed[1]
    assert patch.transform_sources(*changed) == changed


def test_unknown_package_version_fails_before_writes():
    with pytest.raises(ValueError):
        patch_module().transform_sources("unrecognized server", "unrecognized trainer")


def test_two_local_runs_allowed_but_duplicate_names_rejected(monkeypatch):
    root = Path(__file__).parents[1]
    contract = read_challenge(root / "src/microduck_challenges/sprint_2m/challenge.toml")
    ws = SimpleNamespace(root=root, challenges={contract.event: contract},
                         lock=threading.Lock(), trainings={})
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(workshop=ws)))
    monkeypatch.setattr(server, "find_runs", lambda *args: [])
    monkeypatch.setattr(server, "recorded_jobs", lambda *args: [])
    monkeypatch.setattr(server, "git", lambda *args: "")
    monkeypatch.setattr(server, "has_gpu", lambda: True)
    monkeypatch.setattr(server, "write_values", lambda *args: None)
    monkeypatch.setattr(server, "commit_all", lambda *args: None)
    monkeypatch.setattr(server, "ticket", lambda ws, c, name: {"name": name})
    monkeypatch.setattr(server, "Training", lambda *args: SimpleNamespace(where="local", state="running"))
    # Test the patched route even on a fresh environment with an unpatched wheel.
    source_root = Path(server.__file__).parent
    source, _ = patch_module().transform_sources(
        (source_root / "app.py").read_text(), (source_root / "training.py").read_text())
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "launch")
    node.decorator_list = []
    namespace = dict(vars(server))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "patched_launch", "exec"), namespace)
    launch = namespace["launch"]
    for name in ["first-run", "second-run"]:
        body = server.Launch(name=name, where="local", recipe=contract.recipe, params={})
        assert launch(contract.event, body, request) == {"name": name}
    assert len(ws.trainings) == 2
    with pytest.raises(HTTPException) as error:
        launch(contract.event, body, request)
    assert error.value.status_code == 409
    assert "already a run" in error.value.detail
    assert ws.lock.acquire(blocking=False)
    with pytest.raises(HTTPException) as error:
        launch(contract.event, body, request)
    assert error.value.status_code == 409
    ws.lock.release()


def test_launch_snapshot_survives_later_settings_and_commit_changes(tmp_path, monkeypatch):
    task = tmp_path / "sprint_2m/env.py"
    task.parent.mkdir()
    target = task.with_name("challenge.toml")
    target.write_text('[params.step_reward]\nvalue=3.0\n')
    metadata = {"commit": "first-commit", "branch": "experiments/test", "dirty": False}
    snapshot = {"root": str(tmp_path), "toml": str(target),
                "params": {"step_reward": 3.0}, "provenance": metadata}
    monkeypatch.setenv(workshop_runtime.SNAPSHOT_ENV, json.dumps(snapshot))
    # Record originals with monkeypatch so the runtime hooks are undone afterward.
    monkeypatch.setattr(challenge, "params", lambda path: {"step_reward": 9.0})
    monkeypatch.setattr(provenance, "checkout", lambda cwd=None: {"commit": "later-commit"})
    workshop_runtime.apply_launch_snapshot()
    target.write_text('[params.step_reward]\nvalue=5.0\n')
    assert challenge.params(task) == {"step_reward": 3.0}
    assert provenance.checkout(tmp_path) == metadata
    assert challenge.params(tmp_path / "other/env.py") == {"step_reward": 9.0}
    assert provenance.checkout(tmp_path / "other") == {"commit": "later-commit"}
