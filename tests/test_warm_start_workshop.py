"""The workshop selects an existing local checkpoint for a fresh training."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
from fastapi import HTTPException

from arena.workshop import app as server
from arena.workshop import training
from arena.workshop.challenges import read_challenge


ROOT = Path(__file__).parents[1]


def installer():
    spec = importlib.util.spec_from_file_location("warm_start_installer", ROOT / "scripts/enable_warm_start.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installer_is_idempotent_and_rejects_unknown_sources():
    patch = installer()
    root = Path(server.__file__).parents[1]
    paths = [root / "workshop/app.py", root / "workshop/training.py",
             root / "frontend/assets/workshop-CIygtbAu.js", root / "frontend/workshop.html"]
    transforms = [patch.transform_app, patch.transform_training,
                  patch.transform_frontend, patch.transform_html]
    for path, transform in zip(paths, transforms, strict=True):
        source = path.read_text()
        assert transform(source) == source
    with pytest.raises(ValueError):
        patch.transform_app("unsupported version")


def test_checkpoint_selection_is_exact_and_local(monkeypatch):
    contract = read_challenge(ROOT / "src/microduck_challenges/sprint_2m/challenge.toml")
    ws = SimpleNamespace(root=ROOT, challenges={contract.event: contract},
                         lock=threading.Lock(), trainings={})
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(workshop=ws)))
    source = SimpleNamespace(id="2026-10-14_13-41-28_com5_step3_ablation",
                             name="com5-step3", checkpoints=(750, 1000))
    launches = []
    monkeypatch.setattr(server, "find_runs", lambda *args: [source])
    monkeypatch.setattr(server, "recorded_jobs", lambda *args: [])
    monkeypatch.setattr(server, "git", lambda *args: "")
    monkeypatch.setattr(server, "has_gpu", lambda: True)
    monkeypatch.setattr(server, "write_values", lambda *args: None)
    monkeypatch.setattr(server, "commit_all", lambda *args: None)
    monkeypatch.setattr(server, "ticket", lambda ws, c, name: {"name": name})
    monkeypatch.setattr(server, "Training", lambda *args: launches.append(args) or SimpleNamespace(where="local", state="running"))
    body = server.Launch(name="warm-try", where="local", recipe=contract.recipe,
                         warm_start={"run_id": source.id, "checkpoint": 750})
    assert server.launch(contract.event, body, request) == {"name": "warm-try"}
    assert launches[0][-1] == (source.id, 750)

    body.name = "wrong-checkpoint"
    body.warm_start.checkpoint = 751
    with pytest.raises(HTTPException) as error:
        server.launch(contract.event, body, request)
    assert error.value.status_code == 400

    body.name = "cloud-try"
    body.where = "hf-jobs"
    body.warm_start.checkpoint = 750
    with pytest.raises(HTTPException) as error:
        server.launch(contract.event, body, request)
    assert error.value.status_code == 400


def test_warm_command_chooses_one_checkpoint_and_fresh_schedule():
    contract = read_challenge(ROOT / "src/microduck_challenges/sprint_2m/challenge.toml")
    selected = ("2026-10-14_13-41-28_com5_step3_ablation", 1000)
    command = training.train_command(contract, "warm-try", contract.recipe, None, selected)
    assert command[command.index("--agent.resume") + 1] == "True"
    assert command[command.index("--agent.load-run") + 1] == "^2026\\-10\\-14_13\\-41\\-28_com5_step3_ablation$"
    assert command[command.index("--agent.load-checkpoint") + 1] == r"^model_1000\.pt$"
    assert "--agent.max-iterations" in command
