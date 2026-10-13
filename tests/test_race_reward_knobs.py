"""Hill and roller UI controls must cover every reward and survive curricula."""

import importlib
from pathlib import Path
import tomllib

import pytest

from arena.workshop.challenges import read_challenge, write_values


@pytest.mark.parametrize("folder,count", [("hill_climb_1m5", 18)])
def test_all_rewards_exposed_and_defaults_match(folder, count):
    module = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = module.make_env_cfg()
    path = Path(module.__file__).with_name("challenge.toml")
    values = module.challenge.params(module.__file__)
    assert len(module.REWARD_WEIGHT_KNOBS) == count
    assert set(module.REWARD_WEIGHT_KNOBS.values()) == set(cfg.rewards)
    assert set(values) == set(module.REWARD_WEIGHT_KNOBS) | ({"max_slope_deg"} if folder == "hill_climb_1m5" else set())
    assert len(read_challenge(path).knobs) == len(values)
    for knob, term in module.REWARD_WEIGHT_KNOBS.items():
        ramps = [c for c in cfg.curriculum.values() if c.params.get("reward_name") == term]
        if ramps:
            stages = ramps[0].params["weight_stages"]
            assert stages[-1]["weight"] == pytest.approx(values[knob])
            assert cfg.rewards[term].weight == pytest.approx(stages[0]["weight"])
        else:
            assert cfg.rewards[term].weight == pytest.approx(values[knob])


@pytest.mark.parametrize("folder", ["hill_climb_1m5"])
def test_disabling_ramped_weights_stays_disabled(folder, monkeypatch):
    module = importlib.import_module(f"microduck_challenges.{folder}.env")
    values = module.challenge.params(module.__file__)
    values["action_rate_reward"] = 0
    if folder == "hill_climb_1m5":
        values["head_pose_bias_reward"] = 0
    monkeypatch.setattr(module.challenge, "params", lambda _: values)
    cfg = module.make_env_cfg()
    for curriculum in cfg.curriculum.values():
        if "weight_stages" in curriculum.params:
            assert all(s["weight"] == 0 for s in curriculum.params["weight_stages"])
            assert cfg.rewards[curriculum.params["reward_name"]].weight == 0


@pytest.mark.parametrize("folder", ["hill_climb_1m5"])
def test_workshop_can_save_all_reward_weights(folder, tmp_path):
    module = importlib.import_module(f"microduck_challenges.{folder}.env")
    path = tmp_path / "challenge.toml"
    path.write_text(Path(module.__file__).with_name("challenge.toml").read_text())
    contract = read_challenge(path)
    values = {k.name: (k.min + k.max) / 2 for k in contract.knobs}
    write_values(contract, contract.recipe, values)
    actual = tomllib.loads(path.read_text())["params"]
    assert all(actual[name]["value"] == value for name, value in values.items())
