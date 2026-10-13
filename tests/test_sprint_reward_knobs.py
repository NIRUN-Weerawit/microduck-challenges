"""All sprint weights exposed by the workshop must affect training."""

from pathlib import Path
import tomllib

import pytest

from arena.workshop.challenges import read_challenge, write_values
from microduck_challenges.sprint_2m import env


def test_every_reward_has_a_workshop_knob():
    cfg = env.make_env_cfg()
    table = tomllib.loads(Path(env.__file__).with_name("challenge.toml").read_text())
    assert set(env.REWARD_WEIGHT_KNOBS.values()) == set(cfg.rewards)
    assert set(env.REWARD_WEIGHT_KNOBS) == set(table["params"])


@pytest.mark.parametrize("knob", env.REWARD_WEIGHT_KNOBS)
def test_each_workshop_knob_controls_its_weight(knob, monkeypatch):
    values = env.challenge.params(env.__file__)
    values[knob] = 0.73 if values[knob] >= 0 else -0.73
    monkeypatch.setattr(env.challenge, "params", lambda _: values)
    cfg = env.make_env_cfg()
    reward = cfg.rewards[env.REWARD_WEIGHT_KNOBS[knob]]
    if knob == "action_rate_reward":
        stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
        assert stages[-1]["weight"] == values[knob]
        assert reward.weight == pytest.approx(values[knob] * 0.2)
    else:
        assert reward.weight == values[knob]


def test_zero_smoothing_stays_disabled_through_curriculum(monkeypatch):
    values = env.challenge.params(env.__file__)
    values["action_rate_reward"] = 0.0
    monkeypatch.setattr(env.challenge, "params", lambda _: values)
    cfg = env.make_env_cfg()
    assert cfg.rewards["action_rate_l2"].weight == 0
    assert all(s["weight"] == 0 for s in cfg.curriculum["action_rate_weight"].params["weight_stages"])


def test_workshop_accepts_signed_penalty_values(tmp_path):
    path = tmp_path / "challenge.toml"
    path.write_text(Path(env.__file__).with_name("challenge.toml").read_text())
    contract = read_challenge(path)
    assert len(contract.knobs) == 15
    write_values(contract, contract.recipe, {"action_rate_reward": -0.7, "body_ang_vel_reward": -0.1})
    updated = tomllib.loads(path.read_text())
    assert updated["params"]["action_rate_reward"]["value"] == -0.7
    assert updated["params"]["body_ang_vel_reward"]["value"] == -0.1
