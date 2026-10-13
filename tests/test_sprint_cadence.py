"""Cadence must reward real alternating steps, not contact chatter or falling."""

from types import SimpleNamespace

import pytest
import torch

from microduck_challenges.sprint_2m.rewards import alternating_step_cadence


def setup_reward(speed=1.2):
    sensor = SimpleNamespace(data=SimpleNamespace(last_air_time=torch.full((1, 2), 0.1)))
    sensor.landing = torch.tensor([[False, False]])
    sensor.compute_first_contact = lambda dt: sensor.landing
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_b=torch.tensor([[speed, 0., 0.]]),
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
    ))
    env = SimpleNamespace(
        num_envs=1, device="cpu", step_dt=0.02,
        scene={"feet": sensor, "robot": robot}, episode_length_buf=torch.tensor([1]),
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[speed, 0., 0.]])),
        extras={"log": {}},
    )
    return alternating_step_cadence(None, env), env, sensor, robot


def touchdown(reward, env, sensor, foot, elapsed):
    reward.elapsed[:] = elapsed - env.step_dt
    sensor.landing[:] = False
    sensor.landing[:, foot] = True
    return reward(env, sensor_name="feet", command_name="twist").item()


def test_target_timing_rewards_alternation_and_rejects_repeated_foot():
    reward, env, sensor, _ = setup_reward()
    assert touchdown(reward, env, sensor, 0, 0.25) == 0
    assert touchdown(reward, env, sensor, 1, 0.25) == pytest.approx(12.5)
    assert touchdown(reward, env, sensor, 1, 0.25) == 0
    assert touchdown(reward, env, sensor, 0, 0.08) < 1.0


@pytest.mark.parametrize("bad_state", ["stationary", "backward", "fallen", "chatter", "both_feet"])
def test_exploits_earn_zero(bad_state):
    reward, env, sensor, robot = setup_reward()
    touchdown(reward, env, sensor, 0, 0.25)
    if bad_state == "stationary":
        robot.data.root_link_lin_vel_b[:, 0] = 0
    elif bad_state == "backward":
        robot.data.root_link_lin_vel_b[:, 0] = -1
    elif bad_state == "fallen":
        robot.data.root_link_quat_w[:] = torch.tensor([[0., 1., 0., 0.]])
    elif bad_state == "chatter":
        sensor.data.last_air_time[:] = 0.02
    else:
        sensor.landing[:] = True
        assert reward(env, sensor_name="feet", command_name="twist").item() == 0
        return
    assert touchdown(reward, env, sensor, 1, 0.25) == 0


def test_target_cadence_increases_with_speed_and_reset_clears_history():
    reward, env, sensor, _ = setup_reward(speed=0.2)
    touchdown(reward, env, sensor, 0, 0.5)
    assert touchdown(reward, env, sensor, 1, 0.5) == pytest.approx(25.0)
    assert env.extras["log"]["Metrics/target_step_cadence"].item() == 2.0
    reward.reset(torch.tensor([0]))
    assert touchdown(reward, env, sensor, 0, 0.5) == 0
    env.episode_length_buf[:] = 0
    assert touchdown(reward, env, sensor, 1, 0.5) == 0


def test_zero_command_earns_nothing_and_forgets_previous_steps():
    reward, env, sensor, _ = setup_reward(speed=0.0)
    reward.last_foot[:] = 0
    assert touchdown(reward, env, sensor, 1, 0.25) == 0
    assert reward.last_foot.item() == -1
