"""Reject motion without supported forward travel in hill and roller tasks."""

from types import SimpleNamespace

import pytest
import torch

from microduck_challenges.hill_climb_1m5.rewards import supported_forward_progress
from microduck_challenges.race_rewards import alternating_landings
from microduck_challenges.roller_sprint_2m.rewards import grounded_wheel_progress


def fixture_env():
    sensor = SimpleNamespace(data=SimpleNamespace(
        current_contact_time=torch.tensor([[0.2, 0.0]]),
        last_air_time=torch.tensor([[0.15, 0.15]]),
    ), landing=torch.tensor([[False, False]]))
    sensor.compute_first_contact = lambda dt: sensor.landing
    # Deliberately shuffled wheel indices: rewards must select by name.
    ids = {"passive_LF_?wheel": 2, "passive_LR_?wheel": 0,
           "passive_RF_?wheel": 3, "passive_RR_?wheel": 1}
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
        root_link_lin_vel_w=torch.tensor([[0.3, 0., 0.]]),
        joint_vel=torch.full((1, 4), 0.3 / 0.0175),
    ), find_joints=lambda pattern: ([ids[pattern]], [pattern]))
    command = torch.tensor([[0.6, 0., 0.]])
    env = SimpleNamespace(
        num_envs=1, device="cpu", step_dt=0.02,
        scene={"robot": robot, "feet": sensor}, _heading_ref=torch.tensor([0.]),
        command_manager=SimpleNamespace(get_command=lambda name: command),
        episode_length_buf=torch.tensor([1]), extras={"log": {}},
    )
    return env, robot.data, sensor, command


@pytest.mark.parametrize("reward", [supported_forward_progress, grounded_wheel_progress])
@pytest.mark.parametrize("bad_state", ["stopped", "backward", "sideways", "fallen", "zero_command"])
def test_no_progress_reward_for_invalid_motion(reward, bad_state):
    env, robot, _, command = fixture_env()
    if bad_state == "stopped":
        robot.root_link_lin_vel_w[:] = 0
    elif bad_state == "backward":
        robot.root_link_lin_vel_w[:, 0] = -0.3
    elif bad_state == "sideways":
        robot.root_link_lin_vel_w[:] = torch.tensor([[0., 0.3, 0.]])
    elif bad_state == "fallen":
        robot.root_link_quat_w[:] = torch.tensor([[0., 1., 0., 0.]])
    else:
        command[:] = 0
    assert reward(env, sensor_name="feet", command_name="twist").item() == 0


@pytest.mark.parametrize("reward", [supported_forward_progress, grounded_wheel_progress])
def test_unsupported_motion_earns_zero(reward):
    env, _, sensor, _ = fixture_env()
    sensor.data.current_contact_time[:] = 0
    assert reward(env, sensor_name="feet", command_name="twist").item() == 0


def test_hill_progress_is_bounded_and_monotonic():
    env, robot, _, _ = fixture_env()
    assert supported_forward_progress(env, "feet", "twist").item() == pytest.approx(0.5)
    robot.root_link_lin_vel_w[:, 0] = 0.6
    assert supported_forward_progress(env, "feet", "twist").item() == 1
    robot.root_link_lin_vel_w[:, 0] = 6.0
    assert supported_forward_progress(env, "feet", "twist").item() == 1


def test_roller_unloaded_spin_cannot_increase_reward():
    env, robot, _, command = fixture_env()
    baseline = grounded_wheel_progress(env, "feet", "twist").item()
    assert baseline > 0
    robot.joint_vel[:, [1, 3]] = 10000
    assert grounded_wheel_progress(env, "feet", "twist").item() == pytest.approx(baseline)
    command[:, 0] = 2.0  # Arena high commands must not inflate reward mass.
    assert grounded_wheel_progress(env, "feet", "twist").item() == pytest.approx(baseline)
    robot.joint_vel[:, [0, 2]] = 0
    assert grounded_wheel_progress(env, "feet", "twist").item() == 0


def test_recovery_rewards_completed_alternation_not_an_airborne_hold():
    env, _, sensor, _ = fixture_env()
    reward = alternating_landings(None, env)
    params = {"sensor_name": "feet", "command_name": "twist",
              "speed_min": 0.1, "speed_max": 0.6, "cadence_min": 1.25,
              "cadence_max": 2.5, "min_swing_time": 0.1, "scale_by_push": True}
    reward.elapsed[:] = 0.38
    assert reward(env, **params).item() == 0  # No landing, no payout.
    sensor.landing[:] = torch.tensor([[True, False]])
    assert reward(env, **params).item() == 0  # First foot establishes history.
    reward.elapsed[:] = 0.38
    sensor.landing[:] = torch.tensor([[False, True]])
    assert reward(env, **params).item() == pytest.approx(6.0)
    reward.elapsed[:] = 0.38
    assert reward(env, **params).item() == 0  # Repeating the same foot.
    sensor.landing[:] = True
    assert reward(env, **params).item() == 0  # Two-foot hopping.
    reward.reset()
    assert reward.last_foot.item() == -1
