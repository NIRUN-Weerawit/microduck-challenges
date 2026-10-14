"""Cadence must reward real alternating steps, not contact chatter or falling."""

from types import SimpleNamespace

import pytest
import torch

from microduck_challenges.sprint_2m.rewards import alternating_step_cadence, sprint_forward_progress


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


def test_free_cadence_rewards_alternation_without_pay_for_faster_timing():
    reward, env, sensor, _ = setup_reward()
    assert touchdown(reward, env, sensor, 0, 0.25) == 0
    assert touchdown(reward, env, sensor, 1, 0.25) == pytest.approx(12.5)
    assert touchdown(reward, env, sensor, 1, 0.25) == 0
    short = touchdown(reward, env, sensor, 0, 0.08)
    assert short == pytest.approx(4.0)
    assert short / 0.08 == pytest.approx(12.5 / 0.25)


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


def test_slow_completed_steps_and_reset_clears_history():
    reward, env, sensor, _ = setup_reward(speed=0.2)
    touchdown(reward, env, sensor, 0, 0.5)
    assert touchdown(reward, env, sensor, 1, 0.5) == pytest.approx(25.0)
    reward.reset(torch.tensor([0]))
    assert touchdown(reward, env, sensor, 0, 0.5) == 0
    env.episode_length_buf[:] = 0
    assert touchdown(reward, env, sensor, 1, 0.5) == 0


def test_zero_command_earns_nothing_and_forgets_previous_steps():
    reward, env, sensor, _ = setup_reward(speed=0.0)
    reward.last_foot[:] = 0
    assert touchdown(reward, env, sensor, 1, 0.25) == 0
    assert reward.last_foot.item() == -1


def progress_fixture():
    _, env, _, robot = setup_reward()
    robot.data.root_link_lin_vel_w = torch.tensor([[0., 0., 0.]])
    return sprint_forward_progress(None, env), env, robot


def test_progress_increases_above_the_arena_maximum_command():
    reward, env, robot = progress_fixture()
    values = []
    for speed in [0., 0.5, 1.2, 2.2, 3.0]:
        robot.data.root_link_lin_vel_w[:, 0] = speed
        values.append(reward(env, command_name="twist").item())
    assert all(a < b for a, b in zip(values, values[1:]))
    assert 0 <= values[0] and values[-1] <= 1
    robot.data.root_link_lin_vel_w[:, 0] = -1
    assert reward(env, command_name="twist").item() < 0


def test_progress_does_not_tax_vertical_running_motion():
    reward, env, robot = progress_fixture()
    robot.data.root_link_lin_vel_w[:, 0] = 2.2
    baseline = reward(env, command_name="twist").item()
    robot.data.root_link_lin_vel_w[:, 2] = 1.5
    assert reward(env, command_name="twist").item() == pytest.approx(baseline)


def test_course_direction_is_frozen_until_reset():
    reward, env, robot = progress_fixture()
    reward(env, command_name="twist")
    robot.data.root_link_quat_w[:] = torch.tensor([[2**-0.5, 0., 0., 2**-0.5]])
    robot.data.root_link_lin_vel_w[:] = torch.tensor([[0., 2.2, 0.]])
    assert abs(reward(env, command_name="twist").item()) < 1e-6
    reward.reset(torch.tensor([0]))
    assert reward(env, command_name="twist").item() > 0


def test_progress_pays_nothing_when_fallen_or_idle():
    reward, env, robot = progress_fixture()
    robot.data.root_link_lin_vel_w[:, 0] = 2.2
    robot.data.root_link_quat_w[:] = torch.tensor([[0., 1., 0., 0.]])
    assert reward(env, command_name="twist").item() == 0
    robot.data.root_link_quat_w[:] = torch.tensor([[1., 0., 0., 0.]])
    env.command_manager.get_command = lambda _: torch.zeros(1, 3)
    assert reward(env, command_name="twist").item() == 0
