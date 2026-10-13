"""Sprint-specific rewards for completed, alternating steps."""

import math

import torch
from mjlab.managers.manager_base import ManagerTermBase


class alternating_step_cadence(ManagerTermBase):
    """Pay at touchdown for alternation and speed-dependent step timing.

    Cadence counts BOTH feet: 2--4 landings/s over the Arena's speed range.
    Require a real swing and forward progress; simultaneous landings earn zero.
    Interval scaling keeps reward per second comparable across target cadences
    despite RewardManager's dt scaling. No phase is added to observations.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.last_foot = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.elapsed = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.last_foot[ids] = -1
        self.elapsed[ids] = 0.0

    def __call__(
        self, env, sensor_name, command_name, command_threshold=0.01,
        speed_min=0.2, speed_max=1.2, cadence_min=2.0, cadence_max=4.0,
        timing_std=0.35, min_swing_time=0.06,
    ):
        sensor = env.scene[sensor_name]
        command = env.command_manager.get_command(command_name)[:, 0]
        active = command > command_threshold
        # Standing and episode resets must not carry step history into a sprint.
        self.reset(torch.nonzero(~active | (env.episode_length_buf == 0)).flatten())
        self.elapsed += env.step_dt
        landing = sensor.compute_first_contact(dt=env.step_dt)
        valid_swing = sensor.data.last_air_time >= min_swing_time
        single_landing = landing.sum(dim=1) == 1
        foot = landing.long().argmax(dim=1)
        valid = single_landing & (landing & valid_swing).any(dim=1) & active

        speed_fraction = ((command - speed_min) / (speed_max - speed_min)).clamp(0.0, 1.0)
        target_cadence = cadence_min + speed_fraction * (cadence_max - cadence_min)
        target_interval = target_cadence.reciprocal()
        timing = torch.exp(-((self.elapsed / target_interval - 1.0) / timing_std).square())
        alternating = (self.last_foot >= 0) & (foot != self.last_foot)
        robot = env.scene["robot"]
        forward_speed = robot.data.root_link_lin_vel_b[:, 0]
        progress = (forward_speed / command.clamp_min(speed_min)).clamp(0.0, 1.0)
        quat = robot.data.root_link_quat_w
        upright = (1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())) > math.cos(math.pi / 4)
        reward = valid * alternating * upright * progress * timing * target_interval / env.step_dt

        self.last_foot = torch.where(valid, foot, self.last_foot)
        self.elapsed = torch.where(valid, 0.0, self.elapsed)
        env.extras["log"]["Metrics/target_step_cadence"] = target_cadence[active].sum() / active.sum().clamp_min(1)
        return reward
