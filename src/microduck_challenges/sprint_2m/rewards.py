"""Sprint-specific rewards for completed, alternating steps."""

import math

import torch
from mjlab.managers.manager_base import ManagerTermBase


class alternating_step_cadence(ManagerTermBase):
    """Small touchdown bonus for completed alternation, without a fixed cadence.

    Require a real swing and forward progress; simultaneous landings earn zero.
    Credit actual elapsed time (capped), so faster stepping cannot multiply the
    reward budget. Speed is the main objective; the policy chooses gait timing.
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
        speed_min=0.2, min_swing_time=0.06, max_credit_time=0.5,
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

        alternating = (self.last_foot >= 0) & (foot != self.last_foot)
        robot = env.scene["robot"]
        forward_speed = robot.data.root_link_lin_vel_b[:, 0]
        progress = (forward_speed / command.clamp_min(speed_min)).clamp(0.0, 1.0)
        quat = robot.data.root_link_quat_w
        upright = (1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())) > math.cos(math.pi / 4)
        credited_time = self.elapsed.clamp(max=max_credit_time)
        reward = valid * alternating * upright * progress * credited_time / env.step_dt

        self.last_foot = torch.where(valid, foot, self.last_foot)
        self.elapsed = torch.where(valid, 0.0, self.elapsed)
        env.extras["log"]["Metrics/completed_steps_per_second"] = (valid & alternating).float().mean() / env.step_dt
        return reward


class sprint_forward_progress(ManagerTermBase):
    """Bounded course-speed reward that never penalizes exceeding the command.

    The positive command selects effort, not a target-speed ceiling. Project
    world velocity onto the spawn heading so turning sideways is not progress.
    Vertical speed is deliberately absent: running can include flight phases.
    Zero commands remain idle; fallen states cannot collect positive speed pay.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.heading = torch.zeros(env.num_envs, device=env.device)
        self.initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.initialized[ids] = False

    def __call__(self, env, command_name, speed_scale=2.0, command_max=1.2):
        robot = env.scene["robot"]
        w, x, y, z = robot.data.root_link_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        first = ~self.initialized | (env.episode_length_buf == 0)
        self.heading = torch.where(first, yaw, self.heading)
        self.initialized[:] = True
        velocity = robot.data.root_link_lin_vel_w
        forward = velocity[:, 0] * self.heading.cos() + velocity[:, 1] * self.heading.sin()
        effort = (env.command_manager.get_command(command_name)[:, 0] / command_max).clamp(0.0, 1.0)
        upright = 1 - 2 * (x.square() + y.square()) > 0.5
        env.extras["log"]["Metrics/course_forward_speed"] = forward.mean()
        return effort * torch.tanh(forward / speed_scale) * upright
