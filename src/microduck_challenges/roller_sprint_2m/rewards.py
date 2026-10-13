"""Skating rewards that require grounded rolling and useful forward travel."""

import torch

from ..race_rewards import course_forward_speed, upright_mask


def grounded_wheel_progress(
    env, sensor_name, command_name, wheel_radius=0.0175, vel_scale=0.3,
):
    """Pay only for grounded wheel spin accompanied by forward body travel.

    Unloaded wheels do not contribute. The slower of grounded wheel speed and
    body progress determines the reward, so spinning wheels on the spot or
    sideways cannot increase it. Push intensity is capped at the trained 0.6.
    """
    robot = env.scene["robot"]
    speeds = []
    for patterns in (("passive_LF_?wheel", "passive_LR_?wheel"),
                     ("passive_RF_?wheel", "passive_RR_?wheel")):
        ids = [robot.find_joints(pattern)[0][0] for pattern in patterns]
        speeds.append(robot.data.joint_vel[:, ids].mean(dim=1) * wheel_radius)
    wheel_speed = torch.stack(speeds, dim=1).clamp_min(0.0)
    contact = env.scene[sensor_name].data.current_contact_time > 0
    grounded_speed = (wheel_speed * contact).sum(dim=1) / contact.sum(dim=1).clamp_min(1)
    forward = course_forward_speed(env).clamp_min(0.0)
    useful_speed = torch.minimum(grounded_speed, forward)
    command = env.command_manager.get_command(command_name)[:, 0].clamp(0.0, 0.6)
    return command * torch.tanh(useful_speed / vel_scale) * upright_mask(env)
