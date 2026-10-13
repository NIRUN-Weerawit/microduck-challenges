"""Hill walking: reward supported travel, not foot motion alone."""

from ..race_rewards import course_forward_speed, upright_mask


def supported_forward_progress(env, sensor_name, command_name):
    """Bounded, linear speed incentive while upright and supported by terrain.

    This retains a gradient when velocity tracking's Gaussian is far from its
    target. Horizontal course progress is used so descending training patches
    are not penalized for losing height. No reward for reversing, sideways
    travel, or unsupported flight.
    """
    command = env.command_manager.get_command(command_name)[:, 0]
    support = (env.scene[sensor_name].data.current_contact_time > 0).any(dim=1)
    progress = (course_forward_speed(env) / command.clamp_min(0.15)).clamp(0.0, 1.0)
    return progress * support * upright_mask(env) * (command > 0.01)
