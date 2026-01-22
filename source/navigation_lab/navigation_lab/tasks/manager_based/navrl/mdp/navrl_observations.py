import torch

def goal_relative_position(env):
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    goal_pos = env.scene["goal"].data.root_pos_w[:, :2]
    return goal_pos - robot_pos

def base_velocity_xy(env):
    return env.scene["robot"].data.root_lin_vel_w[:, :2]

def raycast_distances(env):
    return env.scene.sensors["raycast"].data.distances

def dynamic_obstacle_relative_state(env):
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    robot_vel = env.scene["robot"].data.root_lin_vel_w[:, :2]

    obs_pos = env.scene["obstacle"].data.root_pos_w[:, :2]
    obs_vel = env.scene["obstacle"].data.root_lin_vel_w[:, :2]

    rel_pos = obs_pos - robot_pos
    rel_vel = obs_vel - robot_vel

    return torch.cat([rel_pos, rel_vel], dim=-1)
