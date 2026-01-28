
"""Custom observation functions for exploration environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.envs.mdp import *  # noqa: F401, F403
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedRLEnv,ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

def base_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the height (z position) of the robot base.
    
    Returns:
        Height as a tensor of shape (num_envs, 1).
    """
    asset = env.scene["robot"]
    return asset.data.root_pos_w[:, 2:3]  # shape: (num_envs, 1)


def yaw_to_target(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """Get the yaw angle error to the target point (in body frame).
    
    Returns the angle between current heading and target direction.
    - 0: facing target
    - positive: target is to the left
    - negative: target is to the right
    
    Useful for:
    - Knowing when the robot is turning (yaw_error != 0)
    - Slowing down during turns
    
    Returns:
        Yaw error in radians, shape (num_envs, 1), range [-π, π]
    """
    # 获取目标点相对位置（机体坐标系）
    pose_command = env.command_manager.get_command(command_name)  # (num_envs, 3 or 4)
    
    # 获取机器人位置和姿态
    asset = env.scene["robot"]
    robot_pos = asset.data.root_pos_w[:, :3]
    robot_quat = asset.data.root_quat_w
    
    # pose_command 是世界系下的目标位置，需要转换到机体系
    # 计算世界系下的目标方向向量
    target_pos_world = pose_command[:, :3] + env.scene.env_origins  # 加上环境原点
    goal_vec_world = target_pos_world - robot_pos
    
    # 转换到机体系
    goal_vec_body = math_utils.quat_apply_inverse(robot_quat, goal_vec_world)
    
    # 计算水平方向的 yaw 角度（机体系下目标点的方向）
    yaw_error = torch.atan2(goal_vec_body[:, 1], goal_vec_body[:, 0])
    
    return yaw_error.unsqueeze(-1)  # shape: (num_envs, 1)



def pose_command_position_only(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Extract only position from pose command, ignoring heading.
    
    Args:
        env: The environment.
        command_name: The name of the command to extract position from.
        
    Returns:
        The position part of the pose command (x, y, z) without heading.
        Shape is (num_envs, 3).
    """
    full_command = env.command_manager.get_command(command_name)
    # 只返回位置部分 (前3维)，去掉朝向 (第4维)
    return full_command[:, :3]


def pose_command_position_2d(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Extract only 2D position from pose command for UGV navigation.
    
    Args:
        env: The environment.
        command_name: The name of the command to extract position from.
        
    Returns:
        The 2D position part of the pose command (x, y) for UGV.
        Shape is (num_envs, 2).
    """
    full_command = pose_command_position_only(env, command_name)
    # 只返回2D位置部分 (前2维)，忽略高度和朝向
    return full_command[:, :2]


def lidar_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """3D distance scan from the given sensor w.r.t. the sensor's frame.

    Returns the true 3D distance (ray length) from sensor to hit points.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # 3D distance: full ray length from sensor to hit point
    depth = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)

    # optional, return the min distance and corresponding index
    # min_value, min_index = torch.min(depth, keepdim=True, dim=-1)
    # return torch.cat((min_value, min_index / sensor.num_rays),dim=1)

    return depth