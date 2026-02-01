
"""Custom observation functions for exploration environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.envs.mdp import *  # noqa: F401, F403
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation  # 实例对象，用于获取 .data
from isaaclab.assets import ArticulationCfg  # 配置对象
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


def generated_commands_relative_to_base(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """计算目标点在机器人本地坐标系下的相对位置 (x, y)。"""
    
    # 1. 获取机器人资产实例 (Articulation 实例包含实时的 .data 属性)
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 2. 获取目标点在世界坐标系下的位置 [num_envs, 2]
    target_pose_w = env.command_manager.get_command(command_name)
    target_pos_w = target_pose_w[:, :2] 
    
    # 3. 获取机器人当前的世界位置和四元数
    current_pos_w = robot.data.root_pos_w[:, :2] 
    current_quat_w = robot.data.root_quat_w      # [num_envs, 4]
    
    # 4. 计算位置差向量并扩充为 3D (x, y, 0)
    relative_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    relative_pos_w[:, :2] = target_pos_w - current_pos_w
    
    # 5. 【修正】使用 quat_apply_inverse 将向量转换到机器人本地坐标系
    # 该函数代替了旧版的 quat_rotate_inverse
    relative_pos_b = math_utils.quat_apply_inverse(current_quat_w, relative_pos_w)
    
    # 6. 返回本地坐标系下的 (x, y)
    return relative_pos_b[:, :2]

def generated_commands_polar(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """【强烈推荐】转换为极坐标 (距离, 角度)，导航训练收敛最快。"""
    
    # 获取本地坐标 (x, y)
    rel_pos = generated_commands_relative_to_base(env, command_name, asset_cfg)
    
    # 计算到目标的欧式距离 [num_envs, 1]
    dist = torch.norm(rel_pos, dim=1, keepdim=True)
    
    # 计算目标相对于当前朝向的偏角 (Heading Error) [num_envs, 1]
    # atan2 返回范围 [-pi, pi]
    angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]).unsqueeze(1)
    
    # 连接为 [距离, 角度] 传给神经网络
    return torch.cat([dist, angle], dim=1)

def lidar_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # 获取射线长度
    depth = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    # 归一化到 [0, 1]，4.0 是雷达的最大量程
    return torch.clamp(depth / 4.0, 0.0, 1.0)