"""Custom reward functions for exploration environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
from .observations import generated_commands_relative_to_base

def goal_distance_progress(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """计算向目标靠近的进度奖励（基于相对坐标）"""
    # 1. 获取目标相对于机器人的位置（局部坐标系）
    rel_pos = generated_commands_relative_to_base(env, command_name)
    
    # 2. 计算当前距离（只考虑xy平面，忽略高度）
    curr_dist = torch.norm(rel_pos[:, :2], dim=-1)
    
    # 3. 初始化缓存
    if "prev_goal_dist" not in env.extras:
        env.extras["prev_goal_dist"] = curr_dist.clone()
    
    # 4. 获取上一帧距离并计算进度
    prev_dist = env.extras["prev_goal_dist"]
    
    # 5. 【关键】重置保护：新回合第一步不计算奖励，避免跳变
    reset_envs = env.episode_length_buf == 0
    prev_dist[reset_envs] = curr_dist[reset_envs]
    
    # 6. 计算奖励（靠近目标为正，远离为负）
    reward = prev_dist - curr_dist
    
    # 7. 限制范围，防止异常值
    reward = torch.clamp(reward, min=-0.1, max=0.1)
    
    # 8. 更新缓存供下一帧使用
    env.extras["prev_goal_dist"] = curr_dist.clone()
    
    return reward


def heading_to_goal(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """Reward for aligning heading with the goal direction.
    
    Args:
        env: The environment.
        command_name: The name of the command containing the goal position.
        
    Returns:
        The heading alignment reward (higher when facing the goal).
    """
    # Get goal position from command
    pose_command = env.command_manager.get_command(command_name)  # (num_envs, 3 or 4)
    goal_pos = pose_command[:, :2]  # 2D position (x, y)
    
    # Get robot position and orientation
    asset = env.scene["robot"]
    robot_pos = asset.data.root_pos_w[:, :2]
    robot_quat = asset.data.root_quat_w
    
    # Calculate direction to goal in world frame
    # Note: pose_command is relative to env origin, so we need to add it
    goal_pos_world = goal_pos + env.scene.env_origins[:, :2]
    goal_vec_world = goal_pos_world - robot_pos
    
    # Transform to body frame (need 3D vector for quaternion rotation)
    goal_vec_world_3d = torch.cat([goal_vec_world, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)
    goal_vec_body = math_utils.quat_apply_inverse(robot_quat, goal_vec_world_3d)
    
    # Calculate yaw angle to goal in body frame
    yaw_to_goal = torch.atan2(goal_vec_body[:, 1], goal_vec_body[:, 0])
    
    # Reward is cosine of the angle (1 when facing goal, 0 when perpendicular)
    return torch.cos(yaw_to_goal)

 
def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:  # 不期望接触惩罚函数
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    """惩罚不期望的接触，作为超过阈值的违规数量。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 从场景中获取接触传感器
    # check if contact force is above threshold
    # 检查接触力是否超过阈值
    net_contact_forces = contact_sensor.data.net_forces_w_history  # 获取接触力历史（世界坐标系）
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold  # 判断是否接触：接触力历史的最大范数大于阈值（布尔张量，仅针对配置中指定的身体）
    # sum over contacts for each environment
    # 对每个环境的接触求和
    reward = torch.sum(is_contact, dim=1).float()  # 计算奖励：每个环境中超过阈值的接触数量（在身体维度上求和并转换为浮点数）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示不期望的接触越多）


def reach_goal_bonus(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    command_name: str = "pose_command"
) -> torch.Tensor:
    """
    当机器人到达目标点范围内时给予奖励（稀疏奖励）。
    
    Args:
        env: 环境对象
        threshold: 判定到达的距离阈值 (米)
        command_name: 包含目标位置的指令名称
        
    Returns:
        奖励张量：到达的为 1.0，未到达的为 0.0
    """
    # 1. 获取目标位置 (相对于环境原点)
    pose_command = env.command_manager.get_command(command_name)
    goal_pos_local = pose_command[:, :2]  # (num_envs, 2)
    
    # 2. 获取机器人位置 (世界坐标)
    asset = env.scene["robot"]
    robot_pos_world = asset.data.root_pos_w[:, :2]
    
    # 3. 将目标位置转换为世界坐标
    # 注意：必须加上 env_origins 才能和 robot_pos_world 进行比较
    goal_pos_world = goal_pos_local + env.scene.env_origins[:, :2]
    
    # 4. 计算欧几里得距离
    # dim=-1 表示在 (x,y) 维度上求范数
    distance = torch.norm(goal_pos_world - robot_pos_world, dim=-1)
    
    # 5. 判断是否在阈值内
    is_reached = distance <= threshold
    
    # 6. 返回奖励 (将布尔值转换为浮点数: True->1.0, False->0.0)
    return is_reached.float()


def lidar_proximity_penalty(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float = 0.5
) -> torch.Tensor:
    """
    基于雷达数据的避障惩罚：当障碍物距离小于阈值时，距离越近惩罚越大。
    """
    # 1. 获取雷达传感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 2. 获取射线击中点和传感器位置
    #   ray_hits_w:   [num_envs, num_rays, 3]
    #   pos_w:        [num_envs, 3]
    ray_hits = sensor.data.ray_hits_w
    ray_origins = sensor.data.pos_w.unsqueeze(1)  # [num_envs, 1, 3]，广播到每条射线
    
    # 3. 计算每条射线的距离
    dist_to_hit = torch.norm(ray_hits - ray_origins, dim=-1)  # [num_envs, num_rays]
    
    # 4. 计算接近惩罚：threshold - dist，越近惩罚越大，远于阈值则为 0
    proximity_error = torch.clamp(threshold - dist_to_hit, min=0.0)
    
    # 5. 对每个环境取“最近障碍物”的惩罚（最大值）
    max_penalty_per_env = torch.max(proximity_error, dim=1)[0]
    
    return max_penalty_per_env