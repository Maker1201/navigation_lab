"""Custom reward functions for exploration environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_distance_progress(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """Reward for making progress towards the goal.
    
    Args:
        env: The environment.
        command_name: The name of the command containing the goal position.
        
    Returns:
        The progress reward (positive when getting closer to goal).
    """
    # Get goal position from command
    pose_command = env.command_manager.get_command(command_name)  # (num_envs, 3 or 4)
    goal_pos = pose_command[:, :2]  # 2D position (x, y) relative to env origin
    
    # Get robot position
    asset = env.scene["robot"]
    robot_pos = asset.data.root_pos_w[:, :2]
    
    # Convert goal position to world coordinates
    goal_pos_world = goal_pos + env.scene.env_origins[:, :2]
    
    # Calculate current distance
    curr_dist = torch.norm(goal_pos_world - robot_pos, dim=-1)
    
    # Get previous distance from extras (initialize if not exists)
    if "prev_goal_dist" not in env.extras:
        env.extras["prev_goal_dist"] = curr_dist.clone()
    
    prev_dist = env.extras["prev_goal_dist"]
    
    # Update previous distance
    env.extras["prev_goal_dist"] = curr_dist
    
    # Reward is the reduction in distance (positive when getting closer)
    return prev_dist - curr_dist


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


def contact_force_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for contact forces (collision penalty).
    
    Args:
        env: The environment.
        
    Returns:
        The contact force penalty (positive when in contact).
    """
    asset = env.scene["robot"]
    # Check if robot has any contact
    if hasattr(asset.data, "has_contact"):
        return asset.data.has_contact.float()
    else:
        # Fallback: return zero if contact detection not available
        return torch.zeros(env.num_envs, device=env.device)
