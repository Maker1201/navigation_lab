# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "pose_command",
    success_threshold: float = 1.0,
) -> torch.Tensor:
    """Curriculum based on whether the robot successfully reaches the target pose.

    This term is used to increase the difficulty of the terrain when the robot successfully reaches
    the target position and decrease the difficulty when the robot fails to reach the target.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Args:
        env: The environment.
        env_ids: The environment IDs to update.
        asset_cfg: The asset configuration.
        command_name: The name of the pose command.
        success_threshold: The distance threshold (in meters) to consider a target as reached.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    
    # Get the pose command (target position)
    pose_command = env.command_manager.get_command(command_name)  # (num_envs, 3 or 4)
    goal_pos = pose_command[:, :2]  # 2D position (x, y) relative to env origin
    
    # Convert goal position to world coordinates
    goal_pos_world = goal_pos + env.scene.env_origins[:, :2]
    
    # Get robot position
    robot_pos = asset.data.root_pos_w[:, :2]
    
    # Compute distance to target
    distance_to_target = torch.norm(goal_pos_world[env_ids] - robot_pos[env_ids], dim=1)
    
    # Robots that successfully reach the target (distance < threshold) progress to harder terrains
    move_up = distance_to_target < success_threshold
    
    # Robots that are far from target (distance > terrain_size/2) go to simpler terrains
    # This indicates they're struggling to navigate
    terrain_size = terrain.cfg.terrain_generator.size[0] if hasattr(terrain.cfg, "terrain_generator") else 10.0
    move_down = distance_to_target > terrain_size / 2
    move_down *= ~move_up
    
    # Update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # Return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())