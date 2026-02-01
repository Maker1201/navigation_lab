"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.managers import SceneEntityCfg
from .observations import generated_commands_relative_to_base

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # Return a per-env boolean tensor of False (no termination on plane)
        _asset: RigidObject = env.scene[asset_cfg.name]
        return torch.zeros((_asset.data.root_pos_w.shape[0],), dtype=torch.bool, device=_asset.data.root_pos_w.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def out_of_height_limit(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    min_height: float = 1.0,
    max_height: float = 5.0
) -> torch.Tensor:
    """Terminate when the actor is outside the height limits.

    Args:
        env: The environment object.
        asset_cfg: The asset configuration.
        min_height: Minimum allowed height (m).
        max_height: Maximum allowed height (m).
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # check if the agent is out of height bounds
    z = asset.data.root_pos_w[:, 2]
    too_low = z < min_height
    too_high = z > max_height
    return torch.logical_or(too_low, too_high)


def least_lidar_depth(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the minimum lidar depth is below threshold (3D distance)."""
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # 3D distance: full ray length from sensor to hit point
    depth = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    return torch.any(depth < threshold, dim=1)



def roll_over(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.1
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # check if the agent is out of bounds
    is_roll_over = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) > threshold
    return is_roll_over

def reach_target(env: ManagerBasedRLEnv, threshold: float, command_name: str) -> torch.Tensor:
    # 修正：调用现有的相对坐标计算函数
    rel_pos = generated_commands_relative_to_base(env, command_name)
    
    distance = torch.norm(rel_pos, dim=1)
    return distance <= threshold