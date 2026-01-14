from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用前向引用

import torch  # 导入PyTorch库，用于张量操作和数学计算
from typing import TYPE_CHECKING  # 导入TYPE_CHECKING，用于类型检查时的条件导入

if TYPE_CHECKING:  # 仅在类型检查时执行（运行时不会执行）
    from isaaclab.envs import ManagerBasedEnv  # 导入基于管理器的环境类（仅用于类型注解）


def _get_terrain_column_range(terrain_cfg, terrain_name: str, device) -> tuple[int, int] | None:  # 获取地形列范围的辅助函数
    """Helper function to calculate column range for a terrain type.

    Args:
        terrain_cfg: The terrain generator configuration.
        terrain_name: Name of the terrain.
        device: Torch device.

    Returns:
        Tuple of (col_start, col_end) or None if terrain not found.
    """
    """计算地形类型的列范围的辅助函数。

    参数：
        terrain_cfg: 地形生成器配置。
        terrain_name: 地形名称。
        device: PyTorch设备。

    返回：
        (col_start, col_end) 元组，如果未找到地形则返回None。
    """
    if terrain_cfg.sub_terrains is None or terrain_name not in terrain_cfg.sub_terrains:  # 如果子地形配置为None或地形名称不在子地形中
        return None  # 返回None（未找到地形）

    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())  # 获取所有子地形名称列表
    proportions = torch.tensor([sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()], device=device)  # 创建比例张量：从每个子地形配置中提取proportion值（在指定设备上）
    proportions = proportions / proportions.sum()  # 归一化比例：将比例除以总和（确保总和为1）
    cumsum_props = torch.cumsum(proportions, dim=0)  # 计算累积和：在维度0上计算比例的累积和（用于确定每个地形的列范围）

    terrain_idx = sub_terrain_names.index(terrain_name)  # 获取指定地形名称在列表中的索引
    # Use round() instead of int() to properly allocate columns
    # 使用round()而不是int()来正确分配列
    col_start = round((0.0 if terrain_idx == 0 else cumsum_props[terrain_idx - 1].item()) * terrain_cfg.num_cols)  # 计算列起始索引：如果是第一个地形则为0，否则为前一个地形的累积和乘以总列数，然后四舍五入
    col_end = round(cumsum_props[terrain_idx].item() * terrain_cfg.num_cols)  # 计算列结束索引：当前地形的累积和乘以总列数，然后四舍五入

    return (col_start, col_end)  # 返回列范围元组（起始索引，结束索引）


def is_env_assigned_to_terrain(env: ManagerBasedEnv, terrain_name: str) -> torch.Tensor:  # 检查环境是否分配给指定地形类型的函数
    """Check which environments are initially assigned to the specified terrain type.

    Each environment is assigned to a specific terrain cell at initialization.
    This function returns a mask indicating which environments were assigned to the given terrain type.

    Args:
        env: The environment instance.
        terrain_name: Name of the terrain to check (e.g., "pits", "stairs").

    Returns:
        Boolean tensor of shape (num_envs,) where True means the environment is assigned to this terrain.
    """
    """检查哪些环境在初始化时被分配给指定的地形类型。

    每个环境在初始化时被分配到一个特定的地形单元格。
    此函数返回一个掩码，指示哪些环境被分配给了给定的地形类型。

    参数：
        env: 环境实例。
        terrain_name: 要检查的地形名称（例如，"pits"、"stairs"）。

    返回：
        形状为(num_envs,)的布尔张量，其中True表示环境被分配到此地形。
    """
    # Check if terrain and terrain generator are available
    # 检查地形和地形生成器是否可用
    terrain = getattr(env.scene, "terrain", None)  # 从场景中获取terrain属性，如果不存在则返回None
    if terrain is None or not hasattr(terrain, "terrain_types"):  # 如果地形为None或没有terrain_types属性
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # 返回全零布尔张量（所有环境，布尔类型，在指定设备上）
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:  # 如果地形类型不是"generator"或地形生成器为None
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # 返回全零布尔张量（所有环境，布尔类型，在指定设备上）

    terrain_cfg = terrain.cfg.terrain_generator  # 获取地形生成器配置
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)  # 调用辅助函数获取指定地形的列范围
    if col_range is None:  # 如果列范围为None（未找到地形）
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # 返回全零布尔张量（所有环境，布尔类型，在指定设备上）

    col_start, col_end = col_range  # 解包列范围：起始索引和结束索引
    # terrain_types directly stores column indices, so just check if they're in range
    # terrain_types直接存储列索引，所以只需检查它们是否在范围内
    return (terrain.terrain_types >= col_start) & (terrain.terrain_types < col_end)  # 返回布尔掩码：地形类型列索引大于等于起始索引且小于结束索引（使用位与操作）


def is_robot_on_terrain(env: ManagerBasedEnv, terrain_name: str, asset_name: str = "robot") -> torch.Tensor:  # 检查机器人是否在指定地形类型上的函数
    """Check which robots are currently standing on the specified terrain type.

    This function calculates which terrain grid cell each robot is on based on its world position,
    then checks if that cell's terrain type matches the specified terrain.

    Args:
        env: The environment instance.
        terrain_name: Name of the terrain to check (e.g., "pits", "stairs").
        asset_name: Name of the robot asset. Defaults to "robot".

    Returns:
        Boolean tensor of shape (num_envs,) where True means the robot is currently on this terrain.
    """
    """检查哪些机器人当前站在指定的地形类型上。

    此函数根据机器人的世界位置计算每个机器人在哪个地形网格单元格上，
    然后检查该单元格的地形类型是否与指定的地形匹配。

    参数：
        env: 环境实例。
        terrain_name: 要检查的地形名称（例如，"pits"、"stairs"）。
        asset_name: 机器人资产名称。默认为"robot"。

    返回：
        形状为(num_envs,)的布尔张量，其中True表示机器人当前在此地形上。
    """
    # Check if terrain and terrain generator are available
    # 检查地形和地形生成器是否可用
    terrain = getattr(env.scene, "terrain", None)  # 从场景中获取terrain属性，如果不存在则返回None
    if terrain is None or not hasattr(terrain, "terrain_types"):  # 如果地形为None或没有terrain_types属性
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # 返回全零布尔张量（所有环境，布尔类型，在指定设备上）
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:  # 如果地形类型不是"generator"或地形生成器为None
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # 返回全零布尔张量（所有环境，布尔类型，在指定设备上）

    terrain_cfg = terrain.cfg.terrain_generator  # 获取地形生成器配置
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)  # 调用辅助函数获取指定地形的列范围
    if col_range is None:  # 如果列范围为None（未找到地形）
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # 返回全零布尔张量（所有环境，布尔类型，在指定设备上）

    col_start, col_end = col_range  # 解包列范围：起始索引和结束索引

    # Get robot positions in world frame
    # 获取机器人在世界坐标系中的位置
    asset = env.scene[asset_name]  # 从场景中获取资产对象（机器人）
    robot_pos_w = asset.data.root_pos_w[:, :2]  # [num_envs, 2] (x, y)  # 获取根位置在世界坐标系中的前两个维度（x, y坐标）

    # Get terrain grid information
    # 获取地形网格信息
    terrain_origins = terrain.terrain_origins  # [num_rows, num_cols, 3]  # 获取地形原点（行数，列数，3个维度）
    num_rows, num_cols, _ = terrain_origins.shape  # 解包地形原点的形状：行数、列数、维度数（忽略第三个维度）

    # Use terrain_origins to directly compute which cell each robot is in
    # 使用terrain_origins直接计算每个机器人在哪个单元格中
    # terrain_origins[r, c, :2] is the center of cell (r, c)
    # terrain_origins[r, c, :2]是单元格(r, c)的中心
    # We need to find the closest terrain origin for each robot
    # 我们需要为每个机器人找到最近的地形原点

    # Reshape terrain_origins for distance calculation
    # 重塑terrain_origins以便计算距离
    terrain_origins_2d = terrain_origins[:, :, :2].reshape(num_rows * num_cols, 2)  # [num_rows*num_cols, 2]  # 将地形原点重塑为2D：提取前两个维度（x, y）并重塑为（行数*列数，2）的形状

    # Calculate distances from each robot to all terrain origins
    # 计算从每个机器人到所有地形原点的距离
    distances = torch.cdist(robot_pos_w, terrain_origins_2d)  # [num_envs, num_rows*num_cols]  # 计算成对距离：使用cdist函数计算机器人位置与地形原点之间的欧几里得距离

    # Find the closest terrain origin for each robot
    # 为每个机器人找到最近的地形原点
    closest_flat_idx = torch.argmin(distances, dim=1)  # [num_envs]  # 找到最近的地形原点索引：在距离张量的维度1上找到最小值的索引（扁平索引）

    # Convert flat index to column index
    # 将扁平索引转换为列索引
    # flat_idx = row * num_cols + col
    # 扁平索引 = 行 * 列数 + 列
    col_idx = closest_flat_idx % num_cols  # [num_envs]  # 计算列索引：扁平索引对列数取模（扁平索引除以列数的余数即为列索引）

    # Check if the robot's current terrain column is in the specified terrain's range
    # 检查机器人当前的地形列是否在指定地形的范围内
    return (col_idx >= col_start) & (col_idx < col_end)  # 返回布尔掩码：列索引大于等于起始索引且小于结束索引（使用位与操作）
