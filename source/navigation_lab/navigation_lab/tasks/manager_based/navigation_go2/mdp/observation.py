from isaaclab.managers import ObservationTermCfg as ObsTerm
import torch
from dataclasses import dataclass

# -----------------------------
# 1️⃣ 扇区化处理函数
# -----------------------------
def lidar_to_sector_fn(lidar_tensor: torch.Tensor, num_sectors: int = 36, max_range: float = 10.0) -> torch.Tensor:
    """
    将 LiDAR 原始射线压缩成 sector 最小距离
    """
    num_rays = lidar_tensor.shape[0]
    rays_per_sector = num_rays // num_sectors

    sector_obs = []
    for i in range(num_sectors):
        start_idx = i * rays_per_sector
        end_idx = (i + 1) * rays_per_sector if i < num_sectors - 1 else num_rays
        sector_rays = lidar_tensor[start_idx:end_idx]
        sector_rays = torch.clamp(sector_rays, max=max_range)
        sector_obs.append(torch.min(sector_rays))
    return torch.stack(sector_obs)

# -----------------------------
# 2️⃣ Observation Term
# -----------------------------
@ObsTerm
def lidar_sector_observation(env, actor_idx: int, params):
    """
    ObsTerm 用法：获取 LiDAR 扇区最小距离
    """
    # 获取原始 LiDAR 射线
    sensor_name = params.get("sensor_name", f"lidar_{actor_idx}")
    lidar_tensor = env.get_sensor(sensor_name)  # IsaacLab 提供的获取 sensor 方法

    # 扇区化
    num_sectors = params.get("num_sectors", 36)
    max_range = params.get("max_range", 10.0)
    sector_obs = lidar_to_sector_fn(lidar_tensor, num_sectors=num_sectors, max_range=max_range)

    return sector_obs