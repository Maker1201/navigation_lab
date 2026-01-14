from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用前向引用

import torch  # 导入PyTorch库，用于张量操作和数学计算
from typing import TYPE_CHECKING  # 导入TYPE_CHECKING，用于类型检查时的条件导入

from isaaclab.assets import Articulation  # 导入关节系统类，用于表示机器人
from isaaclab.managers import SceneEntityCfg  # 导入场景实体配置类

if TYPE_CHECKING:  # 仅在类型检查时执行（运行时不会执行）
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv  # 导入基于管理器的环境类（仅用于类型注解）


def joint_pos_rel_without_wheel(  # 计算关节位置相对值（排除轮子关节）的函数
    env: ManagerBasedEnv,  # 环境对象
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 资产配置，默认为"robot"
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 轮子资产配置，默认为"robot"（用于指定轮子关节）
) -> torch.Tensor:  # 返回关节位置相对值的张量
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    """资产关节位置相对于默认关节位置的值（排除轮子关节）。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象（关节系统）
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # 计算关节位置相对值：当前关节位置减去默认关节位置（仅针对配置中指定的关节）
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0  # 将轮子关节的相对位置设置为0（排除轮子关节的影响）
    return joint_pos_rel  # 返回关节位置相对值张量


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:  # 计算相位信息的函数
    """计算周期性运动的相位信息，返回正弦和余弦值。"""
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:  # 如果环境没有episode_length_buf属性或该属性为None
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)  # 初始化回合长度缓冲区为零张量（所有环境，长整型）
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time  # 计算相位：回合长度（扩展为列向量）乘以步长时间除以周期时间，得到归一化的相位值（0到1之间）
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)  # 创建相位张量：将相位的正弦值和余弦值沿最后一个维度连接（用于表示周期性运动的相位信息）
    return phase_tensor  # 返回相位张量（形状为(num_envs, 2)，包含sin和cos值）
