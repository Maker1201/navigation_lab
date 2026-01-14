from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用前向引用

import torch  # 导入PyTorch库，用于张量操作和数学计算
from typing import TYPE_CHECKING  # 导入TYPE_CHECKING，用于类型检查时的条件导入

if TYPE_CHECKING:  # 仅在类型检查时执行（运行时不会执行）
    from isaaclab.envs import ManagerBasedRLEnv  # 导入基于管理器的强化学习环境类（仅用于类型注解）


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:  # 位置命令误差双曲正切奖励函数
    """Reward position tracking with tanh kernel."""
    """使用双曲正切核函数的位置跟踪奖励。"""
    command = env.command_manager.get_command(command_name)  # 从命令管理器获取指定名称的命令（包含位置和方向信息）
    des_pos_b = command[:, :3]  # 提取期望位置的前3个维度（x, y, z坐标），相对于基座坐标系
    distance = torch.norm(des_pos_b, dim=1)  # 计算期望位置的2范数（欧几里得距离），dim=1表示在特征维度上计算
    return 1 - torch.tanh(distance / std)  # 返回奖励值：1减去双曲正切函数（距离除以标准差），距离越小奖励越接近1，距离越大奖励越接近0


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:  # 航向命令误差绝对值惩罚函数
    """Penalize tracking orientation error."""
    """惩罚方向跟踪误差。"""
    command = env.command_manager.get_command(command_name)  # 从命令管理器获取指定名称的命令
    heading_b = command[:, 3]  # 提取命令的第4个维度（索引3），即航向角（相对于基座坐标系）
    return heading_b.abs()  # 返回航向角的绝对值作为惩罚值（误差越大，惩罚越大）
