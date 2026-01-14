from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用前向引用
"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""
"""可用于为学习环境创建课程学习的通用函数。

这些函数可以传递给 :class:`isaaclab.managers.CurriculumTermCfg` 对象以启用
函数引入的课程学习。
"""
import torch  # 导入PyTorch库，用于张量操作和数学计算
from collections.abc import Sequence  # 导入Sequence抽象基类，用于类型注解（表示序列类型）
from typing import TYPE_CHECKING  # 导入TYPE_CHECKING，用于类型检查时的条件导入

if TYPE_CHECKING:  # 仅在类型检查时执行（运行时不会执行）
    from isaaclab.envs import ManagerBasedRLEnv  # 导入基于管理器的强化学习环境类（仅用于类型注解）


def command_levels_lin_vel(  # 线性速度命令课程学习函数
    env: ManagerBasedRLEnv,  # 环境对象
    env_ids: Sequence[int],  # 环境ID序列
    reward_term_name: str,  # 奖励项名称（用于评估性能）
    range_multiplier: Sequence[float] = (0.1, 1.0),  # 范围乘数，默认从0.1到1.0（初始难度到最终难度）
) -> None:  # 返回类型为None（实际返回张量，但类型注解可能不准确）
    """command_levels_lin_vel"""
    """线性速度命令课程学习：根据性能逐步增加线性速度命令的难度范围。"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges  # 获取基座速度命令的范围配置
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    # 获取原始速度范围（仅在第一个回合）
    if env.common_step_counter == 0:  # 如果是第一个全局步（环境初始化时）
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)  # 保存原始x方向线性速度范围
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)  # 保存原始y方向线性速度范围
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]  # 计算初始x方向速度范围（原始范围乘以初始乘数）
        env._final_vel_x = env._original_vel_x * range_multiplier[1]  # 计算最终x方向速度范围（原始范围乘以最终乘数）
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]  # 计算初始y方向速度范围（原始范围乘以初始乘数）
        env._final_vel_y = env._original_vel_y * range_multiplier[1]  # 计算最终y方向速度范围（原始范围乘以最终乘数）

        # Initialize command ranges to initial values
        # 将命令范围初始化为初始值
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()  # 设置x方向速度范围为初始值（转换为列表）
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()  # 设置y方向速度范围为初始值（转换为列表）

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    # 避免在每一步都更新命令课程，因为最大命令对所有环境都是共同的
    if env.common_step_counter % env.max_episode_length == 0:  # 如果达到回合结束（每回合结束时更新一次）
        episode_sums = env.reward_manager._episode_sums[reward_term_name]  # 获取指定奖励项的回合累计和
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)  # 获取奖励项的配置（包含权重等信息）
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)  # 定义命令范围的增量（上下限各增加0.1）

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # 如果跟踪奖励超过最大值的80%，则增加命令范围
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:  # 计算平均奖励率，如果超过权重的80%
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command  # 计算新的x方向速度范围（当前范围加上增量）
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command  # 计算新的y方向速度范围（当前范围加上增量）

            # Clamp to ensure we don't exceed final ranges
            # 限制范围以确保不超过最终范围
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])  # 将x方向速度范围限制在最终范围内
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])  # 将y方向速度范围限制在最终范围内

            # Update ranges
            # 更新范围
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()  # 更新x方向速度范围（转换为列表）
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()  # 更新y方向速度范围（转换为列表）

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)  # 返回x方向速度范围的上限（作为课程难度指标）


def command_levels_ang_vel(  # 角速度命令课程学习函数
    env: ManagerBasedRLEnv,  # 环境对象
    env_ids: Sequence[int],  # 环境ID序列
    reward_term_name: str,  # 奖励项名称（用于评估性能）
    range_multiplier: Sequence[float] = (0.1, 1.0),  # 范围乘数，默认从0.1到1.0（初始难度到最终难度）
) -> None:  # 返回类型为None（实际返回张量，但类型注解可能不准确）
    """command_levels_ang_vel"""
    """角速度命令课程学习：根据性能逐步增加角速度命令的难度范围。"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges  # 获取基座速度命令的范围配置
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    # 获取原始角速度范围（仅在第一个回合）
    if env.common_step_counter == 0:  # 如果是第一个全局步（环境初始化时）
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)  # 保存原始z方向角速度范围
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]  # 计算初始z方向角速度范围（原始范围乘以初始乘数）
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]  # 计算最终z方向角速度范围（原始范围乘以最终乘数）

        # Initialize command ranges to initial values
        # 将命令范围初始化为初始值
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()  # 设置z方向角速度范围为初始值（转换为列表）

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    # 避免在每一步都更新命令课程，因为最大命令对所有环境都是共同的
    if env.common_step_counter % env.max_episode_length == 0:  # 如果达到回合结束（每回合结束时更新一次）
        episode_sums = env.reward_manager._episode_sums[reward_term_name]  # 获取指定奖励项的回合累计和
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)  # 获取奖励项的配置（包含权重等信息）
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)  # 定义命令范围的增量（上下限各增加0.1）

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # 如果跟踪奖励超过最大值的80%，则增加命令范围
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:  # 计算平均奖励率，如果超过权重的80%
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command  # 计算新的z方向角速度范围（当前范围加上增量）

            # Clamp to ensure we don't exceed final ranges
            # 限制范围以确保不超过最终范围
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])  # 将z方向角速度范围限制在最终范围内

            # Update ranges
            # 更新范围
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()  # 更新z方向角速度范围（转换为列表）

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)  # 返回z方向角速度范围的上限（作为课程难度指标）
