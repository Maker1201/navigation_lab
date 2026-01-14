from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用前向引用

import torch  # 导入PyTorch库，用于张量操作和数学计算
from typing import TYPE_CHECKING  # 导入TYPE_CHECKING，用于类型检查时的条件导入

import isaaclab.utils.math as math_utils  # 导入Isaac Lab数学工具模块，用于数学计算（如四元数操作）
from isaaclab.assets import Articulation, RigidObject  # 导入关节系统和刚体对象类
from isaaclab.envs import mdp  # 导入MDP模块，包含其他奖励函数
from isaaclab.managers import ManagerTermBase  # 导入管理器项基类
from isaaclab.managers import RewardTermCfg as RewTerm  # 导入奖励项配置类，别名为RewTerm
from isaaclab.managers import SceneEntityCfg  # 导入场景实体配置类
from isaaclab.sensors import ContactSensor, RayCaster  # 导入接触传感器和射线投射器
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # 导入四元数逆应用和航向四元数函数

if TYPE_CHECKING:  # 仅在类型检查时执行（运行时不会执行）
    from isaaclab.envs import ManagerBasedRLEnv  # 导入基于管理器的强化学习环境类（仅用于类型注解）


def track_lin_vel_xy_exp(  # 使用指数核函数跟踪线性速度命令（xy轴）的奖励函数
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")  # 环境对象、标准差、命令名称、资产配置
) -> torch.Tensor:  # 返回奖励张量
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    """使用指数核函数跟踪线性速度命令（xy轴）的奖励。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    # compute the error
    # 计算误差
    lin_vel_error = torch.sum(  # 计算线性速度误差的平方和
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),  # 命令的前两个维度（x, y）减去基座在基座坐标系中的线性速度的前两个维度，然后平方
        dim=1,  # 在特征维度上求和
    )
    reward = torch.exp(-lin_vel_error / std**2)  # 使用指数核函数计算奖励：exp(-误差/标准差²)，误差越小奖励越大
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立），限制在0到0.7之间并归一化
    return reward  # 返回奖励张量


def track_ang_vel_z_exp(  # 使用指数核函数跟踪角速度命令（偏航）的奖励函数
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")  # 环境对象、标准差、命令名称、资产配置
) -> torch.Tensor:  # 返回奖励张量
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    """使用指数核函数跟踪角速度命令（偏航）的奖励。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    # compute the error
    # 计算误差
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])  # 命令的第3个维度（z，偏航角速度）减去基座在基座坐标系中的角速度的z分量，然后平方
    reward = torch.exp(-ang_vel_error / std**2)  # 使用指数核函数计算奖励：exp(-误差/标准差²)，误差越小奖励越大
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立），限制在0到0.7之间并归一化
    return reward  # 返回奖励张量


def track_lin_vel_xy_yaw_frame_exp(  # 在重力对齐的机器人坐标系中使用指数核函数跟踪线性速度命令（xy轴）的奖励函数
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")  # 环境对象、标准差、命令名称、资产配置
) -> torch.Tensor:  # 返回奖励张量
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    """在重力对齐的机器人坐标系中使用指数核函数跟踪线性速度命令（xy轴）的奖励。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset = env.scene[asset_cfg.name]  # 从场景中获取资产对象
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])  # 将世界坐标系中的线性速度转换到航向对齐的坐标系（使用航向四元数的逆变换）
    lin_vel_error = torch.sum(  # 计算线性速度误差的平方和
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1  # 命令的前两个维度减去转换后的速度的前两个维度，然后平方
    )
    reward = torch.exp(-lin_vel_error / std**2)  # 使用指数核函数计算奖励：exp(-误差/标准差²)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def track_ang_vel_z_world_exp(  # 在世界坐标系中使用指数核函数跟踪角速度命令（偏航）的奖励函数
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")  # 环境对象、命令名称、标准差、资产配置
) -> torch.Tensor:  # 返回奖励张量
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    """在世界坐标系中使用指数核函数跟踪角速度命令（偏航）的奖励。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset = env.scene[asset_cfg.name]  # 从场景中获取资产对象
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])  # 计算角速度误差：命令的第3个维度（z，偏航）减去世界坐标系中的角速度z分量，然后平方
    reward = torch.exp(-ang_vel_error / std**2)  # 使用指数核函数计算奖励：exp(-误差/标准差²)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:  # 关节功率奖励函数
    """Reward joint_power"""
    """奖励关节功率（惩罚高功率消耗）。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象（关节系统）
    # compute the reward
    # 计算奖励
    reward = torch.sum(  # 计算关节功率的总和
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),  # 关节速度乘以应用扭矩的绝对值（功率 = 速度 × 扭矩）
        dim=1,  # 在关节维度上求和
    )
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示功率消耗越大）


def stand_still(  # 站立静止奖励函数
    env: ManagerBasedRLEnv,  # 环境对象
    command_name: str,  # 命令名称
    command_threshold: float = 0.06,  # 命令阈值，默认0.06
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 资产配置，默认为"robot"
) -> torch.Tensor:  # 返回奖励张量
    """Penalize offsets from the default joint positions when the command is very small."""
    """当命令非常小时，惩罚关节位置相对于默认位置的偏移。"""
    # Penalize motion when command is nearly zero.
    # 当命令接近零时惩罚运动
    reward = mdp.joint_deviation_l1(env, asset_cfg)  # 计算关节位置与默认位置的L1偏差（使用MDP模块的函数）
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold  # 仅当命令的范数小于阈值时应用惩罚（命令很小时才惩罚）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def joint_pos_penalty(  # 关节位置惩罚函数
    env: ManagerBasedRLEnv,  # 环境对象
    command_name: str,  # 命令名称
    asset_cfg: SceneEntityCfg,  # 资产配置
    stand_still_scale: float,  # 站立静止时的缩放因子
    velocity_threshold: float,  # 速度阈值
    command_threshold: float,  # 命令阈值
) -> torch.Tensor:  # 返回奖励张量
    """Penalize joint position error from default on the articulation."""
    """惩罚关节位置相对于默认位置的误差。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象（关节系统）
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)  # 计算命令的2范数（命令的大小）
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)  # 计算基座在基座坐标系中xy方向线性速度的2范数
    running_reward = torch.linalg.norm(  # 计算运行时的奖励（关节位置误差的2范数）
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1  # 当前关节位置减去默认关节位置，然后计算2范数
    )
    reward = torch.where(  # 根据条件选择奖励值
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),  # 如果命令大于阈值或身体速度大于阈值（机器人正在运动）
        running_reward,  # 使用运行时的奖励
        stand_still_scale * running_reward,  # 否则使用站立静止时的缩放奖励（更大的惩罚）
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def wheel_vel_penalty(  # 轮子速度惩罚函数
    env: ManagerBasedRLEnv,  # 环境对象
    sensor_cfg: SceneEntityCfg,  # 传感器配置
    command_name: str,  # 命令名称
    velocity_threshold: float,  # 速度阈值
    command_threshold: float,  # 命令阈值
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 资产配置，默认为"robot"
) -> torch.Tensor:  # 返回奖励张量
    asset: Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象（关节系统）
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)  # 计算命令的2范数（命令的大小）
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)  # 计算基座在基座坐标系中xy方向线性速度的2范数
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])  # 获取关节速度的绝对值（仅针对配置中指定的关节）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 从场景中获取接触传感器
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]  # 计算身体是否首次在空中（布尔张量，仅针对配置中指定的身体）
    running_reward = torch.sum(in_air * joint_vel, dim=1)  # 计算运行时的奖励：在空中时关节速度的总和（惩罚轮子在空中的速度）
    standing_reward = torch.sum(joint_vel, dim=1)  # 计算站立时的奖励：所有关节速度的总和（惩罚所有关节速度）
    reward = torch.where(  # 根据条件选择奖励值
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),  # 如果命令大于阈值或身体速度大于阈值（机器人正在运动）
        running_reward,  # 使用运行时的奖励（只惩罚空中的轮子速度）
        standing_reward,  # 否则使用站立时的奖励（惩罚所有轮子速度）
    )
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示轮子速度越大）


class GaitReward(ManagerTermBase):  # 步态强化奖励项类，继承自ManagerTermBase
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """
    """四足动物的步态强化奖励项。

    此奖励惩罚在 :attr:`synced_feet_pair_names` 中定义的选定足对之间的接触时间差异，
    以引导策略朝向期望的步态，即小跑、跳跃或踱步。注意此奖励仅适用于具有两对同步足的四足步态。
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):  # 初始化方法
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        """初始化奖励项。

        参数：
            cfg: 奖励的配置。
            env: 强化学习环境实例。
        """
        super().__init__(cfg, env)  # 调用父类初始化方法
        self.std: float = cfg.params["std"]  # 标准差参数（用于指数核函数）
        self.command_name: str = cfg.params["command_name"]  # 命令名称
        self.max_err: float = cfg.params["max_err"]  # 最大误差（用于限制误差值）
        self.velocity_threshold: float = cfg.params["velocity_threshold"]  # 速度阈值
        self.command_threshold: float = cfg.params["command_threshold"]  # 命令阈值
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]  # 从场景中获取接触传感器
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]  # 从场景中获取资产对象（关节系统）
        # match foot body names with corresponding foot body ids
        # 匹配足部身体名称与对应的足部身体ID
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]  # 从配置中获取同步足对名称列表
        if (  # 验证同步足对配置的有效性
            len(synced_feet_pair_names) != 2  # 如果同步足对数量不等于2
            or len(synced_feet_pair_names[0]) != 2  # 或者第一个足对的数量不等于2
            or len(synced_feet_pair_names[1]) != 2  # 或者第二个足对的数量不等于2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")  # 抛出值错误异常
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]  # 查找第一个同步足对的身体ID（返回列表的第一个元素）
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]  # 查找第二个同步足对的身体ID（返回列表的第一个元素）
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]  # 存储同步足对的身体ID列表

    def __call__(  # 调用方法（使类实例可调用）
        self,  # 类实例自身
        env: ManagerBasedRLEnv,  # 环境对象
        std: float,  # 标准差参数
        command_name: str,  # 命令名称
        max_err: float,  # 最大误差
        velocity_threshold: float,  # 速度阈值
        command_threshold: float,  # 命令阈值
        synced_feet_pair_names,  # 同步足对名称（未使用，已在初始化时处理）
        asset_cfg: SceneEntityCfg,  # 资产配置（未使用，已在初始化时处理）
        sensor_cfg: SceneEntityCfg,  # 传感器配置（未使用，已在初始化时处理）
    ) -> torch.Tensor:  # 返回奖励张量
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        """计算奖励。

        此奖励定义为六个项的乘积，其中两项强制成对足部同步，
        其他四项奖励所有其他剩余对是否不同步。

        参数：
            env: 强化学习环境实例。
        返回：
            奖励值。
        """
        # for synchronous feet, the contact (air) times of two feet should match
        # 对于同步足部，两个足部的接触（空中）时间应该匹配
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])  # 计算第一个同步足对的同步奖励（足对0中的两个足部）
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])  # 计算第二个同步足对的同步奖励（足对1中的两个足部）
        sync_reward = sync_reward_0 * sync_reward_1  # 计算总同步奖励（两个同步足对奖励的乘积）
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        # 对于异步足部，一个足部的接触时间应该匹配另一个足部的空中时间
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])  # 计算异步奖励0：足对0的第0个足部与足对1的第0个足部
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])  # 计算异步奖励1：足对0的第1个足部与足对1的第1个足部
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])  # 计算异步奖励2：足对0的第0个足部与足对1的第1个足部
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])  # 计算异步奖励3：足对1的第0个足部与足对0的第1个足部
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3  # 计算总异步奖励（四个异步奖励的乘积）
        # only enforce gait if cmd > 0
        # 仅在命令大于0时强制执行步态
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)  # 计算命令的2范数（命令的大小）
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)  # 计算基座质心在基座坐标系中xy方向线性速度的2范数
        reward = torch.where(  # 根据条件选择奖励值
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),  # 如果命令大于阈值或身体速度大于阈值（机器人正在运动）
            sync_reward * async_reward,  # 使用同步奖励和异步奖励的乘积（强制执行步态）
            0.0,  # 否则奖励为0（不强制执行步态）
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
        return reward  # 返回奖励张量

    """
    Helper functions.
    """
    """
    辅助函数。
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:  # 同步奖励函数
        """Reward synchronization of two feet."""
        """奖励两个足部的同步。"""
        air_time = self.contact_sensor.data.current_air_time  # 获取当前空中时间（所有足部）
        contact_time = self.contact_sensor.data.current_contact_time  # 获取当前接触时间（所有足部）
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        # 惩罚同步足对之间最近空中时间和接触时间的差异。
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)  # 计算两个足部空中时间差的平方，并限制在最大误差²以内
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)  # 计算两个足部接触时间差的平方，并限制在最大误差²以内
        return torch.exp(-(se_air + se_contact) / self.std)  # 使用指数核函数计算奖励：exp(-(空中时间误差² + 接触时间误差²) / 标准差)，误差越小奖励越大

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:  # 异步奖励函数
        """Reward anti-synchronization of two feet."""
        """奖励两个足部的反同步（一个在空中时另一个接触地面）。"""
        air_time = self.contact_sensor.data.current_air_time  # 获取当前空中时间（所有足部）
        contact_time = self.contact_sensor.data.current_contact_time  # 获取当前接触时间（所有足部）
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        # 惩罚相反接触模式之间的差异：足部1的空中时间与足部2的接触时间，
        # 以及足部1的接触时间与足部2的空中时间（对于不同步的足对）。
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)  # 计算足部0的空中时间与足部1的接触时间差的平方，并限制在最大误差²以内
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)  # 计算足部0的接触时间与足部1的空中时间差的平方，并限制在最大误差²以内
        return torch.exp(-(se_act_0 + se_act_1) / self.std)  # 使用指数核函数计算奖励：exp(-(误差0² + 误差1²) / 标准差)，误差越小奖励越大


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:  # 关节镜像奖励函数
    """惩罚镜像关节对之间的位置差异，鼓励对称运动。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象（关节系统）
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:  # 如果环境没有关节镜像缓存或缓存为None
        # Cache joint positions for all pairs
        # 为所有关节对缓存关节位置
        env.joint_mirror_joints_cache = [  # 创建关节镜像缓存
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints  # 为每个镜像关节对查找关节索引（列表推导式）
        ]
    reward = torch.zeros(env.num_envs, device=env.device)  # 初始化奖励张量为零（所有环境）
    # Iterate over all joint pairs
    # 遍历所有关节对
    for joint_pair in env.joint_mirror_joints_cache:  # 遍历每个关节对
        # Calculate the difference for each pair and add to the total reward
        # 计算每对的差异并添加到总奖励中
        diff = torch.sum(  # 计算差异的总和
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),  # 计算两个镜像关节位置差的平方（joint_pair[0][0]和joint_pair[1][0]是关节索引列表的第一个元素）
            dim=-1,  # 在最后一个维度上求和（如果关节位置是多维的）
        )
        reward += diff  # 将差异添加到总奖励中（累积惩罚）
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0  # 将奖励除以镜像关节对的数量（平均化），如果对数为0则乘以0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示镜像关节位置差异越大）


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:  # 动作镜像奖励函数
    """惩罚镜像关节对之间的动作差异，鼓励对称动作。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象（关节系统）
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:  # 如果环境没有动作镜像缓存或缓存为None
        # Cache joint positions for all pairs
        # 为所有关节对缓存关节位置
        env.action_mirror_joints_cache = [  # 创建动作镜像缓存
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints  # 为每个镜像关节对查找关节索引（列表推导式）
        ]
    reward = torch.zeros(env.num_envs, device=env.device)  # 初始化奖励张量为零（所有环境）
    # Iterate over all joint pairs
    # 遍历所有关节对
    for joint_pair in env.action_mirror_joints_cache:  # 遍历每个关节对
        # Calculate the difference for each pair and add to the total reward
        # 计算每对的差异并添加到总奖励中
        diff = torch.sum(  # 计算差异的总和
            torch.square(  # 计算平方
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])  # 第一个镜像关节的动作绝对值
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])  # 减去第二个镜像关节的动作绝对值
            ),
            dim=-1,  # 在最后一个维度上求和（如果动作是多维的）
        )
        reward += diff  # 将差异添加到总奖励中（累积惩罚）
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0  # 将奖励除以镜像关节对的数量（平均化），如果对数为0则乘以0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示镜像关节动作差异越大）


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:  # 动作同步奖励函数
    """惩罚关节组内动作的方差，鼓励组内关节动作同步。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象（关节系统）

    # Cache joint indices if not already done
    # 如果尚未完成，则缓存关节索引
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:  # 如果环境没有动作同步缓存或缓存为None
        env.action_sync_joint_cache = [  # 创建动作同步缓存
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups  # 为每个关节组查找关节索引（列表推导式）
        ]

    reward = torch.zeros(env.num_envs, device=env.device)  # 初始化奖励张量为零（所有环境）
    # Iterate over each joint group
    # 遍历每个关节组
    for joint_group in env.action_sync_joint_cache:  # 遍历每个关节组
        if len(joint_group) < 2:  # 如果关节组中的关节数量少于2个
            continue  # need at least 2 joints to compare  # 需要至少2个关节才能比较，跳过此组

        # Get absolute actions for all joints in this group
        # 获取此组中所有关节的绝对动作
        actions = torch.stack(  # 堆叠动作张量
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1  # 为组中每个关节获取动作绝对值，沿维度1堆叠
        )  # shape: (num_envs, num_joints_in_group)  # 形状：(环境数，组内关节数)

        # Calculate mean action for each environment
        # 计算每个环境的平均动作
        mean_actions = torch.mean(actions, dim=1, keepdim=True)  # 在关节维度上计算平均值，保持维度以便后续广播

        # Calculate variance from mean for each joint
        # 计算每个关节相对于均值的方差
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)  # 计算动作与均值差的平方的平均值（方差）

        # Add to reward (we want to minimize this variance)
        # 添加到奖励中（我们希望最小化此方差）
        reward += variance.squeeze()  # 将方差添加到总奖励中（去除单维度并累积惩罚）
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0  # 将奖励除以关节组的数量（平均化），如果组数为0则乘以0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示组内关节动作差异越大）


def feet_air_time(  # 足部空中时间奖励函数
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float  # 环境对象、命令名称、传感器配置、阈值
) -> torch.Tensor:  # 返回奖励张量
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    """使用L2核函数奖励足部迈出的长步。

    此函数奖励智能体迈出超过阈值的步长。这有助于确保机器人将足部抬离地面并迈步。
    奖励计算为足部在空中的时间总和。

    如果命令很小（即智能体不应该迈步），则奖励为零。
    """
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 从场景中获取接触传感器
    # compute the reward
    # 计算奖励
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # 计算身体是否首次接触地面（布尔张量，仅针对配置中指定的身体）
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # 获取最后一次空中时间（仅针对配置中指定的身体）
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)  # 计算奖励：空中时间超过阈值的部分乘以首次接触标志，然后在身体维度上求和（只奖励首次接触时的长步）
    # no reward for zero command
    # 零命令时无奖励
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1  # 仅当命令的范数大于0.1时应用奖励（命令很小时不奖励）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:  # 双足动物足部空中时间正奖励函数
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    """奖励双足动物足部迈出的长步。

    此函数奖励智能体迈出达到指定阈值的步长，并保持一次只有一只脚在空中。

    如果命令很小（即智能体不应该迈步），则奖励为零。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 从场景中获取接触传感器
    # compute the reward
    # 计算奖励
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # 获取当前空中时间（仅针对配置中指定的身体）
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # 获取当前接触时间（仅针对配置中指定的身体）
    in_contact = contact_time > 0.0  # 判断身体是否在接触地面（布尔张量）
    in_mode_time = torch.where(in_contact, contact_time, air_time)  # 根据接触状态选择时间：接触时使用接触时间，否则使用空中时间
    single_stance = torch.sum(in_contact.int(), dim=1) == 1  # 判断是否为单足支撑（只有一个身体在接触地面）
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]  # 计算奖励：仅在单足支撑时取模式时间的最小值（确保一次只有一只脚在空中），否则为0
    reward = torch.clamp(reward, max=threshold)  # 将奖励限制在阈值以内（不超过阈值）
    # no reward for zero command
    # 零命令时无奖励
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1  # 仅当命令的范数大于0.1时应用奖励（命令很小时不奖励）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:  # 足部空中时间方差惩罚函数
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    """惩罚每个足部在空中/地面上的时间相对于彼此的方差。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 从场景中获取接触传感器
    # compute the reward
    # 计算奖励
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # 获取最后一次空中时间（仅针对配置中指定的身体）
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]  # 获取最后一次接触时间（仅针对配置中指定的身体）
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(  # 计算奖励：限制在0.5秒以内的空中时间方差加上
        torch.clip(last_contact_time, max=0.5), dim=1  # 限制在0.5秒以内的接触时间方差（在身体维度上计算方差）
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示足部时间差异越大）


def feet_contact(  # 足部接触奖励函数
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg  # 环境对象、命令名称、期望接触数量、传感器配置
) -> torch.Tensor:  # 返回奖励张量
    """Reward feet contact"""
    """奖励足部接触（惩罚不符合期望接触数量的情况）。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 从场景中获取接触传感器
    # compute the reward
    # 计算奖励
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # 计算身体是否首次接触地面（布尔张量，仅针对配置中指定的身体）
    contact_num = torch.sum(contact, dim=1)  # 计算接触的身体数量（在身体维度上求和）
    reward = (contact_num != expect_contact_num).float()  # 计算奖励：如果接触数量不等于期望数量则为1（惩罚），否则为0
    # no reward for zero command
    # 零命令时无奖励
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1  # 仅当命令的范数大于0.1时应用奖励（命令很小时不奖励）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示接触数量越不符合期望）


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:  # 无命令时足部接触奖励函数
    """Reward feet contact"""
    """奖励足部接触（仅在命令很小时）。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 从场景中获取接触传感器
    # compute the reward
    # 计算奖励
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # 计算身体是否首次接触地面（布尔张量，仅针对配置中指定的身体）
    reward = torch.sum(contact, dim=-1).float()  # 计算奖励：接触的身体数量（在最后一个维度上求和并转换为浮点数）
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1  # 仅当命令的范数小于0.1时应用奖励（命令很小时才奖励接触）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（值越大表示接触的足部越多）


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:  # 足部绊倒惩罚函数
    """惩罚足部撞击垂直表面的情况。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 从场景中获取接触传感器
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])  # 获取z方向接触力的绝对值（垂直方向，仅针对配置中指定的身体）
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)  # 计算xy方向接触力的2范数（水平方向，在xy维度上计算范数）
    # Penalize feet hitting vertical surfaces
    # 惩罚足部撞击垂直表面
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()  # 计算奖励：如果任何足部的水平力大于垂直力的4倍则为1（惩罚撞击垂直表面），否则为0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值为1表示有足部撞击垂直表面）


def feet_distance_y_exp(  # 足部y方向距离指数奖励函数
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")  # 环境对象、站姿宽度、标准差、资产配置
) -> torch.Tensor:  # 返回奖励张量
    """使用指数核函数奖励足部y方向距离符合期望站姿宽度。"""
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[  # 计算当前足部位置相对于根链接的偏移（世界坐标系）
        :, :  # 所有维度
    ].unsqueeze(1)  # 扩展维度以便广播
    n_feet = len(asset_cfg.body_ids)  # 获取足部数量
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)  # 初始化足部在基座坐标系中的位置张量（所有环境，足部数，3个维度）
    for i in range(n_feet):  # 遍历每个足部
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(  # 将足部位置转换到基座坐标系
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]  # 使用根链接四元数的共轭进行旋转变换
        )
    side_sign = torch.tensor(  # 创建侧边符号张量（用于区分左右足部）
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],  # 偶数索引为1.0（右侧），奇数索引为-1.0（左侧）
        device=env.device,  # 在指定设备上创建
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)  # 创建站姿宽度张量（所有环境，1个值）
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)  # 计算期望的y位置：站姿宽度的一半乘以侧边符号（扩展维度以便广播）
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])  # 计算站姿差异的平方：期望y位置减去实际y位置（索引1表示y维度）
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))  # 使用指数核函数计算奖励：exp(-差异总和/标准差²)，差异越小奖励越大
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def feet_distance_xy_exp(  # 足部xy方向距离指数奖励函数
    env: ManagerBasedRLEnv,  # 环境对象
    stance_width: float,  # 站姿宽度
    stance_length: float,  # 站姿长度
    std: float,  # 标准差
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 资产配置，默认为"robot"
) -> torch.Tensor:  # 返回奖励张量
    """使用指数核函数奖励足部xy方向距离符合期望站姿（宽度和长度）。"""
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）

    # Compute the current footstep positions relative to the root
    # 计算当前足部位置相对于根链接的偏移
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[  # 计算当前足部位置相对于根链接的偏移（世界坐标系）
        :, :  # 所有维度
    ].unsqueeze(1)  # 扩展维度以便广播

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)  # 初始化足部在基座坐标系中的位置张量（所有环境，4个足部，3个维度）
    for i in range(4):  # 遍历每个足部（假设是四足动物）
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(  # 将足部位置转换到基座坐标系
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]  # 使用根链接四元数的共轭进行旋转变换
        )

    # Desired x and y positions for each foot
    # 每个足部的期望x和y位置
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)  # 创建站姿宽度张量（所有环境，1个值）
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)  # 创建站姿长度张量（所有环境，1个值）

    desired_xs = torch.cat(  # 连接期望的x位置
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],  # 前两个足部在x方向为正（前方），后两个为负（后方）
        dim=1,  # 沿维度1连接
    )
    desired_ys = torch.cat(  # 连接期望的y位置
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1  # 交替的y位置：右、左、右、左（站姿宽度的一半）
    )

    # Compute differences in x and y
    # 计算x和y方向的差异
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])  # 计算x方向站姿差异的平方（索引0表示x维度）
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])  # 计算y方向站姿差异的平方（索引1表示y维度）

    # Combine x and y differences and compute the exponential penalty
    # 合并x和y方向的差异并计算指数惩罚
    stance_diff = stance_diff_x + stance_diff_y  # 将x和y方向的差异相加
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)  # 使用指数核函数计算奖励：exp(-差异总和/标准差²)，差异越小奖励越大
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def feet_height(  # 足部高度奖励函数
    env: ManagerBasedRLEnv,  # 环境对象
    command_name: str,  # 命令名称
    asset_cfg: SceneEntityCfg,  # 资产配置
    target_height: float,  # 目标高度
    tanh_mult: float,  # 双曲正切乘数
) -> torch.Tensor:  # 返回奖励张量
    """Reward the swinging feet for clearing a specified height off the ground"""
    """奖励摆动足部达到指定的离地高度。"""
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)  # 计算足部z位置与目标高度差的平方（索引2表示z维度，仅针对配置中指定的身体）
    foot_velocity_tanh = torch.tanh(  # 计算足部速度的双曲正切值
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)  # 足部在世界坐标系中xy方向线性速度的2范数乘以乘数（在xy维度上计算范数）
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)  # 计算奖励：高度误差乘以速度双曲正切值，然后在身体维度上求和（只奖励有速度的足部达到目标高度）
    # no reward for zero command
    # 零命令时无奖励
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1  # 仅当命令的范数大于0.1时应用奖励（命令很小时不奖励）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def feet_height_body(  # 基座坐标系中足部高度奖励函数
    env: ManagerBasedRLEnv,  # 环境对象
    command_name: str,  # 命令名称
    asset_cfg: SceneEntityCfg,  # 资产配置
    target_height: float,  # 目标高度
    tanh_mult: float,  # 双曲正切乘数
) -> torch.Tensor:  # 返回奖励张量
    """Reward the swinging feet for clearing a specified height off the ground"""
    """奖励摆动足部达到指定的离地高度（在基座坐标系中计算）。"""
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)  # 计算当前足部位置相对于根位置的偏移（世界坐标系）
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)  # 初始化足部在基座坐标系中的位置张量
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[  # 计算当前足部速度相对于根速度的偏移（世界坐标系）
        :, :  # 所有维度
    ].unsqueeze(1)  # 扩展维度以便广播
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)  # 初始化足部在基座坐标系中的速度张量
    for i in range(len(asset_cfg.body_ids)):  # 遍历每个身体
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(  # 将足部位置转换到基座坐标系
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]  # 使用根四元数的逆变换
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(  # 将足部速度转换到基座坐标系
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]  # 使用根四元数的逆变换
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)  # 计算足部z位置与目标高度差的平方（索引2表示z维度），并重塑为2D张量
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))  # 计算足部速度的双曲正切值：xy方向速度的2范数乘以乘数（在xy维度上计算范数）
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)  # 计算奖励：高度误差乘以速度双曲正切值，然后在身体维度上求和（只奖励有速度的足部达到目标高度）
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1  # 仅当命令的范数大于0.1时应用奖励（命令很小时不奖励）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量


def feet_slide(  # 足部滑动惩罚函数
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")  # 环境对象、传感器配置、资产配置
) -> torch.Tensor:  # 返回奖励张量
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    """惩罚足部滑动。

    此函数惩罚智能体在地面上滑动足部。奖励计算为足部线性速度的范数乘以二进制接触传感器。
    这确保智能体仅在足部与地面接触时受到惩罚。
    """
    # Penalize feet sliding
    # 惩罚足部滑动
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 从场景中获取接触传感器
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0  # 判断身体是否在接触：接触力历史的最大范数大于1.0（布尔张量）
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[  # 计算当前足部速度相对于根速度的偏移（世界坐标系）
        :, :  # 所有维度
    ].unsqueeze(1)  # 扩展维度以便广播
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)  # 初始化足部在基座坐标系中的速度张量
    for i in range(len(asset_cfg.body_ids)):  # 遍历每个身体
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(  # 将足部速度转换到基座坐标系
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]  # 使用根四元数的逆变换
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(  # 计算足部横向速度：xy方向速度的平方和的平方根（在xy维度上计算）
        env.num_envs, -1  # 重塑为2D张量
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)  # 计算奖励：横向速度乘以接触标志，然后在身体维度上求和（只惩罚接触时的滑动）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示足部滑动越大）


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:  # 向上奖励函数
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    """使用L2平方核函数惩罚非向上的基座方向（奖励向上）。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])  # 计算奖励：投影重力z分量与1的差的平方（z分量为-1时奖励为0，偏离时惩罚增加）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示基座越不向上）


def base_height_l2(  # 基座高度L2惩罚函数
    env: ManagerBasedRLEnv,  # 环境对象
    target_height: float,  # 目标高度
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 资产配置，默认为"robot"
    sensor_cfg: SceneEntityCfg | None = None,  # 传感器配置，可选
) -> torch.Tensor:  # 返回奖励张量
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    """使用L2平方核函数惩罚资产高度偏离目标。

    注意：
        对于平坦地形，目标高度在世界坐标系中。对于崎岖地形，
        传感器读数可以调整目标高度以考虑地形。
    """
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    if sensor_cfg is not None:  # 如果提供了传感器配置
        sensor: RayCaster = env.scene[sensor_cfg.name]  # 从场景中获取射线投射器传感器
        # Adjust the target height using the sensor data
        # 使用传感器数据调整目标高度
        ray_hits = sensor.data.ray_hits_w[..., 2]  # 获取射线命中点的z坐标（世界坐标系）
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:  # 如果射线命中数据无效（包含NaN、Inf或过大值）
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]  # 使用当前根链接位置作为调整后的目标高度（避免无效数据）
        else:  # 如果射线命中数据有效
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)  # 计算调整后的目标高度：目标高度加上射线命中点的平均z坐标（考虑地形高度）
    else:  # 如果没有提供传感器配置
        # Use the provided target height directly for flat terrain
        # 对于平坦地形，直接使用提供的目标高度
        adjusted_target_height = target_height  # 使用原始目标高度
    # Compute the L2 squared penalty
    # 计算L2平方惩罚
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)  # 计算奖励：根位置z坐标与调整后目标高度差的平方
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示高度偏离目标越大）


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:  # z方向线性速度L2惩罚函数
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    """使用L2平方核函数惩罚z方向基座线性速度。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])  # 计算奖励：基座在基座坐标系中z方向线性速度的平方（索引2表示z维度）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示z方向速度越大）


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:  # xy方向角速度L2惩罚函数
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    """使用L2平方核函数惩罚xy方向基座角速度。"""
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)  # 计算奖励：基座在基座坐标系中xy方向角速度的平方和（索引:2表示前两个维度，在特征维度上求和）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示xy方向角速度越大）


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


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:  # 平坦方向L2惩罚函数
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    """使用L2平方核函数惩罚非平坦的基座方向。

    这是通过惩罚投影重力向量的xy分量来计算的。
    """
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体）
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)  # 计算奖励：投影重力向量xy分量的平方和（索引:2表示前两个维度，在特征维度上求和）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 根据投影重力的z分量缩放奖励（确保机器人直立）
    return reward  # 返回奖励张量（实际上是惩罚，值越大表示基座方向越不平坦）
