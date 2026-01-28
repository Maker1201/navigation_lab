from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用前向引用

import torch  # 导入PyTorch库，用于张量操作和神经网络
from dataclasses import MISSING  # 导入MISSING常量，用于标记必需但未提供的配置字段
from typing import TYPE_CHECKING  # 导入TYPE_CHECKING，用于类型检查时的条件导入

import isaaclab.utils.math as math_utils  # 导入Isaac Lab数学工具模块，用于数学计算（如四元数操作）
from isaaclab.assets import Articulation  # 导入关节系统类，用于表示机器人
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager  # 导入动作项、动作配置、观测组配置和观测管理器
from isaaclab.markers import VisualizationMarkers  # 导入可视化标记类，用于在场景中显示可视化元素
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG  # 导入蓝色和绿色箭头标记配置
from isaaclab.utils import configclass  # 导入配置类装饰器
from isaaclab.utils.assets import check_file_path, read_file  # 导入文件路径检查和文件读取函数

if TYPE_CHECKING:  # 仅在类型检查时执行（运行时不会执行）
    from isaaclab.envs import ManagerBasedRLEnv  # 导入基于管理器的强化学习环境类（仅用于类型注解）


class PreTrainedPolicyAction(ActionTerm):  # 预训练策略动作项类，继承自ActionTerm
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """
    r"""预训练策略动作项。

    此动作项推理预训练策略并将相应的低层动作应用到机器人上。
    原始动作对应于预训练策略的命令。

    """

    cfg: PreTrainedPolicyActionCfg  # 动作项的配置对象
    """The configuration of the action term."""
    """动作项的配置。"""

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:  # 初始化方法
        # initialize the action term
        # 初始化动作项
        super().__init__(cfg, env)  # 调用父类初始化方法

        self.robot: Articulation = env.scene[cfg.asset_name]  # 从场景中获取机器人资产对象

        # load policy
        # 加载策略
        if not check_file_path(cfg.policy_path):  # 检查策略文件路径是否存在
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")  # 如果文件不存在，抛出文件未找到异常
        file_bytes = read_file(cfg.policy_path)  # 读取策略文件为字节流
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()  # 加载TorchScript模型，移动到指定设备，设置为评估模式

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)  # 初始化原始动作张量（所有环境，动作维度，在指定设备上）

        # prepare low level actions
        # 准备低层动作
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)  # 创建低层动作项实例
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)  # 初始化低层动作张量

        def last_action():  # 定义内部函数：获取上一次的动作
            # reset the low level actions if the episode was reset
            # 如果回合被重置，则重置低层动作
            if hasattr(env, "episode_length_buf"):  # 检查环境是否有回合长度缓冲区
                self.low_level_actions[env.episode_length_buf == 0, :] = 0  # 对于新回合（长度为0），将低层动作重置为0
            return self.low_level_actions  # 返回低层动作

        # remap some of the low level observations to internal observations
        # 将一些低层观测重新映射到内部观测
        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()  # 将动作观测函数设置为返回上一次动作
        cfg.low_level_observations.actions.params = dict()  # 清空动作观测的参数
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions  # 将速度命令观测函数设置为返回原始动作
        cfg.low_level_observations.velocity_commands.params = dict()  # 清空速度命令观测的参数

        # add the low level observations to the observation manager
        # 将低层观测添加到观测管理器
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)  # 创建低层观测管理器，观测组名为"ll_policy"

        self._counter = 0  # 初始化计数器，用于控制低层动作的更新频率

    """
    Properties.
    """
    """
    属性。
    """

    @property  # 属性装饰器
    def action_dim(self) -> int:  # 动作维度属性
        return 3  # 返回动作维度为3（x、y速度命令和可能的其他命令）

    @property  # 属性装饰器
    def raw_actions(self) -> torch.Tensor:  # 原始动作属性
        return self._raw_actions  # 返回原始动作张量

    @property  # 属性装饰器
    def processed_actions(self) -> torch.Tensor:  # 处理后的动作属性
        return self.raw_actions  # 返回原始动作（此实现中处理后的动作等于原始动作）

    """
    Operations.
    """
    """
    操作。
    """

    def process_actions(self, actions: torch.Tensor):  # 处理动作方法
        self._raw_actions[:] = actions  # 将输入动作复制到原始动作张量中（使用切片赋值以保持引用）

    def apply_actions(self):  # 应用动作方法
        if self._counter % self.cfg.low_level_decimation == 0:  # 如果计数器达到低层降采样倍数（每N步执行一次）
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")  # 计算低层策略观测组
            self.low_level_actions[:] = self.policy(low_level_obs)  # 使用预训练策略推理低层动作
            self._low_level_action_term.process_actions(self.low_level_actions)  # 处理低层动作
            self._counter = 0  # 重置计数器
        self._low_level_action_term.apply_actions()  # 应用低层动作到机器人
        self._counter += 1  # 增加计数器

    """
    Debug visualization.
    """
    """
    调试可视化。
    """

    def _set_debug_vis_impl(self, debug_vis: bool):  # 设置调试可视化的实现方法
        # set visibility of markers
        # 设置标记的可见性
        # note: parent only deals with callbacks. not their visibility
        # 注意：父类只处理回调，不处理可见性
        if debug_vis:  # 如果启用调试可视化
            # create markers if necessary for the first time
            # 如果需要，首次创建标记
            if not hasattr(self, "base_vel_goal_visualizer"):  # 如果还没有目标速度可视化器
                # -- goal
                # -- 目标
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()  # 复制绿色箭头标记配置
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"  # 设置标记的原始路径
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)  # 设置箭头标记的缩放比例
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)  # 创建目标速度可视化标记
                # -- current
                # -- 当前
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()  # 复制蓝色箭头标记配置
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"  # 设置标记的原始路径
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)  # 设置箭头标记的缩放比例
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)  # 创建当前速度可视化标记
            # set their visibility to true
            # 将它们的可见性设置为true
            self.base_vel_goal_visualizer.set_visibility(True)  # 显示目标速度可视化器
            self.base_vel_visualizer.set_visibility(True)  # 显示当前速度可视化器
        else:  # 如果禁用调试可视化
            if hasattr(self, "base_vel_goal_visualizer"):  # 如果存在可视化器
                self.base_vel_goal_visualizer.set_visibility(False)  # 隐藏目标速度可视化器
                self.base_vel_visualizer.set_visibility(False)  # 隐藏当前速度可视化器

    def _debug_vis_callback(self, event):  # 调试可视化回调方法
        # check if robot is initialized
        # 检查机器人是否已初始化
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        # 注意：这是必需的，以防机器人被取消初始化。我们无法访问数据
        if not self.robot.is_initialized:  # 如果机器人未初始化
            return  # 直接返回，不执行后续操作
        # get marker location
        # 获取标记位置
        # -- base state
        # -- 基座状态
        base_pos_w = self.robot.data.root_pos_w.clone()  # 克隆基座在世界坐标系中的位置
        base_pos_w[:, 2] += 0.5  # 在z轴上增加0.5米（将标记显示在基座上方）
        # -- resolve the scales and quaternions
        # -- 解析缩放和四元数
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])  # 将期望速度（原始动作的前两个维度）转换为箭头方向和缩放
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])  # 将当前基座线速度（前两个维度）转换为箭头方向和缩放
        # display markers
        # 显示标记
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)  # 可视化目标速度箭头
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)  # 可视化当前速度箭头

    """
    Internal helpers.
    """
    """
    内部辅助函数。
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # 将XY速度转换为箭头的辅助方法
        """Converts the XY base velocity command to arrow direction rotation."""
        """将XY基座速度命令转换为箭头方向旋转。"""
        # obtain default scale of the marker
        # 获取标记的默认缩放
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale  # 从配置中获取箭头标记的默认缩放
        # arrow-scale
        # 箭头缩放
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)  # 创建箭头缩放张量，为每个环境重复默认缩放
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0  # 根据速度大小调整箭头长度（x方向缩放乘以速度的2范数再乘以3.0）
        # arrow-direction
        # 箭头方向
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])  # 计算航向角（使用atan2计算y/x的反正切）
        zeros = torch.zeros_like(heading_angle)  # 创建与航向角相同形状的零张量
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)  # 从欧拉角（0, 0, 航向角）创建四元数
        # convert everything back from base to world frame
        # 将所有内容从基座坐标系转换回世界坐标系
        base_quat_w = self.robot.data.root_quat_w  # 获取基座在世界坐标系中的四元数
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)  # 将箭头四元数与基座四元数相乘，得到世界坐标系中的箭头方向

        return arrow_scale, arrow_quat  # 返回箭头缩放和四元数


@configclass  # 配置类装饰器
class PreTrainedPolicyActionCfg(ActionTermCfg):  # 预训练策略动作项配置类，继承自ActionTermCfg
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """
    """预训练策略动作项的配置。

    更多详情请参见 :class:`PreTrainedPolicyAction`。
    """

    class_type: type[ActionTerm] = PreTrainedPolicyAction  # 动作项的类类型
    """ Class of the action term."""
    """动作项的类。"""
    asset_name: str = MISSING  # 资产名称（必需字段）
    """Name of the asset in the environment for which the commands are generated."""
    """为其生成命令的环境中的资产名称。"""
    policy_path: str = MISSING  # 策略文件路径（必需字段）
    """Path to the low level policy (.pt files)."""
    """低层策略的路径（.pt文件）。"""
    low_level_decimation: int = 4  # 低层降采样倍数，默认值为4
    """Decimation factor for the low level action term."""
    """低层动作项的降采样因子。"""
    low_level_actions: ActionTermCfg = MISSING  # 低层动作配置（必需字段）
    """Low level action configuration."""
    """低层动作配置。"""
    low_level_observations: ObservationGroupCfg = MISSING  # 低层观测配置（必需字段）
    """Low level observation configuration."""
    """低层观测配置。"""
    debug_vis: bool = True  # 是否可视化调试信息，默认值为True
    """Whether to visualize debug information. Defaults to False."""
    """是否可视化调试信息。默认为False。"""
