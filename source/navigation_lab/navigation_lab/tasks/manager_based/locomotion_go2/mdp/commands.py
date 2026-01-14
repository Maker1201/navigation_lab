from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用前向引用

import torch  # 导入PyTorch库，用于张量操作和数学计算
from collections.abc import Sequence  # 导入Sequence抽象基类，用于类型注解（表示序列类型）
from typing import TYPE_CHECKING  # 导入TYPE_CHECKING，用于类型检查时的条件导入

from isaaclab.managers import CommandTerm, CommandTermCfg  # 导入命令项基类和命令项配置基类
from isaaclab.utils import configclass  # 导入配置类装饰器

from . import mdp  # 导入当前包下的mdp模块（相对导入）

from .utils import is_robot_on_terrain  # 从utils模块导入判断机器人是否在特定地形上的函数

if TYPE_CHECKING:  # 仅在类型检查时执行（运行时不会执行）
    from isaaclab.envs import ManagerBasedEnv  # 导入基于管理器的环境类（仅用于类型注解）


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):  # 带阈值的均匀速度命令类，继承自UniformVelocityCommand
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """
    """从均匀分布生成带阈值的SE(2)速度命令的命令生成器。

    此命令生成器自动检测"坑洞"地形并应用限制：
    - 对于坑洞地形：只允许向前移动（不允许横向或旋转移动）
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore  # 命令生成器的配置对象（忽略类型检查）
    """The configuration of the command generator."""
    """命令生成器的配置。"""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):  # 初始化方法
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        """初始化命令生成器。

        参数：
            cfg: 命令生成器的配置。
            env: 环境对象。
        """
        super().__init__(cfg, env)  # 调用父类初始化方法
        # Track which robots were on pit terrain in the previous step
        # 跟踪在上一步中哪些机器人在坑洞地形上
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 创建布尔张量，记录每个环境中的机器人是否在坑洞上

    def _resample_command(self, env_ids: Sequence[int]):  # 重新采样命令方法
        """Resample velocity commands with threshold."""
        """使用阈值重新采样速度命令。"""
        super()._resample_command(env_ids)  # 调用父类的重新采样方法
        # set small commands to zero
        # 将小的命令设置为零
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)  # 如果速度命令的前两个维度（x, y）的2范数小于等于0.2，则将其置零（阈值过滤）

    def _update_command(self):  # 更新命令方法
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        """实时更新命令并应用地形感知限制。

        此函数：
        1. 调用父类的更新方法以处理航向和站立环境
        2. 检查哪些机器人当前在坑洞地形上
        3. 对于离开坑洞的机器人：重新采样它们的命令
        4. 对于在坑洞上的机器人：限制为仅向前移动并将航向设置为0
        """
        # First, call parent's update command
        # 首先，调用父类的更新命令方法
        super()._update_command()  # 调用父类的更新命令方法

        # Check which robots are currently on pit terrain (real-time check every step)
        # 检查哪些机器人当前在坑洞地形上（每步实时检查）
        on_pits = is_robot_on_terrain(self._env, "pits")  # 检查每个环境中的机器人是否在"pits"地形上

        # Find robots that just left pit terrain (need to resample)
        # 找到刚刚离开坑洞地形的机器人（需要重新采样）
        left_pit_mask = self.was_on_pit & ~on_pits  # 计算掩码：之前在坑洞上但现在不在（使用位与和位非操作）
        if left_pit_mask.any():  # 如果有任何机器人离开了坑洞
            left_pit_env_ids = torch.where(left_pit_mask)[0]  # 获取离开坑洞的环境ID
            # Resample commands for robots that left pits
            # 为离开坑洞的机器人重新采样命令
            self._resample_command(left_pit_env_ids)  # 调用重新采样方法

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        # 对于当前在坑洞上的机器人：限制为仅向前移动，并设置最小/最大速度
        if on_pits.any():  # 如果有任何机器人在坑洞上
            pit_env_ids = torch.where(on_pits)[0]  # 获取在坑洞上的环境ID
            # Force forward-only movement with min and max speed limits
            # 强制仅向前移动，并设置最小和最大速度限制
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(  # 限制x方向速度（前进方向）
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6  # 取绝对值并限制在0.3到0.6之间
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement  # 将y方向速度（横向）设置为0（不允许横向移动）
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation  # 将z方向速度（偏航旋转）设置为0（不允许旋转）
            # Set heading to 0 for pit robots
            # 为坑洞上的机器人设置航向为0
            if self.cfg.heading_command:  # 如果配置中启用了航向命令
                self.heading_target[pit_env_ids] = 0.0  # 将航向目标设置为0

        # Update tracking state
        # 更新跟踪状态
        self.was_on_pit = on_pits  # 更新"之前在坑洞上"的状态为当前状态


@configclass  # 配置类装饰器
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):  # 带阈值的均匀速度命令配置类，继承自UniformVelocityCommandCfg
    """Configuration for the uniform threshold velocity command generator."""
    """带阈值的均匀速度命令生成器的配置。"""

    class_type: type = UniformThresholdVelocityCommand  # 类类型设置为UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):  # 离散命令控制器类，继承自CommandTerm
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """
    """
    为环境分配离散命令的命令生成器。

    命令存储为预定义的整数列表。
    控制器通过索引映射这些命令（例如，索引0 -> 10，索引1 -> 20）。
    """

    cfg: DiscreteCommandControllerCfg  # 命令控制器的配置对象
    """Configuration for the command controller."""
    """命令控制器的配置。"""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):  # 初始化方法
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        """
        初始化命令控制器。

        参数：
            cfg: 命令控制器的配置。
            env: 环境对象。
        """
        # Initialize the base class
        # 初始化基类
        super().__init__(cfg, env)  # 调用父类初始化方法

        # Validate that available_commands is non-empty
        # 验证available_commands不为空
        if not self.cfg.available_commands:  # 如果可用命令列表为空
            raise ValueError("The available_commands list cannot be empty.")  # 抛出值错误异常

        # Ensure all elements are integers
        # 确保所有元素都是整数
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):  # 如果存在非整数元素
            raise ValueError("All elements in available_commands must be integers.")  # 抛出值错误异常

        # Store the available commands
        # 存储可用命令
        self.available_commands = self.cfg.available_commands  # 保存配置中的可用命令列表

        # Create buffers to store the command
        # 创建缓冲区以存储命令
        # -- command buffer: stores discrete action indices for each environment
        # -- 命令缓冲区：存储每个环境的离散动作索引
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # 创建命令缓冲区张量（所有环境，整数类型）

        # -- current_commands: stores a snapshot of the current commands (as integers)
        # -- current_commands：存储当前命令的快照（作为整数）
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command  # 初始化当前命令列表，默认使用第一个可用命令

    def __str__(self) -> str:  # 字符串表示方法
        """Return a string representation of the command controller."""
        """返回命令控制器的字符串表示。"""
        return (  # 返回格式化的字符串
            "DiscreteCommandController:\n"  # 类名
            f"\tNumber of environments: {self.num_envs}\n"  # 环境数量
            f"\tAvailable commands: {self.available_commands}\n"  # 可用命令列表
        )

    """
    Properties
    """
    """
    属性
    """

    @property  # 属性装饰器
    def command(self) -> torch.Tensor:  # 命令属性
        """Return the current command buffer. Shape is (num_envs, 1)."""
        """返回当前命令缓冲区。形状为(num_envs, 1)。"""
        return self.command_buffer  # 返回命令缓冲区张量

    """
    Implementation specific functions.
    """
    """
    特定于实现的函数。
    """

    def _update_metrics(self):  # 更新指标方法
        """Update metrics for the command controller."""
        """更新命令控制器的指标。"""
        pass  # 空实现（不执行任何操作）

    def _resample_command(self, env_ids: Sequence[int]):  # 重新采样命令方法
        """Resample commands for the given environments."""
        """为给定的环境重新采样命令。"""
        sampled_indices = torch.randint(  # 随机生成索引
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device  # 从0到可用命令数量之间随机选择，生成与env_ids数量相同的索引
        )
        sampled_commands = torch.tensor(  # 创建命令张量
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device  # 根据采样的索引从可用命令列表中获取对应的命令值
        )
        self.command_buffer[env_ids] = sampled_commands  # 将采样的命令赋值给指定环境的命令缓冲区

    def _update_command(self):  # 更新命令方法
        """Update and store the current commands."""
        """更新并存储当前命令。"""
        self.current_commands = self.command_buffer.tolist()  # 将命令缓冲区转换为Python列表并存储到current_commands


@configclass  # 配置类装饰器
class DiscreteCommandControllerCfg(CommandTermCfg):  # 离散命令控制器配置类，继承自CommandTermCfg
    """Configuration for the discrete command controller."""
    """离散命令控制器的配置。"""

    class_type: type = DiscreteCommandController  # 类类型设置为DiscreteCommandController

    available_commands: list[int] = []  # 可用命令列表，默认为空列表
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
    """
    可用离散命令列表，其中每个元素都是整数。
    示例：[10, 20, 30, 40, 50]
    """
