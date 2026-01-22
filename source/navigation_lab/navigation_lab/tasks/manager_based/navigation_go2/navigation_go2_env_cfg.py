import math  # 导入数学库，用于数学计算（如圆周率等）

from isaaclab.envs import ManagerBasedRLEnvCfg  # 导入基于管理器的强化学习环境配置基类
from isaaclab.managers import EventTermCfg as EventTerm  # 导入事件项配置类，别名为EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # 导入观测组配置类，别名为ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm  # 导入观测项配置类，别名为ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm  # 导入奖励项配置类，别名为RewTerm
from isaaclab.managers import SceneEntityCfg  # 导入场景实体配置类
from isaaclab.managers import TerminationTermCfg as DoneTerm  # 导入终止项配置类，别名为DoneTerm
from isaaclab.utils import configclass  # 导入配置类装饰器，用于创建配置类
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # 导入Isaac Lab资源目录路径常量

import navigation_lab.tasks.manager_based.navigation_go2.mdp as mdp  # 导入导航任务的MDP（马尔可夫决策过程）模块
from navigation_lab.tasks.manager_based.locomotion_go2.locomotion_go2Exteroception_env_cfg import UnitreeGo2ExteroceptionRoughEnvCfg  # 导入Unitree Go2机器人的底层运动环境配置

LOW_LEVEL_ENV_CFG = UnitreeGo2ExteroceptionRoughEnvCfg()  # 创建底层环境配置实例，用于获取底层环境的配置参数


@configclass  # 配置类装饰器，将此类标记为配置类
class EventCfg:
    """Configuration for events."""
    """事件配置类：定义环境重置时的事件处理"""

    reset_base = EventTerm(  # 定义重置基座状态的事件项
        func=mdp.reset_root_state_uniform,  # 使用均匀分布重置根状态的函数
        mode="reset",  # 事件模式设置为"reset"（重置模式）
        params={  # 事件参数配置
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},  # 姿态范围：x、y位置范围±0.5米，偏航角范围±π弧度
            "velocity_range": {  # 速度范围配置
                "x": (-0.0, 0.0),  # x方向线速度范围（米/秒）
                "y": (-0.0, 0.0),  # y方向线速度范围（米/秒）
                "z": (-0.0, 0.0),  # z方向线速度范围（米/秒）
                "roll": (-0.0, 0.0),  # 横滚角速度范围（弧度/秒）
                "pitch": (-0.0, 0.0),  # 俯仰角速度范围（弧度/秒）
                "yaw": (-0.0, 0.0),  # 偏航角速度范围（弧度/秒）
            },
        },
    )


@configclass  # 配置类装饰器
class ActionsCfg:
    """Action terms for the MDP."""
    """动作配置类：定义MDP中的动作项"""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(  # 预训练策略动作配置
        asset_name="robot",  # 资产名称：机器人
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",  # 预训练策略模型文件路径
        low_level_decimation=4,  # 底层策略的降采样倍数（每4个高层步执行一次底层策略）
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,  # 底层动作配置：使用底层环境的关节位置控制
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,  # 底层观测配置：使用底层环境的策略观测
    )


@configclass  # 配置类装饰器
class ObservationsCfg:
    """Observation specifications for the MDP."""
    """观测配置类：定义MDP中的观测规范"""

    @configclass  # 嵌套配置类装饰器
    class PolicyCfg(ObsGroup):  # 策略观测组配置类，继承自ObsGroup
        """Observations for policy group."""
        """策略组的观测配置"""

        # observation terms (order preserved)
        # 观测项（保持顺序）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 基座线速度观测项：获取机器人基座的线性速度
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 投影重力观测项：获取投影到基座坐标系的重力向量
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})  # 姿态命令观测项：获取生成的姿态命令（pose_command）

    # observation groups
    # 观测组
    policy: PolicyCfg = PolicyCfg()  # 策略观测组实例


@configclass  # 配置类装饰器
class RewardsCfg:
    """Reward terms for the MDP."""
    """奖励配置类：定义MDP中的奖励项"""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)  # 终止惩罚奖励项：当环境终止时给予-400.0的惩罚
    position_tracking = RewTerm(  # 位置跟踪奖励项
        func=mdp.position_command_error_tanh,  # 使用双曲正切函数计算位置命令误差
        weight=0.5,  # 奖励权重为0.5
        params={"std": 2.0, "command_name": "pose_command"},  # 参数：标准差2.0米，命令名称为pose_command
    )
    position_tracking_fine_grained = RewTerm(  # 精细位置跟踪奖励项
        func=mdp.position_command_error_tanh,  # 使用双曲正切函数计算位置命令误差
        weight=0.5,  # 奖励权重为0.5
        params={"std": 0.2, "command_name": "pose_command"},  # 参数：标准差0.2米（更精细），命令名称为pose_command
    )
    orientation_tracking = RewTerm(  # 方向跟踪奖励项
        func=mdp.heading_command_error_abs,  # 使用绝对航向命令误差函数
        weight=-0.2,  # 奖励权重为-0.2（负值表示惩罚）
        params={"command_name": "pose_command"},  # 参数：命令名称为pose_command
    )


@configclass  # 配置类装饰器
class CommandsCfg:
    """Command terms for the MDP."""
    """命令配置类：定义MDP中的命令项"""

    pose_command = mdp.UniformPose2dCommandCfg(  # 均匀分布的2D姿态命令配置
        asset_name="robot",  # 资产名称：机器人
        simple_heading=False,  # 不使用简单航向模式（使用完整的方向角）
        resampling_time_range=(8.0, 8.0),  # 重新采样时间范围：8.0秒（固定间隔）
        debug_vis=True,  # 启用调试可视化
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),  # 命令范围：x、y位置±3.0米，航向角±π弧度
    )


@configclass  # 配置类装饰器
class TerminationsCfg:
    """Termination terms for the MDP."""
    """终止配置类：定义MDP中的终止条件"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 超时终止项：当达到最大时间步时终止，time_out标志为True
    base_contact = DoneTerm(  # 基座接触终止项：当基座发生非法接触时终止
        func=mdp.illegal_contact,  # 使用非法接触检测函数
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},  # 参数：接触力传感器配置（检测base身体），阈值1.0
    )


@configclass  # 配置类装饰器
class NavigationEnvCfg(ManagerBasedRLEnvCfg):  # 导航环境配置类，继承自ManagerBasedRLEnvCfg
    """Configuration for the navigation environment."""
    """导航环境配置：定义完整的导航环境配置"""

    # environment settings
    # 环境设置
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene  # 场景配置：使用底层环境的场景配置
    actions: ActionsCfg = ActionsCfg()  # 动作配置实例
    observations: ObservationsCfg = ObservationsCfg()  # 观测配置实例
    events: EventCfg = EventCfg()  # 事件配置实例
    # mdp settings
    # MDP设置
    commands: CommandsCfg = CommandsCfg()  # 命令配置实例
    rewards: RewardsCfg = RewardsCfg()  # 奖励配置实例
    terminations: TerminationsCfg = TerminationsCfg()  # 终止配置实例

    def __post_init__(self):  # 后初始化方法：在对象创建后自动调用
        """Post initialization."""
        """后初始化：设置环境参数"""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt  # 仿真时间步长：使用底层环境的仿真时间步长
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation  # 渲染间隔：使用底层环境的降采样倍数
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10  # 环境降采样倍数：底层环境的10倍（高层策略执行频率更低）
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]  # 回合长度（秒）：等于命令重新采样的时间范围上限（8.0秒）

        if self.scene.height_scanner is not None:  # 如果场景中存在高度扫描器
            self.scene.height_scanner.update_period = (  # 设置高度扫描器的更新周期
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt  # 等于底层降采样倍数乘以仿真时间步长
            )
        if self.scene.contact_forces is not None:  # 如果场景中存在接触力传感器
            self.scene.contact_forces.update_period = self.sim.dt  # 设置接触力传感器的更新周期为仿真时间步长


class NavigationEnvCfg_PLAY(NavigationEnvCfg):  # 用于游戏/演示的导航环境配置类，继承自NavigationEnvCfg
    def __post_init__(self) -> None:  # 后初始化方法，返回类型为None
        # post init of parent
        # 调用父类的后初始化方法
        super().__post_init__()

        # make a smaller scene for play
        # 为游戏/演示创建更小的场景
        self.scene.num_envs = 50  # 环境数量：设置为50个（比训练时更少，节省计算资源）
        self.scene.env_spacing = 2.5  # 环境间距：2.5米（环境之间的间隔距离）
        # disable randomization for play
        # 禁用随机化以便于游戏/演示
        self.observations.policy.enable_corruption = False  # 禁用观测的噪声/损坏（确保观测是干净的，便于观察）