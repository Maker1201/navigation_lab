# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils  # 导入isaaclab.sim模块并重命名为sim_utils：用于仿真工具
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg,RayCasterCfg,patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR,ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


##
# Scene definition
##


@configclass  # 装饰器：将类标记为配置类，用于自动生成配置对象
class MySceneCfg(InteractiveSceneCfg):  # 场景配置类，继承自交互式场景配置基类
    """Configuration for a cart-pole scene."""  # 类的文档字符串：说明这是用于cart-pole场景的配置

    # ground plane  # 注释：地面平面配置
    terrain = TerrainImporterCfg(  # 地面资产的基础配置对象
        prim_path="/World/ground",  # 在USD场景图中的原始路径，指定地面对象的位置
        terrain_type="generator",  # 地形类型：使用生成器模式创建地形
        terrain_generator=ROUGH_TERRAINS_CFG,  # 地形生成器配置：使用预定义的粗糙地形配置
        max_init_terrain_level=5,  # 最大初始地形等级：地形生成时的最高难度级别
        collision_group=-1,  # 碰撞组：-1表示不与任何碰撞组交互（禁用碰撞）
        physics_material=sim_utils.RigidBodyMaterialCfg(  # 物理材质配置：定义地面的物理属性
            friction_combine_mode="multiply",  # 摩擦系数组合模式：使用乘法模式合并摩擦系数
            restitution_combine_mode="multiply",  # 恢复系数组合模式：使用乘法模式合并恢复系数
            static_friction=1.0,  # 静摩擦系数：物体静止时的摩擦系数值
            dynamic_friction=1.0,  # 动摩擦系数：物体运动时的摩擦系数值
            restitution=1.0,  # 恢复系数（弹性系数）：碰撞后的能量恢复比例，1.0表示完全弹性碰撞
            ),  # 物理材质配置结束

        visual_material=sim_utils.MdlFileCfg(  # 视觉材质配置：定义地面的外观材质
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",  # 材质定义文件路径：指向大理石瓷砖材质文件
            project_uvw=True,  # UVW投影：启用UVW坐标投影以正确映射纹理
            texture_scale=(0.25, 0.25),  # 纹理缩放：在U和V方向上将纹理缩放为原来的0.25倍（放大纹理）
            ),  # 视觉材质配置结束
        debug_vis=False,  # 调试可视化：关闭调试可视化功能

    )  # 地面配置结束

    # robot  # 注释：机器人配置
    robot: ArticulationCfg = MISSING  # 机器人关节配置：使用MISSING标记，表示此配置必须在子类中提供

    # sensors  # 注释：传感器配置
    height_scanner = RayCasterCfg(  # 高度扫描器配置：使用射线投射器扫描地形高度
        prim_path="{ENV_REGEX_NS}/Robot/base",  # 传感器挂载路径：使用正则表达式匹配所有环境中的机器人基座
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 偏移量配置：传感器相对于基座的位置偏移（x, y, z），z=20.0表示在基座上方20米
        ray_alignment="yaw",  # 射线对齐方式：射线方向与机器人的偏航角（yaw）对齐
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 扫描模式配置：网格模式，分辨率0.1米，扫描区域大小为1.6x1.0米
        debug_vis=False,  # 调试可视化：关闭调试可视化
        mesh_prim_paths=["/World/ground"],  # 网格原始路径列表：指定要检测的地面网格路径
    )  # 高度扫描器配置结束
    height_scanner_base = RayCasterCfg(  # 基座高度扫描器配置：用于精确检测基座下方地形的扫描器
        prim_path="{ENV_REGEX_NS}/Robot/base",  # 传感器挂载路径：同样挂载在机器人基座上
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 偏移量配置：与主高度扫描器相同的偏移位置
        ray_alignment="yaw",  # 射线对齐方式：与机器人偏航角对齐
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),  # 扫描模式配置：更高分辨率的网格（0.05米），更小的扫描区域（0.1x0.1米）
        debug_vis=False,  # 调试可视化：关闭调试可视化
        mesh_prim_paths=["/World/ground"],  # 网格原始路径列表：检测地面网格
    )  # 基座高度扫描器配置结束
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)  # 接触力传感器配置：检测机器人所有部件的接触力，历史长度为3帧，跟踪空中时间
    # lights  # 注释：灯光配置
    sky_light = AssetBaseCfg(  # 天空光资产配置：场景的环境光照配置
        prim_path="/World/skyLight",  # 灯光在场景图中的路径：指定天空光对象的位置
        spawn=sim_utils.DomeLightCfg(  # 生成配置：使用圆顶光（环境光）配置
            intensity=750.0,  # 光照强度：设置环境光的强度值
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",  # 纹理文件路径：使用HDR天空纹理文件作为环境光照源
        ),  # 圆顶光配置结束
    )  # 天空光配置结束


##
# MDP settings
##


@configclass  # 装饰器：将类标记为配置类
class CommandsCfg:  # 命令配置类：定义MDP中的命令规格
    """Command specifications for the MDP."""  # 类的文档字符串：说明这是MDP的命令规格配置

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(  # 基础速度命令配置：使用均匀阈值速度命令生成器
        asset_name="robot",  # 资产名称：指定应用命令的机器人资产
        resampling_time_range=(10.0, 10.0),  # 重采样时间范围：命令重新采样的时间间隔范围（秒），这里固定为10秒
        rel_standing_envs=0.02,  # 相对站立环境比例：在命令采样时保持站立状态的环境比例（2%）
        rel_heading_envs=1.0,  # 相对朝向环境比例：在命令采样时改变朝向的环境比例（100%）
        heading_command=True,  # 朝向命令：是否启用朝向命令（True表示启用）
        heading_control_stiffness=0.5,  # 朝向控制刚度：朝向控制的刚度系数，值越大响应越快
        debug_vis=True,  # 调试可视化：是否启用调试可视化（True表示启用）
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(  # 命令范围配置：定义速度命令的取值范围
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)  # 线性速度x范围(-1到1m/s)，线性速度y范围(-1到1m/s)，角速度z范围(-1到1rad/s)，朝向角度范围(-π到π)
        ),  # 命令范围配置结束
    )  # 基础速度命令配置结束


@configclass  # 装饰器：将类标记为配置类
class ActionsCfg:  # 动作配置类：定义MDP中的动作规格
    """Action specifications for the MDP."""  # 类的文档字符串：说明这是MDP的动作规格配置

    joint_pos = mdp.JointPositionActionCfg(  # 关节位置动作配置：使用关节位置控制作为动作空间
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, clip=None, preserve_order=True  # 资产名称：机器人资产；关节名称：匹配所有关节的正则表达式；缩放因子：0.5；使用默认偏移：True；裁剪范围：None（不裁剪）；保持顺序：True
    )  # 关节位置动作配置结束


@configclass  # 装饰器：将类标记为配置类
class ObservationsCfg:  # 观测配置类：定义MDP中的观测规格
    """Observation specifications for the MDP."""  # 类的文档字符串：说明这是MDP的观测规格配置

    @configclass  # 装饰器：将内部类标记为配置类
    class PolicyCfg(ObsGroup):  # 策略观测组配置类：继承自观测组基类，用于策略网络的观测
        """Observations for policy group."""  # 类的文档字符串：说明这是策略组的观测配置

        # observation terms (order preserved)  # 注释：观测项（保持顺序）
        base_lin_vel = ObsTerm(  # 基座线性速度观测项：观测机器人基座的线性速度
            func=mdp.base_lin_vel,  # 观测函数：使用基座线性速度计算函数
            noise=Unoise(n_min=-0.1, n_max=0.1),  # 噪声配置：添加高斯噪声，范围从-0.1到0.1
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数，1.0表示不缩放
        )  # 基座线性速度观测项结束
        base_ang_vel = ObsTerm(  # 基座角速度观测项：观测机器人基座的角速度
            func=mdp.base_ang_vel,  # 观测函数：使用基座角速度计算函数
            noise=Unoise(n_min=-0.2, n_max=0.2),  # 噪声配置：添加高斯噪声，范围从-0.2到0.2
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 基座角速度观测项结束
        projected_gravity = ObsTerm(  # 投影重力观测项：观测重力在机器人基座坐标系中的投影
            func=mdp.projected_gravity,  # 观测函数：使用投影重力计算函数
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 噪声配置：添加高斯噪声，范围从-0.05到0.05
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 投影重力观测项结束
        velocity_commands = ObsTerm(  # 速度命令观测项：观测生成的速度命令
            func=mdp.generated_commands,  # 观测函数：使用生成命令的计算函数
            params={"command_name": "base_velocity"},  # 参数：指定命令名称为"base_velocity"
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 速度命令观测项结束
        joint_pos = ObsTerm(  # 关节位置观测项：观测关节的相对位置
            func=mdp.joint_pos_rel,  # 观测函数：使用关节相对位置计算函数
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},  # 参数：场景实体配置，匹配所有关节并保持顺序
            noise=Unoise(n_min=-0.01, n_max=0.01),  # 噪声配置：添加高斯噪声，范围从-0.01到0.01
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 关节位置观测项结束
        joint_vel = ObsTerm(  # 关节速度观测项：观测关节的相对速度
            func=mdp.joint_vel_rel,  # 观测函数：使用关节相对速度计算函数
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},  # 参数：场景实体配置，匹配所有关节并保持顺序
            noise=Unoise(n_min=-1.5, n_max=1.5),  # 噪声配置：添加高斯噪声，范围从-1.5到1.5
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 关节速度观测项结束
        actions = ObsTerm(  # 动作观测项：观测上一时刻执行的动作
            func=mdp.last_action,  # 观测函数：使用上一动作的计算函数
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 动作观测项结束
        height_scan = ObsTerm(  # 高度扫描观测项：观测地形高度扫描数据
            func=mdp.height_scan,  # 观测函数：使用高度扫描计算函数
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},  # 参数：指定高度扫描器传感器配置
            noise=Unoise(n_min=-0.1, n_max=0.1),  # 噪声配置：添加高斯噪声，范围从-0.1到0.1
            clip=(-1.0, 1.0),  # 裁剪范围：将观测值裁剪到-1到1之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 高度扫描观测项结束

        def __post_init__(self):  # 后初始化方法：在对象创建后自动调用
            self.enable_corruption = True  # 启用损坏：允许对观测添加噪声和损坏（True表示启用）
            self.concatenate_terms = True  # 连接项：将所有观测项连接成一个向量（True表示启用）

    @configclass  # 装饰器：将内部类标记为配置类
    class CriticCfg(ObsGroup):  # 评价器观测组配置类：继承自观测组基类，用于价值网络的观测
        """Observations for critic group."""  # 类的文档字符串：说明这是评价器组的观测配置

        # observation terms (order preserved)  # 注释：观测项（保持顺序）
        base_lin_vel = ObsTerm(  # 基座线性速度观测项：观测机器人基座的线性速度
            func=mdp.base_lin_vel,  # 观测函数：使用基座线性速度计算函数
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 基座线性速度观测项结束
        base_ang_vel = ObsTerm(  # 基座角速度观测项：观测机器人基座的角速度
            func=mdp.base_ang_vel,  # 观测函数：使用基座角速度计算函数
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 基座角速度观测项结束
        projected_gravity = ObsTerm(  # 投影重力观测项：观测重力在机器人基座坐标系中的投影
            func=mdp.projected_gravity,  # 观测函数：使用投影重力计算函数
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 投影重力观测项结束
        velocity_commands = ObsTerm(  # 速度命令观测项：观测生成的速度命令
            func=mdp.generated_commands,  # 观测函数：使用生成命令的计算函数
            params={"command_name": "base_velocity"},  # 参数：指定命令名称为"base_velocity"
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 速度命令观测项结束
        joint_pos = ObsTerm(  # 关节位置观测项：观测关节的相对位置
            func=mdp.joint_pos_rel,  # 观测函数：使用关节相对位置计算函数
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},  # 参数：场景实体配置，匹配所有关节并保持顺序
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 关节位置观测项结束
        joint_vel = ObsTerm(  # 关节速度观测项：观测关节的相对速度
            func=mdp.joint_vel_rel,  # 观测函数：使用关节相对速度计算函数
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},  # 参数：场景实体配置，匹配所有关节并保持顺序
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 关节速度观测项结束
        actions = ObsTerm(  # 动作观测项：观测上一时刻执行的动作
            func=mdp.last_action,  # 观测函数：使用上一动作的计算函数
            clip=(-100.0, 100.0),  # 裁剪范围：将观测值裁剪到-100到100之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 动作观测项结束
        height_scan = ObsTerm(  # 高度扫描观测项：观测地形高度扫描数据
            func=mdp.height_scan,  # 观测函数：使用高度扫描计算函数
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},  # 参数：指定高度扫描器传感器配置
            clip=(-1.0, 1.0),  # 裁剪范围：将观测值裁剪到-1到1之间
            scale=1.0,  # 缩放因子：观测值的缩放系数
        )  # 高度扫描观测项结束
        # joint_effort = ObsTerm(  # 注释掉的关节力矩观测项：观测关节的力矩（当前未使用）
        #     func=mdp.joint_effort,  # 观测函数：使用关节力矩计算函数
        #     clip=(-100, 100),  # 裁剪范围：将观测值裁剪到-100到100之间
        #     scale=0.01,  # 缩放因子：观测值的缩放系数为0.01
        # )  # 关节力矩观测项结束

        def __post_init__(self):  # 后初始化方法：在对象创建后自动调用
            self.enable_corruption = False  # 启用损坏：不允许对观测添加噪声和损坏（False表示禁用，评价器需要精确观测）
            self.concatenate_terms = True  # 连接项：将所有观测项连接成一个向量（True表示启用）

    # observation groups  # 注释：观测组
    policy: PolicyCfg = PolicyCfg()  # 策略观测组：创建策略网络的观测组实例
    critic: CriticCfg = CriticCfg()  # 评价器观测组：创建价值网络的观测组实例


@configclass  # 装饰器：将类标记为配置类
class EventCfg:  # 事件配置类：定义MDP中的事件规格（用于域随机化）
    """Configuration for events."""  # 类的文档字符串：说明这是事件配置

    # startup  # 注释：启动时事件（在环境初始化时执行一次）
    randomize_rigid_body_material = EventTerm(  # 随机化刚体材质事件：随机化机器人刚体的物理材质属性
        func=mdp.randomize_rigid_body_material,  # 事件函数：使用随机化刚体材质的函数
        mode="startup",  # 事件模式：在启动时执行
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 资产配置：匹配机器人的所有刚体
            "static_friction_range": (0.3, 1.0),  # 静摩擦系数范围：从0.3到1.0
            "dynamic_friction_range": (0.3, 0.8),  # 动摩擦系数范围：从0.3到0.8
            "restitution_range": (0.0, 0.5),  # 恢复系数范围：从0.0到0.5（弹性系数）
            "num_buckets": 64,  # 桶数量：用于离散化材质参数的桶数，64表示将范围分成64个离散值
        },  # 参数配置结束
    )  # 随机化刚体材质事件结束

    randomize_rigid_body_mass_base = EventTerm(  # 随机化基座质量事件：随机化机器人基座的质量
        func=mdp.randomize_rigid_body_mass,  # 事件函数：使用随机化刚体质量的函数
        mode="startup",  # 事件模式：在启动时执行
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 资产配置：匹配机器人的基座（空字符串表示基座）
            "mass_distribution_params": (-1.0, 3.0),  # 质量分布参数：从-1.0到3.0
            "operation": "add",  # 操作类型：使用加法操作（在原始质量基础上增加）
            "recompute_inertia": True,  # 重新计算惯性：根据新质量重新计算惯性矩阵
        },  # 参数配置结束
    )  # 随机化基座质量事件结束

    randomize_rigid_body_mass_others = EventTerm(  # 随机化其他刚体质量事件：随机化机器人其他部件（非基座）的质量
        func=mdp.randomize_rigid_body_mass,  # 事件函数：使用随机化刚体质量的函数
        mode="startup",  # 事件模式：在启动时执行
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 资产配置：匹配机器人的所有刚体
            "mass_distribution_params": (0.7, 1.3),  # 质量分布参数：从0.7到1.3（相对于原始质量的倍数）
            "operation": "scale",  # 操作类型：使用缩放操作（将原始质量乘以一个系数）
            "recompute_inertia": True,  # 重新计算惯性：根据新质量重新计算惯性矩阵
        },  # 参数配置结束
    )  # 随机化其他刚体质量事件结束

    # Skip: inertia updated via mass randomization by setting recompute_inertia=True  # 注释：跳过惯性随机化，因为通过设置recompute_inertia=True已在质量随机化时更新
    # randomize_rigid_body_inertia = EventTerm(  # 注释掉的随机化刚体惯性事件（当前未使用）
    #     func=mdp.randomize_rigid_body_inertia,  # 事件函数：使用随机化刚体惯性的函数
    #     mode="startup",  # 事件模式：在启动时执行
    #     params={  # 参数配置
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 资产配置：匹配机器人的所有刚体
    #         "inertia_distribution_params": (0.5, 1.5),  # 惯性分布参数：从0.5到1.5
    #         "operation": "scale",  # 操作类型：使用缩放操作
    #     },  # 参数配置结束
    # )  # 随机化刚体惯性事件结束

    randomize_com_positions = EventTerm(  # 随机化质心位置事件：随机化机器人刚体的质心位置
        func=mdp.randomize_rigid_body_com,  # 事件函数：使用随机化刚体质心的函数
        mode="startup",  # 事件模式：在启动时执行
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 资产配置：匹配机器人的所有刚体
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},  # 质心范围：x、y、z方向的偏移范围均为-0.05到0.05米
        },  # 参数配置结束
    )  # 随机化质心位置事件结束

    # reset  # 注释：重置时事件（在每次环境重置时执行）
    randomize_apply_external_force_torque = EventTerm(  # 随机化施加外力力矩事件：在重置时对机器人施加随机的外力和力矩
        func=mdp.apply_external_force_torque,  # 事件函数：使用施加外力力矩的函数
        mode="reset",  # 事件模式：在重置时执行
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 资产配置：匹配机器人的基座
            "force_range": (-10.0, 10.0),  # 力范围：外力的大小范围从-10.0到10.0牛顿
            "torque_range": (-10.0, 10.0),  # 力矩范围：外力矩的大小范围从-10.0到10.0牛米
        },  # 参数配置结束
    )  # 随机化施加外力力矩事件结束

    randomize_reset_joints = EventTerm(  # 随机化重置关节事件：在重置时随机化关节的位置和速度
        func=mdp.reset_joints_by_scale,  # 事件函数：使用按比例重置关节的函数
        # func=mdp.reset_joints_by_offset,  # 注释掉的替代函数：使用偏移量重置关节的函数（当前未使用）
        mode="reset",  # 事件模式：在重置时执行
        params={  # 参数配置
            "position_range": (1.0, 1.0),  # 位置范围：关节位置的缩放范围（1.0表示保持默认位置）
            "velocity_range": (0.0, 0.0),  # 速度范围：关节速度的范围（0.0表示重置为零速度）
        },  # 参数配置结束
    )  # 随机化重置关节事件结束

    randomize_actuator_gains = EventTerm(  # 随机化执行器增益事件：在重置时随机化执行器的刚度和阻尼增益
        func=mdp.randomize_actuator_gains,  # 事件函数：使用随机化执行器增益的函数
        mode="reset",  # 事件模式：在重置时执行
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 资产配置：匹配机器人的所有关节
            "stiffness_distribution_params": (0.5, 2.0),  # 刚度分布参数：从0.5到2.0（相对于原始刚度的倍数）
            "damping_distribution_params": (0.5, 2.0),  # 阻尼分布参数：从0.5到2.0（相对于原始阻尼的倍数）
            "operation": "scale",  # 操作类型：使用缩放操作
            "distribution": "uniform",  # 分布类型：使用均匀分布
        },  # 参数配置结束
    )  # 随机化执行器增益事件结束

    randomize_reset_base = EventTerm(  # 随机化重置基座事件：在重置时随机化机器人基座的位置、朝向和速度
        func=mdp.reset_root_state_uniform,  # 事件函数：使用均匀分布重置根状态的函数
        mode="reset",  # 事件模式：在重置时执行
        params={  # 参数配置
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},  # 姿态范围：x位置-0.5到0.5米，y位置-0.5到0.5米，偏航角-π到π弧度
            "velocity_range": {  # 速度范围配置
                "x": (-0.5, 0.5),  # x方向线性速度：从-0.5到0.5米/秒
                "y": (-0.5, 0.5),  # y方向线性速度：从-0.5到0.5米/秒
                "z": (-0.5, 0.5),  # z方向线性速度：从-0.5到0.5米/秒
                "roll": (-0.5, 0.5),  # 滚转角速度：从-0.5到0.5弧度/秒
                "pitch": (-0.5, 0.5),  # 俯仰角速度：从-0.5到0.5弧度/秒
                "yaw": (-0.5, 0.5),  # 偏航角速度：从-0.5到0.5弧度/秒
            },  # 速度范围配置结束
        },  # 参数配置结束
    )  # 随机化重置基座事件结束

    # interval  # 注释：间隔事件（在运行过程中按时间间隔执行）
    randomize_push_robot = EventTerm(  # 随机化推动机器人事件：在运行过程中按间隔对机器人施加推动
        func=mdp.push_by_setting_velocity,  # 事件函数：使用设置速度推动机器人的函数
        mode="interval",  # 事件模式：按间隔执行
        interval_range_s=(10.0, 15.0),  # 间隔范围：事件触发的时间间隔从10.0到15.0秒
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},  # 参数：推动速度范围，x和y方向均为-0.5到0.5米/秒
    )  # 随机化推动机器人事件结束


@configclass  # 装饰器：将类标记为配置类
class RewardsCfg:  # 奖励配置类：定义MDP中的奖励项规格
    """Reward terms for the MDP."""  # 类的文档字符串：说明这是MDP的奖励项配置

    # General  # 注释：通用奖励项
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)  # 终止奖励项：检查是否终止，权重为0.0（当前未使用）

    # Root penalties  # 注释：基座惩罚项（惩罚基座的不理想状态）
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)  # z方向线性速度L2惩罚：惩罚基座在z方向的垂直速度，权重为0.0
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)  # xy方向角速度L2惩罚：惩罚基座在xy平面的角速度（滚转和俯仰），权重为0.0
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)  # 平坦朝向L2惩罚：惩罚基座偏离水平朝向，权重为0.0
    base_height_l2 = RewTerm(  # 基座高度L2惩罚：惩罚基座偏离目标高度
        func=mdp.base_height_l2,  # 奖励函数：使用基座高度L2计算函数
        weight=0.0,  # 权重：0.0（当前未使用）
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 资产配置：匹配机器人的基座
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),  # 传感器配置：使用基座高度扫描器
            "target_height": 0.0,  # 目标高度：期望的基座高度为0.0米（相对于地面）
        },  # 参数配置结束
    )  # 基座高度L2惩罚结束
    body_lin_acc_l2 = RewTerm(  # 基座线性加速度L2惩罚：惩罚基座的线性加速度
        func=mdp.body_lin_acc_l2,  # 奖励函数：使用基座线性加速度L2计算函数
        weight=0.0,  # 权重：0.0（当前未使用）
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},  # 参数：匹配机器人的基座
    )  # 基座线性加速度L2惩罚结束

    # Joint penalties  # 注释：关节惩罚项（惩罚关节的不理想状态）
    joint_torques_l2 = RewTerm(  # 关节力矩L2惩罚：惩罚关节的力矩大小
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}  # 奖励函数：关节力矩L2；权重：0.0；参数：匹配所有关节
    )  # 关节力矩L2惩罚结束
    joint_vel_l2 = RewTerm(  # 关节速度L2惩罚：惩罚关节的速度大小
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}  # 奖励函数：关节速度L2；权重：0.0；参数：匹配所有关节
    )  # 关节速度L2惩罚结束
    joint_acc_l2 = RewTerm(  # 关节加速度L2惩罚：惩罚关节的加速度大小
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}  # 奖励函数：关节加速度L2；权重：0.0；参数：匹配所有关节
    )  # 关节加速度L2惩罚结束

    def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):  # 创建关节偏差L1奖励项方法：动态创建关节偏差奖励项
        rew_term = RewTerm(  # 创建奖励项对象
            func=mdp.joint_deviation_l1,  # 奖励函数：使用关节偏差L1计算函数
            weight=weight,  # 权重：使用传入的权重参数
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},  # 参数：使用传入的关节名称模式
        )  # 奖励项对象创建结束
        setattr(self, attr_name, rew_term)  # 设置属性：将创建的奖励项设置为类的属性

    joint_pos_limits = RewTerm(  # 关节位置限制惩罚：惩罚关节超出位置限制
        func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}  # 奖励函数：关节位置限制；权重：0.0；参数：匹配所有关节
    )  # 关节位置限制惩罚结束
    joint_vel_limits = RewTerm(  # 关节速度限制惩罚：惩罚关节超出速度限制
        func=mdp.joint_vel_limits,  # 奖励函数：关节速度限制
        weight=0.0,  # 权重：0.0
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},  # 参数：匹配所有关节，软比例1.0（完全软限制）
    )  # 关节速度限制惩罚结束
    joint_power = RewTerm(  # 关节功率惩罚：惩罚关节消耗的功率
        func=mdp.joint_power,  # 奖励函数：关节功率计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 资产配置：匹配所有关节
        },  # 参数配置结束
    )  # 关节功率惩罚结束

    stand_still = RewTerm(  # 站立静止奖励：奖励机器人在命令为零时保持静止
        func=mdp.stand_still,  # 奖励函数：站立静止计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "command_name": "base_velocity",  # 命令名称：基础速度命令
            "command_threshold": 0.1,  # 命令阈值：命令小于0.1时视为静止
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 资产配置：匹配所有关节
        },  # 参数配置结束
    )  # 站立静止奖励结束

    joint_pos_penalty = RewTerm(  # 关节位置惩罚：根据运动状态惩罚关节位置偏差
        func=mdp.joint_pos_penalty,  # 奖励函数：关节位置惩罚计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "command_name": "base_velocity",  # 命令名称：基础速度命令
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 资产配置：匹配所有关节
            "stand_still_scale": 5.0,  # 站立静止缩放：静止时的惩罚缩放系数
            "velocity_threshold": 0.5,  # 速度阈值：判断是否运动的基座速度阈值
            "command_threshold": 0.1,  # 命令阈值：判断命令是否为零的阈值
        },  # 参数配置结束
    )  # 关节位置惩罚结束

    wheel_vel_penalty = RewTerm(  # 轮子速度惩罚：惩罚轮子在接触地面时的速度
        func=mdp.wheel_vel_penalty,  # 奖励函数：轮子速度惩罚计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),  # 资产配置：匹配基座（空字符串表示基座）
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),  # 传感器配置：匹配基座的接触力传感器
            "command_name": "base_velocity",  # 命令名称：基础速度命令
            "velocity_threshold": 0.5,  # 速度阈值：判断是否运动的基座速度阈值
            "command_threshold": 0.1,  # 命令阈值：判断命令是否为零的阈值
        },  # 参数配置结束
    )  # 轮子速度惩罚结束

    joint_mirror = RewTerm(  # 关节镜像奖励：奖励左右对称的关节配置
        func=mdp.joint_mirror,  # 奖励函数：关节镜像计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot"),  # 资产配置：机器人资产
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],  # 镜像关节对：前右-后左，前左-后右
        },  # 参数配置结束
    )  # 关节镜像奖励结束

    action_mirror = RewTerm(  # 动作镜像奖励：奖励左右对称的动作
        func=mdp.action_mirror,  # 奖励函数：动作镜像计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot"),  # 资产配置：机器人资产
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],  # 镜像关节对：前右-后左，前左-后右
        },  # 参数配置结束
    )  # 动作镜像奖励结束

    action_sync = RewTerm(  # 动作同步奖励：奖励同类型关节的同步动作
        func=mdp.action_sync,  # 奖励函数：动作同步计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot"),  # 资产配置：机器人资产
            "joint_groups": [  # 关节组：定义需要同步的关节组
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],  # 髋关节组：所有髋关节
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],  # 大腿关节组：所有大腿关节
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],  # 小腿关节组：所有小腿关节
            ],  # 关节组配置结束
        },  # 参数配置结束
    )  # 动作同步奖励结束

    # Action penalties  # 注释：动作惩罚项
    applied_torque_limits = RewTerm(  # 应用力矩限制惩罚：惩罚超出限制的关节力矩
        func=mdp.applied_torque_limits,  # 奖励函数：应用力矩限制计算
        weight=0.0,  # 权重：0.0
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},  # 参数：匹配所有关节
    )  # 应用力矩限制惩罚结束
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)  # 动作变化率L2惩罚：惩罚动作的快速变化，权重为0.0
    # smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0.0)  # 平滑度1奖励（注释）：与action_rate_l2相同
    # smoothness_2 = RewTerm(func=mdp.smoothness_2, weight=0.0)  # 平滑度2奖励（注释）：当前不可用

    # Contact sensor  # 注释：接触传感器相关奖励项
    undesired_contacts = RewTerm(  # 不期望接触惩罚：惩罚不应该接触地面的身体部位
        func=mdp.undesired_contacts,  # 奖励函数：不期望接触计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),  # 传感器配置：匹配基座的接触力传感器
            "threshold": 1.0,  # 阈值：接触力阈值，超过此值视为接触
        },  # 参数配置结束
    )  # 不期望接触惩罚结束
    contact_forces = RewTerm(  # 接触力惩罚：惩罚过大的接触力
        func=mdp.contact_forces,  # 奖励函数：接触力计算
        weight=0.0,  # 权重：0.0
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 100.0},  # 参数：匹配基座接触力传感器，阈值100.0牛顿
    )  # 接触力惩罚结束

    # Velocity-tracking rewards  # 注释：速度跟踪奖励项（奖励跟踪命令速度）
    track_lin_vel_xy_exp = RewTerm(  # 跟踪xy线性速度指数奖励：奖励跟踪xy平面的线性速度命令
        func=mdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}  # 奖励函数：跟踪xy线性速度指数；权重：0.0；参数：命令名称"base_velocity"，标准差sqrt(0.25)=0.5
    )  # 跟踪xy线性速度指数奖励结束
    track_ang_vel_z_exp = RewTerm(  # 跟踪z角速度指数奖励：奖励跟踪z方向的角速度命令
        func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}  # 奖励函数：跟踪z角速度指数；权重：0.0；参数：命令名称"base_velocity"，标准差sqrt(0.25)=0.5
    )  # 跟踪z角速度指数奖励结束

    # Others  # 注释：其他奖励项（主要是足部相关）
    feet_air_time = RewTerm(  # 足部空中时间奖励：奖励足部在空中停留的时间
        func=mdp.feet_air_time,  # 奖励函数：足部空中时间计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "command_name": "base_velocity",  # 命令名称：基础速度命令
            "threshold": 0.5,  # 阈值：判断是否运动的基座速度阈值
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),  # 传感器配置：匹配基座的接触力传感器
        },  # 参数配置结束
    )  # 足部空中时间奖励结束

    feet_air_time_variance = RewTerm(  # 足部空中时间方差惩罚：惩罚足部空中时间的方差（鼓励一致的步态）
        func=mdp.feet_air_time_variance_penalty,  # 奖励函数：足部空中时间方差惩罚计算
        weight=0,  # 权重：0
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},  # 参数：匹配基座的接触力传感器
    )  # 足部空中时间方差惩罚结束

    feet_gait = RewTerm(  # 足部步态奖励：奖励符合期望步态模式的足部接触
        func=mdp.GaitReward,  # 奖励函数：步态奖励计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "std": math.sqrt(0.5),  # 标准差：sqrt(0.5)用于指数奖励计算
            "command_name": "base_velocity",  # 命令名称：基础速度命令
            "max_err": 0.2,  # 最大误差：允许的最大步态误差
            "velocity_threshold": 0.5,  # 速度阈值：判断是否运动的基座速度阈值
            "command_threshold": 0.1,  # 命令阈值：判断命令是否为零的阈值
            "synced_feet_pair_names": (("", ""), ("", "")),  # 同步足部对名称：定义需要同步的足部对（当前为空）
            "asset_cfg": SceneEntityCfg("robot"),  # 资产配置：机器人资产
            "sensor_cfg": SceneEntityCfg("contact_forces"),  # 传感器配置：接触力传感器
        },  # 参数配置结束
    )  # 足部步态奖励结束

    feet_contact = RewTerm(  # 足部接触奖励：奖励符合期望接触数量的足部接触
        func=mdp.feet_contact,  # 奖励函数：足部接触计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),  # 传感器配置：匹配基座的接触力传感器
            "command_name": "base_velocity",  # 命令名称：基础速度命令
            "expect_contact_num": 2,  # 期望接触数量：期望同时接触地面的足部数量为2
        },  # 参数配置结束
    )  # 足部接触奖励结束

    feet_contact_without_cmd = RewTerm(  # 无命令时足部接触奖励：奖励在无命令时保持足部接触
        func=mdp.feet_contact_without_cmd,  # 奖励函数：无命令时足部接触计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),  # 传感器配置：匹配基座的接触力传感器
            "command_name": "base_velocity",  # 命令名称：基础速度命令
        },  # 参数配置结束
    )  # 无命令时足部接触奖励结束

    feet_stumble = RewTerm(  # 足部绊倒惩罚：惩罚足部意外接触地面（绊倒）
        func=mdp.feet_stumble,  # 奖励函数：足部绊倒计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),  # 传感器配置：匹配基座的接触力传感器
        },  # 参数配置结束
    )  # 足部绊倒惩罚结束

    feet_slide = RewTerm(  # 足部滑动惩罚：惩罚足部在地面上滑动
        func=mdp.feet_slide,  # 奖励函数：足部滑动计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),  # 传感器配置：匹配基座的接触力传感器
            "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 资产配置：匹配机器人的基座
        },  # 参数配置结束
    )  # 足部滑动惩罚结束

    feet_height = RewTerm(  # 足部高度奖励：奖励足部保持在目标高度
        func=mdp.feet_height,  # 奖励函数：足部高度计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 资产配置：匹配机器人的基座
            "tanh_mult": 2.0,  # tanh倍数：用于平滑奖励函数的倍数
            "target_height": 0.05,  # 目标高度：期望的足部高度为0.05米（相对于地面）
            "command_name": "base_velocity",  # 命令名称：基础速度命令
        },  # 参数配置结束
    )  # 足部高度奖励结束

    feet_height_body = RewTerm(  # 足部相对基座高度奖励：奖励足部相对于基座保持在目标高度
        func=mdp.feet_height_body,  # 奖励函数：足部相对基座高度计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 资产配置：匹配机器人的基座
            "tanh_mult": 2.0,  # tanh倍数：用于平滑奖励函数的倍数
            "target_height": -0.3,  # 目标高度：期望的足部相对基座高度为-0.3米（基座下方）
            "command_name": "base_velocity",  # 命令名称：基础速度命令
        },  # 参数配置结束
    )  # 足部相对基座高度奖励结束

    feet_distance_y_exp = RewTerm(  # 足部y方向距离指数奖励：奖励足部在y方向保持期望距离
        func=mdp.feet_distance_y_exp,  # 奖励函数：足部y方向距离指数计算
        weight=0.0,  # 权重：0.0
        params={  # 参数配置
            "std": math.sqrt(0.25),  # 标准差：sqrt(0.25)=0.5用于指数奖励计算
            "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 资产配置：匹配机器人的基座
            "stance_width": float,  # 站姿宽度：期望的站姿宽度（浮点数类型，需要具体值）
        },  # 参数配置结束
    )  # 足部y方向距离指数奖励结束

    # feet_distance_xy_exp = RewTerm(  # 注释掉的足部xy方向距离指数奖励（当前未使用）
    #     func=mdp.feet_distance_xy_exp,  # 奖励函数：足部xy方向距离指数计算
    #     weight=0.0,  # 权重：0.0
    #     params={  # 参数配置
    #         "std": math.sqrt(0.25),  # 标准差：sqrt(0.25)=0.5用于指数奖励计算
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 资产配置：匹配机器人的基座
    #         "stance_length": float,  # 站姿长度：期望的站姿长度（浮点数类型）
    #         "stance_width": float,  # 站姿宽度：期望的站姿宽度（浮点数类型）
    #     },  # 参数配置结束
    # )  # 足部xy方向距离指数奖励结束

    upward = RewTerm(func=mdp.upward, weight=0.0)  # 向上奖励：奖励机器人保持向上（不翻转），权重为0.0


@configclass  # 装饰器：将类标记为配置类
class TerminationsCfg:  # 终止条件配置类：定义MDP中的终止条件规格
    """Termination terms for the MDP."""  # 类的文档字符串：说明这是MDP的终止条件配置

    # MDP terminations  # 注释：MDP终止条件
    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 超时终止：当达到最大时间步数时终止，time_out=True表示这是超时终止
    # command_resample  # 注释：命令重采样（注释说明）
    terrain_out_of_bounds = DoneTerm(  # 地形越界终止：当机器人超出地形边界时终止
        func=mdp.terrain_out_of_bounds,  # 终止函数：地形越界检查函数
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},  # 参数：机器人资产配置，距离缓冲区3.0米
        time_out=True,  # 超时标志：True表示这也是一种超时终止（允许重新采样命令）
    )  # 地形越界终止结束

    # Contact sensor  # 注释：接触传感器相关终止条件
    illegal_contact = DoneTerm(  # 非法接触终止：当不应该接触地面的身体部位接触地面时终止
        func=mdp.illegal_contact,  # 终止函数：非法接触检查函数
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},  # 参数：匹配基座的接触力传感器，阈值1.0牛顿
    )  # 非法接触终止结束


@configclass  # 装饰器：将类标记为配置类
class CurriculumCfg:  # 课程学习配置类：定义MDP中的课程学习规格
    """Curriculum terms for the MDP."""  # 类的文档字符串：说明这是MDP的课程学习配置

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)  # 地形等级课程：根据速度跟踪性能调整地形难度等级

    command_levels_lin_vel = CurrTerm(  # 线性速度命令等级课程：根据线性速度跟踪性能调整命令范围
        func=mdp.command_levels_lin_vel,  # 课程函数：线性速度命令等级调整函数
        params={  # 参数配置
            "reward_term_name": "track_lin_vel_xy_exp",  # 奖励项名称：用于评估的奖励项名称
            "range_multiplier": (0.1, 1.0),  # 范围倍数：命令范围的倍数从0.1到1.0（逐步增加难度）
        },  # 参数配置结束
    )  # 线性速度命令等级课程结束

    command_levels_ang_vel = CurrTerm(  # 角速度命令等级课程：根据角速度跟踪性能调整命令范围
        func=mdp.command_levels_ang_vel,  # 课程函数：角速度命令等级调整函数
        params={  # 参数配置
            "reward_term_name": "track_ang_vel_z_exp",  # 奖励项名称：用于评估的奖励项名称
            "range_multiplier": (0.1, 1.0),  # 范围倍数：命令范围的倍数从0.1到1.0（逐步增加难度）
        },  # 参数配置结束
    )  # 角速度命令等级课程结束


##
# Environment configuration  # 注释：环境配置部分
##


@configclass  # 装饰器：将类标记为配置类
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):  # 运动速度跟踪粗糙地形环境配置类：继承自基于管理器的RL环境配置
    """Configuration for the locomotion velocity-tracking environment."""  # 类的文档字符串：说明这是运动速度跟踪环境的配置

    # Scene settings  # 注释：场景设置
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)  # 场景配置：创建场景配置实例，环境数量4096，环境间距2.5米
    # Basic settings  # 注释：基本设置
    observations: ObservationsCfg = ObservationsCfg()  # 观测配置：创建观测配置实例
    actions: ActionsCfg = ActionsCfg()  # 动作配置：创建动作配置实例
    commands: CommandsCfg = CommandsCfg()  # 命令配置：创建命令配置实例
    # MDP settings  # 注释：MDP设置
    rewards: RewardsCfg = RewardsCfg()  # 奖励配置：创建奖励配置实例
    terminations: TerminationsCfg = TerminationsCfg()  # 终止条件配置：创建终止条件配置实例
    events: EventCfg = EventCfg()  # 事件配置：创建事件配置实例
    curriculum: CurriculumCfg = CurriculumCfg()  # 课程学习配置：创建课程学习配置实例

    def __post_init__(self):  # 后初始化方法：在对象创建后自动调用
        """Post initialization."""  # 方法的文档字符串：说明这是后初始化方法
        # general settings  # 注释：通用设置
        self.decimation = 4  # 抽取率：每4个物理步骤执行一次控制（动作频率降低）
        self.episode_length_s = 20.0  # 回合长度：每个回合持续20.0秒
        # simulation settings  # 注释：仿真设置
        self.sim.dt = 0.005  # 仿真时间步长：每个物理步骤的时间间隔为0.005秒（200Hz）
        self.sim.render_interval = self.decimation  # 渲染间隔：每4个物理步骤渲染一次
        self.sim.physics_material = self.scene.terrain.physics_material  # 物理材质：使用场景地形的物理材质
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # PhysX GPU最大刚体补丁数：设置GPU上最大刚体补丁数量为10*2^15=327680
        # update sensor update periods  # 注释：更新传感器更新周期
        # we tick all the sensors based on the smallest update period (physics update period)  # 注释：我们根据最小的更新周期（物理更新周期）来触发所有传感器
        if self.scene.height_scanner is not None:  # 如果高度扫描器存在
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 高度扫描器更新周期：设置为抽取率乘以时间步长（与控制频率一致）
        if self.scene.contact_forces is not None:  # 如果接触力传感器存在
            self.scene.contact_forces.update_period = self.sim.dt  # 接触力传感器更新周期：设置为物理时间步长（每个物理步骤更新）

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator  # 注释：检查是否启用了地形等级课程学习，如果启用，则启用地形生成器的课程学习
        # this generates terrains with increasing difficulty and is useful for training  # 注释：这会生成难度递增的地形，对训练很有用
        if getattr(self.curriculum, "terrain_levels", None) is not None:  # 如果课程学习配置中有地形等级项
            if self.scene.terrain.terrain_generator is not None:  # 如果场景有地形生成器
                self.scene.terrain.terrain_generator.curriculum = True  # 启用地形生成器的课程学习
        else:  # 否则
            if self.scene.terrain.terrain_generator is not None:  # 如果场景有地形生成器
                self.scene.terrain.terrain_generator.curriculum = False  # 禁用地形生成器的课程学习

    def disable_zero_weight_rewards(self):  # 禁用零权重奖励方法：将权重为0的奖励项设置为None以优化性能
        """If the weight of rewards is 0, set rewards to None"""  # 方法的文档字符串：说明如果奖励权重为0，则将奖励设置为None
        for attr in dir(self.rewards):  # 遍历奖励配置的所有属性
            if not attr.startswith("__"):  # 如果不是私有属性（不以__开头）
                reward_attr = getattr(self.rewards, attr)  # 获取奖励属性对象
                if not callable(reward_attr) and reward_attr.weight == 0:  # 如果属性可调用且权重为0
                    setattr(self.rewards, attr, None)  # 将奖励属性设置为None




@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        # Reduce number of environments for smaller GPUs (7-8 GiB)
        # Original: 4096 environments, reduced to 1024 for GPU memory constraints
        self.scene.num_envs = 1024
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
