# config
import math
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# scene
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg,AssetBaseCfg,RigidObjectCfg
from isaaclab.sensors import (
              ContactSensorCfg,
              RayCasterCfg,
              patterns,
              CameraCfg,
              TiledCameraCfg)
# MDP
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
# ENV
from isaaclab.envs import ManagerBasedRLEnvCfg
# Pre-defined 
from . import mdp
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from navigation_lab.tasks.manager_based.locomotion_go2.locomotion_go2_env_cfg import UnitreeGo2RoughEnvCfg  # 导入Unitree Go2机器人的底层运动环境配置

LOW_LEVEL_ENV_CFG = UnitreeGo2RoughEnvCfg()  # 创建底层环境配置实例，用于获取底层环境的配置参数


# -----------------------------------------------------------------------------
# Scene definition
# -----------------------------------------------------------------------------

@configclass
class MySceneCfg(InteractiveSceneCfg):
    # -------------------- Scene --------------------
    # ground plane  # 注释：地面平面配置
    terrain = TerrainImporterCfg(  # 地面资产的基础配置对象
        prim_path="/World/ground/forest",  # 在USD场景图中的原始路径，指定地面对象的位置
        terrain_type="generator",  # 地形类型：使用生成器模式创建地形
        terrain_generator=mdp.FOREST_TERRAINS_CFG,  # 地形生成器配置：使用预定义的粗糙地形配置
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

    # -------------------- Robot --------------------
    robot: ArticulationCfg = MISSING

    # -------------------- Goal marker --------------------
    # goal_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="{ENV_REGEX_NS}/Goal",
    #     markers={
    #         "goal": {
    #             "type": "sphere",
    #             "radius": 0.12,
    #             "color": (0.1, 0.9, 0.1),
    #         }
    #     },
    # )

    # -------------------- Raycast (LiDAR) --------------------
    lidar_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        ray_alignment="base",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground/forest"],
        debug_vis=True,
    )

    # ------------------------ Light --------------------------
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            # texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # ------------------------ camera --------------------------
    # camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/camera",
    #     update_period=5, # 10Hz
    #     height=64,
    #     width=80,
    #     debug_vis=False,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 8.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(-0.4, 0.0, 0.1), rot=(0.5, -0.5, -0.5, 0.5), convention="ros"),
    # )

    # -------------------- Contact forces --------------------
    # contact_forces_cone = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     history_length=1,
    #     track_air_time=False,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Cone"],
    # )

# -----------------------------------------------------------------------------
# MDP: Observations
# -----------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """ Observations Manager """
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # robot state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),    
        )

        pose_command = ObsTerm(func=mdp.pose_command_position_2d, params={"command_name": "pose_command"})
        actions = ObsTerm(func=mdp.last_action)
        lidar_scan = ObsTerm(
            func=mdp.lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_scanner")},
            clip=(-5.0, 5.0),
        )
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# MDP: Actions
# -----------------------------------------------------------------------------

@configclass
class ActionsCfg:
    # -------------------- JointPosition --------------------
    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=[".*"], 
    #     scale=0.5, 
    #     use_default_offset=True
    # )

    # -------------------- PreTrainedPolicy --------------------
    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path="logs/rsl_rl/unitree_go2_rough/2026-01-28_10-42-25/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )

# -----------------------------------------------------------------------------
# MDP: Commands (Goal sampling)
# -----------------------------------------------------------------------------

@configclass
class CommandsCfg:
    # .在障碍物区域内部随机发布目标点 (3D - 随机高度)
    pose_command = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,  # 不使用简单航向模式（使用完整的方向角）
        resampling_time_range=(180.0, 180.0),
        debug_vis=True,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
            heading=(1.57, 4.71)  # 只设置 heading，pos_x 和 pos_y 由地形自动采样
        ),
    )
    # 给出未来轨迹点
    traj_command = mdp.TrajectoryVisCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.1, 0.1),
        max_length=1000,
        threshold=10.0,  # 增大阈值，只有重置时（位置跳跃>10m）才清除轨迹
    )

# -----------------------------------------------------------------------------
# MDP: Events (reset randomization)
# -----------------------------------------------------------------------------

@configclass
class EventsCfg:
    # -------------------- startup --------------------
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    # -------------------- reset --------------------
    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )




# -----------------------------------------------------------------------------
# MDP: Terminations
# -----------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 超时时重置

    out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": -1.0},
    )

    least_lidar_depth = DoneTerm(
        func=mdp.least_lidar_depth,
        params={"sensor_cfg": SceneEntityCfg("lidar_scanner"), "threshold": 0.12},  # 放宽碰撞阈值 30cm
    )

    roll_over = DoneTerm(
        func=mdp.roll_over,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.2},
    )

    # 接近目标点时终止任务并重置 (只需位置到达，不再要求速度减小)
    reach_target = DoneTerm(
        func=mdp.reach_target,
        params={"threshold": 1, "command_name": "pose_command"},
        # velocity_threshold 已移除，不再要求速度减小
    )

# -----------------------------------------------------------------------------
# MDP: Rewards (NavRL-style)
# -----------------------------------------------------------------------------

@configclass
class RewardsCfg:

    # distance to goal
    goal_progress = RewTerm(
        func=mdp.goal_distance_progress,
        weight=3.0,
        params={"command_name": "pose_command"},
    )

    # heading alignment
    heading_alignment = RewTerm(
        func=mdp.heading_to_goal,
        weight=1.0,
        params={"command_name": "pose_command"},
    )

    # collision penalty
    collision = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-2.0,
    )

    # smooth action
    action_smoothness = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )

    # alive bonus
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.5,
    )


# -----------------------------------------------------------------------------
# MDP: Curriculum
# -----------------------------------------------------------------------------

@configclass
class CurriculumCfg:

    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_pose,
        params={"command_name": "pose_command", "success_threshold": 1.0},
    )


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

@configclass
class NavRLEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=256, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settingsd
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # -------------------- Robot binding --------------------

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [0, -40, 60]
        self.viewer.lookat = [1.5, 0, -5.5]
        
        # general settings
        self.decimation = 10 
        self.episode_length_s = 180.0
        # simulation settings
        self.sim.dt = 0.01  # 增大仿真时间步到0.01秒
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # robot configuration
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.lidar_scanner is not None:
            self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False