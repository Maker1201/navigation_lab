# navrl_env_cfg.py
from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# scene assets
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg

# MDP
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as TermTerm,
    EventTermCfg as EventTerm,
    CurriculumTermCfg as CurrTerm,
)

from isaaclab.terrains import TerrainImporterCfg
from . import mdp
from isaaclab.commands import UniformPose2dCommandCfg

# -----------------------------------------------------------------------------
# Scene definition
# -----------------------------------------------------------------------------

@configclass
class NavRLSceneCfg(InteractiveSceneCfg):

    plane = TerrainImporterCfg(
        prim_path


    )
    # -------------------- Robot --------------------
    robot: AssetCfg = MISSING

    # -------------------- Goal marker --------------------
    goal_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        markers={
            "goal": {
                "type": "sphere",
                "radius": 0.12,
                "color": (0.1, 0.9, 0.1),
            }
        },
    )

    # -------------------- Raycast (LiDAR) --------------------
    raycast: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        pattern="circular",
        attach_yaw_only=True,
        resolution=1,
        max_distance=5.0,
        num_rays=360,
        debug_vis=False,
    )

    # -------------------- Contact forces --------------------
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=1,
        track_air_time=True,
    )


# -----------------------------------------------------------------------------
# MDP: Observations
# -----------------------------------------------------------------------------

@configclass
class NavRLObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        # robot state
        base_lin_vel = ObsTerm(func="base_lin_vel")
        base_ang_vel = ObsTerm(func="base_ang_vel")
        projected_gravity = ObsTerm(func="projected_gravity")

        # joints
        joint_pos = ObsTerm(func="joint_pos")
        joint_vel = ObsTerm(func="joint_vel")

        # goal
        goal_position = ObsTerm(func="goal_position_in_robot_frame")

        # lidar
        raycast_distances = ObsTerm(func="raycast_distances")

        # contact
        contact_forces = ObsTerm(func="net_contact_forces")
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# MDP: Actions
# -----------------------------------------------------------------------------

@configclass
class NavRLActionsCfg:
    joint_vel = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=".*",
        scale=1.0,
    )


# -----------------------------------------------------------------------------
# MDP: Commands (Goal sampling)
# -----------------------------------------------------------------------------

@configclass
class NavRLCommandsCfg:
    goal = UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        ranges=UniformPose2dCommandCfg.Ranges(
            x=(-4.0, 4.0),
            y=(-4.0, 4.0),
            heading=(-math.pi, math.pi),
        ),
        debug_vis=True,
    )


# -----------------------------------------------------------------------------
# MDP: Rewards (NavRL-style)
# -----------------------------------------------------------------------------

@configclass
class NavRLRewardsCfg:

    # distance to goal
    goal_progress = RewTerm(
        func="goal_distance_progress",
        weight=3.0,
    )

    # heading alignment
    heading_alignment = RewTerm(
        func="heading_to_goal",
        weight=1.0,
    )

    # collision penalty
    collision = RewTerm(
        func="contact_force_penalty",
        weight=-2.0,
    )

    # smooth action
    action_smoothness = RewTerm(
        func="action_rate_l2",
        weight=-0.05,
    )

    # alive bonus
    alive = RewTerm(
        func="alive",
        weight=0.5,
    )


# -----------------------------------------------------------------------------
# MDP: Terminations
# -----------------------------------------------------------------------------

@configclass
class NavRLTerminationsCfg:

    timeout = TermTerm(
        func="time_out",
        time_out=True,
    )

    fall = TermTerm(
        func="is_fallen",
    )

    goal_reached = TermTerm(
        func="goal_reached",
        threshold=0.3,
    )


# -----------------------------------------------------------------------------
# MDP: Events (reset randomization)
# -----------------------------------------------------------------------------

@configclass
class NavRLEventsCfg:

    reset_robot = EventTerm(
        func="reset_root_state_uniform",
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "yaw": (-math.pi, math.pi),
            },
        },
    )

    reset_joints = EventTerm(
        func="reset_joints_by_scale",
        mode="reset",
        params={"position_scale": 0.1, "velocity_scale": 0.1},
    )


# -----------------------------------------------------------------------------
# MDP: Curriculum
# -----------------------------------------------------------------------------

@configclass
class NavRLCurriculumCfg:

    increase_goal_range = CurrTerm(
        func="increase_command_range",
        params={
            "command_name": "goal",
            "max_range": 6.0,
            "steps": 2_000_000,
        },
    )


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

@configclass
class NavRLEnvCfg(ManagerBasedRLEnvCfg):

    # scene
    scene: NavRLSceneCfg = NavRLSceneCfg(
        num_envs=1024,
        env_spacing=4.0,
    )

    # MDP
    observations = NavRLObservationsCfg()
    actions = NavRLActionsCfg()
    commands = NavRLCommandsCfg()
    rewards = NavRLRewardsCfg()
    terminations = NavRLTerminationsCfg()
    events = NavRLEventsCfg()
    curriculum = NavRLCurriculumCfg()

    # -------------------- Robot binding --------------------
    def __post_init__(self):

        # Unitree Go2 (按你要求的方式)
        self.scene.robot = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # simulation
        self.sim.dt = 0.005
        self.sim.render_interval = 4

        # episode
        self.episode_length_s = 20.0

        # viewer
        self.viewer.eye = (6.0, 6.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
