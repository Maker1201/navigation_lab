import math

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils

# ============================
# 1. 验证用 SceneCfg（加载 USD）
# ============================

@configclass
class Go2NavEvalSceneCfg(InteractiveSceneCfg):

    environment = sim_utils.UsdFileCfg(
        usd_path="assets/scenes/substation/substation.usd",
        scale=(1.0, 1.0, 1.0),
    )

    ground = sim_utils.GroundPlaneCfg()

    dome_light = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    )


# ============================
# 2. 验证用 Low-level EnvCfg
# ============================

from navigation_lab.tasks.manager_based.locomotion_go2.locomotion_go2_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

@configclass
class UnitreeGo2NavEvalLowLevelEnvCfg(UnitreeGo2RoughEnvCfg):

    scene = Go2NavEvalSceneCfg(
        num_envs=1,
        env_spacing=0.0,
    )

    def __post_init__(self):
        super().__post_init__()

        # 验证阶段：彻底关闭随机化 / curriculum
        self.randomization = None
        self.curriculum = None


# ============================
# 3. 验证用 Navigation EnvCfg
# ============================

from navigation_lab.tasks.manager_based.navigation_go2.navigation_env_cfg import (
    NavigationEnvCfg,
)

LOW_LEVEL_ENV_CFG = UnitreeGo2NavEvalLowLevelEnvCfg()


@configclass
class NavigationEnvCfg_EVAL(NavigationEnvCfg):
    """
    验证用 Navigation 配置：
    - MDP / reward / observation / action 完全复用训练
    - 只替换 scene 来源
    """

    scene = LOW_LEVEL_ENV_CFG.scene

    def __post_init__(self):
        super().__post_init__()

        # 强制验证参数
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0

        self.curriculum = None
