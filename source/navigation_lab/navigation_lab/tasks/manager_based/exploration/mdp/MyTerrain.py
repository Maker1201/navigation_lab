"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg

PLANE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(50.0, 50.0),
    border_width=3.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
    },
)

map_range = [20.0, 20.0, 4.5]
HfFOREST_TERRAINS_CFG=TerrainGeneratorCfg(
    seed=0,
    size=(40, 40), 
    border_width=5.0,
    num_rows=1, 
    num_cols=1, 
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=False,
    color_scheme="none",
    sub_terrains={
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            horizontal_scale=0.1,
            vertical_scale=0.1,
            border_width=0.0,
            num_obstacles=600,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.1),
            obstacle_height_range=(1.0, 6.0),
            platform_width=0.0,
        ),
    },
)

FOREST_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0, # adjustable, used for save and load terrains
    size=(50.0, 10.0),  # 每块 50m(x) x 10m(y)
    border_width=0.0,
    num_rows=1,   # 1行
    num_cols=5,   # 5列 → 总共 50m(x) x 50m(y)
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=True,  # ✅ 启用 curriculum，确保5种地形都出现
    sub_terrains={
        # 第1列 - 5个障碍物
        "level_1": terrain_gen.MeshForestTerrainCfg(
            proportion=0.20, 
            obstacle_height_range=(2.0, 6.0), 
            obstacle_radius_range=(0.25, 1.0), 
            num_obstacles=10,
            flat_patch_sampling = {
                "init_pos": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
                "target": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
            }
        ),
        # 第2列 - 10个障碍物
        "level_2": terrain_gen.MeshForestTerrainCfg(
            proportion=0.20,      #比例 
            obstacle_height_range=(2.0, 6.0), 
            obstacle_radius_range=(0.25, 1.0),     #半径
            num_obstacles=40,
            flat_patch_sampling = {
                "init_pos": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
                "target": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
            }
        ),
        # 第3列 - 30个障碍物
        "level_3": terrain_gen.MeshForestTerrainCfg(
            proportion=0.20, 
            obstacle_height_range=(2.0, 6.0), 
            obstacle_radius_range=(0.25, 1.0), 
            num_obstacles=80,
            flat_patch_sampling = {
                "init_pos": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
                "target": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
            }
        ),
        # 第4列 - 60个障碍物
        "level_4": terrain_gen.MeshForestTerrainCfg(
            proportion=0.20, 
            obstacle_height_range=(2.0, 6.0), 
            obstacle_radius_range=(0.25, 1.0), 
            num_obstacles=100,
            flat_patch_sampling = {
                "init_pos": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
                "target": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
            }
        ),
        # 第5列 - 100个障碍物
        "level_5": terrain_gen.MeshForestTerrainCfg(
            proportion=0.20, 
            obstacle_height_range=(2.0, 6.0), 
            obstacle_radius_range=(0.25, 1.0), 
            num_obstacles=120,
            flat_patch_sampling = {
                "init_pos": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
                "target": FlatPatchSamplingCfg(num_patches=200, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
            }
        ),
    },
)
"""Forest terrains configuration.
关键采样逻辑：flat_patch_sampling
    这是该配置最核心的部分。因为地形中布满了 80 个障碍物，如果机器人随机出生，极有可能直接出生在柱子里面导致仿真崩溃。这个参数的作用是**“在乱石阵中找空地”**。
通用子参数 (FlatPatchSamplingCfg):
num_patches=200:
    系统会尝试在地形中寻找 200 个符合条件的平坦候选区域。
patch_radius=0.5:
    平地半径：要求的空地至少要有 0.5 米半径的圆圈范围内没有障碍物，以容纳机器人的底盘。
max_height_diff=0.05:
    高度差阈值：在该圆圈范围内，地面的起伏不能超过 5 厘米。这保证了采样点是真正的“平地”。
z_range=(-100.0, 1.0):
    高度筛选：只在海拔 -100 米到 1 米之间的区域寻找平地。这通常是为了过滤掉障碍物的顶端（如果你不设上限，机器人可能会采样到柱子顶上去）。
两个具体采样目标：
    init_pos (初始位置):
        定义了机器人出生点的采样规则。它确保机器人不会“卡”在柱子里出生。
    target (目标点):
        定义了机器人**需要到达的目标（Goal）**的采样规则。这保证了教练（Command Manager）给出的目标点是可达的，而不是把目标设在柱子中心。
地形布局 (1x5 网格，每块 10m x 50m，总共 50m x 50m):
+------+------------+--------+--------+-------+
| flat | very_sparse| sparse | medium | dense |
| (0)  |   (10)     |  (30)  |  (60)  | (100) |
+------+------------+--------+--------+-------+
  10m      10m        10m      10m      10m
          ← 障碍物密度递增 →
"""