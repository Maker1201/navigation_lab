"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
import isaaclab.terrains.trimesh as mesh_gen


from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg


PLANE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(50.0, 50.0),
    border_width=15.0,
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
    size=(map_range[0]*2, map_range[1]*2), 
    border_width=5.0,
    num_rows=1, 
    num_cols=1, 
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=False,
    color_scheme="height",
    sub_terrains={
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            horizontal_scale=0.1,
            vertical_scale=0.1,
            border_width=0.0,
            num_obstacles=500,
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
            proportion=0.20, 
            obstacle_height_range=(2.0, 6.0), 
            obstacle_radius_range=(0.25, 1.0), 
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

地形布局 (1x5 网格，每块 10m x 50m，总共 50m x 50m):
+------+------------+--------+--------+-------+
| flat | very_sparse| sparse | medium | dense |
| (0)  |   (10)     |  (30)  |  (60)  | (100) |
+------+------------+--------+--------+-------+
  10m      10m        10m      10m      10m
          ← 障碍物密度递增 →
"""