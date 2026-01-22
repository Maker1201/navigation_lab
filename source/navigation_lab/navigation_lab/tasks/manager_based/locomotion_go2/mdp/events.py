from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用前向引用
from isaaclab.managers import EventTermCfg as EventTerm
import random
import torch  # 导入PyTorch库，用于张量操作和数学计算
from typing import TYPE_CHECKING, Literal  # 导入TYPE_CHECKING和Literal，用于类型检查和字面量类型

import isaaclab.utils.math as math_utils  # 导入Isaac Lab数学工具模块，用于数学计算（如采样、四元数操作）
from isaaclab.assets import Articulation, RigidObject  # 导入关节系统和刚体对象类
from isaaclab.managers import SceneEntityCfg  # 导入场景实体配置类

from .utils import is_env_assigned_to_terrain  # 从utils模块导入判断环境是否分配给特定地形的函数

if TYPE_CHECKING:  # 仅在类型检查时执行（运行时不会执行）
    from isaaclab.envs import ManagerBasedEnv  # 导入基于管理器的环境类（仅用于类型注解）


def randomize_rigid_body_inertia(  # 随机化刚体惯性张量函数
    env: ManagerBasedEnv,  # 环境对象
    env_ids: torch.Tensor | None,  # 环境ID张量，None表示所有环境
    asset_cfg: SceneEntityCfg,  # 资产配置（指定要随机化的资产和身体）
    inertia_distribution_params: tuple[float, float],  # 惯性分布参数（如均匀分布的最小值和最大值）
    operation: Literal["add", "scale", "abs"],  # 操作类型：添加、缩放或绝对设置
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",  # 分布类型，默认为均匀分布
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    """通过添加、缩放或设置随机值来随机化身体的惯性张量。

    此函数允许仅随机化身体的对角线惯性张量分量（xx, yy, zz）。
    函数从给定的分布参数中采样随机值，并根据操作将值添加、缩放或设置到物理仿真中。

    .. tip::
        此函数使用CPU张量来分配身体惯性。建议仅在环境初始化期间使用此函数。
    """
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象（刚体或关节系统）

    # resolve environment ids
    # 解析环境ID
    if env_ids is None:  # 如果环境ID为None
        env_ids = torch.arange(env.scene.num_envs, device="cpu")  # 创建包含所有环境ID的张量（在CPU上）
    else:  # 如果环境ID不为None
        env_ids = env_ids.cpu()  # 将环境ID张量移动到CPU

    # resolve body indices
    # 解析身体索引
    if asset_cfg.body_ids == slice(None):  # 如果配置中的身体ID为None（表示所有身体）
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")  # 创建包含所有身体索引的张量
    else:  # 如果指定了特定的身体ID
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")  # 将身体ID列表转换为张量

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    # 获取身体的当前惯性张量（资产数，身体数，9个元素用于关节系统或刚体对象）
    inertias = asset.root_physx_view.get_inertias()  # 从物理视图获取惯性张量

    # apply randomization on default values
    # 在默认值上应用随机化
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()  # 将默认惯性值复制到惯性张量中（为随机化做准备）

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    # 随机化每个对角线元素（xx, yy, zz -> 索引 0, 4, 8）
    for idx in [0, 4, 8]:  # 遍历惯性张量的对角线元素索引（xx, yy, zz）
        # Extract and randomize the specific diagonal element
        # 提取并随机化特定的对角线元素
        randomized_inertias = _randomize_prop_by_op(  # 调用内部辅助函数进行随机化
            inertias[:, :, idx],  # 提取特定对角线元素的所有值
            inertia_distribution_params,  # 分布参数
            env_ids,  # 环境ID
            body_ids,  # 身体ID
            operation,  # 操作类型
            distribution,  # 分布类型
        )
        # Assign the randomized values back to the inertia tensor
        # 将随机化后的值分配回惯性张量
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias  # 更新惯性张量中的对应元素

    # set the inertia tensors into the physics simulation
    # 将惯性张量设置到物理仿真中
    asset.root_physx_view.set_inertias(inertias, env_ids)  # 将更新后的惯性张量写入物理仿真


def randomize_com_positions(  # 随机化质心位置函数
    env: ManagerBasedEnv,  # 环境对象
    env_ids: torch.Tensor | None,  # 环境ID张量，None表示所有环境
    asset_cfg: SceneEntityCfg,  # 资产配置（指定要随机化的资产和身体）
    com_distribution_params: tuple[float, float],  # 质心分布参数（如均匀分布的最小值和最大值）
    operation: Literal["add", "scale", "abs"],  # 操作类型：添加、缩放或绝对设置
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",  # 分布类型，默认为均匀分布
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    """随机化刚体的质心（COM）位置。

    此函数允许随机化物理仿真中身体的质心位置。位置可以通过添加、缩放或设置从指定分布中采样的随机值来随机化。

    .. tip::
        此函数用于初始化或离线调整，因为它直接修改物理属性。

    参数：
        env (ManagerBasedEnv): 仿真环境。
        env_ids (torch.Tensor | None): 要应用随机化的特定环境索引，或None表示所有环境。
        asset_cfg (SceneEntityCfg): 要随机化质心的目标资产的配置。
        com_distribution_params (tuple[float, float]): 分布的参数（例如，均匀分布的最小值和最大值）。
        operation (Literal["add", "scale", "abs"]): 用于随机化的操作。
        distribution (Literal["uniform", "log_uniform", "gaussian"]): 从中采样随机值的分布。
    """
    # Extract the asset (Articulation or RigidObject)
    # 提取资产（关节系统或刚体对象）
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象

    # Resolve environment indices
    # 解析环境索引
    if env_ids is None:  # 如果环境ID为None
        env_ids = torch.arange(env.scene.num_envs, device="cpu")  # 创建包含所有环境ID的张量（在CPU上）
    else:  # 如果环境ID不为None
        env_ids = env_ids.cpu()  # 将环境ID张量移动到CPU

    # Resolve body indices
    # 解析身体索引
    if asset_cfg.body_ids == slice(None):  # 如果配置中的身体ID为None（表示所有身体）
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")  # 创建包含所有身体索引的张量
    else:  # 如果指定了特定的身体ID
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")  # 将身体ID列表转换为张量

    # Get the current COM offsets (num_assets, num_bodies, 3)
    # 获取当前质心偏移量（资产数，身体数，3个维度）
    com_offsets = asset.root_physx_view.get_coms()  # 从物理视图获取质心偏移量

    for dim_idx in range(3):  # Randomize x, y, z independently  # 独立随机化x、y、z维度
        randomized_offset = _randomize_prop_by_op(  # 调用内部辅助函数进行随机化
            com_offsets[:, :, dim_idx],  # 提取特定维度的所有值
            com_distribution_params,  # 分布参数
            env_ids,  # 环境ID
            body_ids,  # 身体ID
            operation,  # 操作类型
            distribution,  # 分布类型
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]  # 更新质心偏移量中的对应维度

    # Set the randomized COM offsets into the simulation
    # 将随机化后的质心偏移量设置到仿真中
    asset.root_physx_view.set_coms(com_offsets, env_ids)  # 将更新后的质心偏移量写入物理仿真



def randomize_obstacles(env, env_ids):
    scene = env.scene

    # ========== 静态障碍物随机化 ==========
    for i in range(env.cfg.scene.num_static_obstacles):
        obj = scene["static_obstacles"]
        x = random.uniform(*env.cfg.scene.obstacle_spawn_range)
        y = random.uniform(*env.cfg.scene.obstacle_spawn_range)
        size = random.uniform(*env.cfg.scene.obstacle_size_range)

        obj.set_world_pose(
            position=torch.tensor([x, y, size / 2], device=env.device)
        )
        obj.set_scale(torch.tensor([size, size, size], device=env.device))

    # ========== 动态障碍物随机化 ==========
    for i in range(env.cfg.scene.num_dynamic_obstacles):
        obj = scene["dynamic_obstacles"]
        x = random.uniform(*env.cfg.scene.obstacle_spawn_range)
        y = random.uniform(*env.cfg.scene.obstacle_spawn_range)

        obj.set_world_pose(
            position=torch.tensor([x, y, 0.2], device=env.device)
        )

"""
Internal helper functions.
"""
"""
内部辅助函数。
"""


def _randomize_prop_by_op(  # 根据操作随机化属性的内部辅助函数
    data: torch.Tensor,  # 要随机化的数据张量
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],  # 分布参数（可以是浮点数或张量）
    dim_0_ids: torch.Tensor | None,  # 第一维的索引，None表示所有
    dim_1_ids: torch.Tensor | slice,  # 第二维的索引，可以是张量或切片
    operation: Literal["add", "scale", "abs"],  # 操作类型：添加、缩放或绝对设置
    distribution: Literal["uniform", "log_uniform", "gaussian"],  # 分布类型
) -> torch.Tensor:  # 返回随机化后的数据张量
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    """根据给定的操作和分布执行数据随机化。

    参数：
        data: 要随机化的数据张量。形状为(dim_0, dim_1)。
        distribution_parameters: 从中采样值的分布的参数。
        dim_0_ids: 要随机化的第一维索引。
        dim_1_ids: 要随机化的第二维索引。
        operation: 对数据执行的操作。选项：'add'、'scale'、'abs'。
        distribution: 从中采样随机值的分布。选项：'uniform'、'log_uniform'。

    返回：
        随机化后的数据张量。形状为(dim_0, dim_1)。

    抛出：
        NotImplementedError: 如果不支持操作或分布。
    """
    # resolve shape
    # 解析形状
    # -- dim 0
    # -- 维度0
    if dim_0_ids is None:  # 如果第一维索引为None
        n_dim_0 = data.shape[0]  # 获取第一维的大小
        dim_0_ids = slice(None)  # 设置为切片（表示所有元素）
    else:  # 如果指定了第一维索引
        n_dim_0 = len(dim_0_ids)  # 获取索引的数量
        if not isinstance(dim_1_ids, slice):  # 如果第二维索引不是切片
            dim_0_ids = dim_0_ids[:, None]  # 将第一维索引扩展为列向量（用于广播）
    # -- dim 1
    # -- 维度1
    if isinstance(dim_1_ids, slice):  # 如果第二维索引是切片
        n_dim_1 = data.shape[1]  # 获取第二维的大小
    else:  # 如果第二维索引是张量
        n_dim_1 = len(dim_1_ids)  # 获取索引的数量

    # resolve the distribution
    # 解析分布
    if distribution == "uniform":  # 如果是均匀分布
        dist_fn = math_utils.sample_uniform  # 使用均匀分布采样函数
    elif distribution == "log_uniform":  # 如果是对数均匀分布
        dist_fn = math_utils.sample_log_uniform  # 使用对数均匀分布采样函数
    elif distribution == "gaussian":  # 如果是高斯分布
        dist_fn = math_utils.sample_gaussian  # 使用高斯分布采样函数
    else:  # 如果是未知分布
        raise NotImplementedError(  # 抛出未实现错误
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    # 执行操作
    if operation == "add":  # 如果是添加操作
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)  # 将采样的随机值添加到数据中
    elif operation == "scale":  # 如果是缩放操作
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)  # 将数据乘以采样的随机值
    elif operation == "abs":  # 如果是绝对设置操作
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)  # 将数据设置为采样的随机值
    else:  # 如果是未知操作
        raise NotImplementedError(  # 抛出未实现错误
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data  # 返回随机化后的数据


def reset_root_state_uniform(  # 均匀重置根状态函数
    env: ManagerBasedEnv,  # 环境对象
    env_ids: torch.Tensor,  # 环境ID张量
    pose_range: dict[str, tuple[float, float]],  # 姿态范围字典（键为"x", "y", "z", "roll", "pitch", "yaw"）
    velocity_range: dict[str, tuple[float, float]],  # 速度范围字典（键为"x", "y", "z", "roll", "pitch", "yaw"）
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 资产配置，默认为"robot"
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Note: If "pits" terrain exists, environments on pit terrain will be reset to default state without random
    perturbations to avoid the robot falling into the pit.
    """
    """将资产根状态重置为给定范围内均匀分布的随机位置和速度。

    此函数随机化资产的根位置和速度。

    * 它从给定范围中采样根位置，并将其添加到默认根位置，然后设置到物理仿真中。
    * 它从给定范围中采样根方向，并设置到物理仿真中。
    * 它从给定范围中采样根速度，并设置到物理仿真中。

    函数接受每个轴和旋转的姿态和速度范围字典。字典的键是``x``、``y``、``z``、``roll``、``pitch``和``yaw``。
    值是形式为``(min, max)``的元组。如果字典不包含键，则该轴的位置或速度设置为零。

    注意：如果存在"pits"地形，坑洞地形上的环境将重置为默认状态，不进行随机扰动，以避免机器人掉入坑洞。
    """
    # extract the used quantities (to enable type-hinting)
    # 提取使用的量（以启用类型提示）
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]  # 从场景中获取资产对象

    # Separate pit and non-pit environments
    # 分离坑洞和非坑洞环境
    # Check which environments are assigned to pit terrain (not random reset)
    # 检查哪些环境被分配给坑洞地形（不进行随机重置）
    assigned_to_pits = is_env_assigned_to_terrain(env, "pits")  # 检查每个环境是否分配给坑洞地形
    pit_env_ids = env_ids[assigned_to_pits[env_ids]]  # 获取分配给坑洞的环境ID
    non_pit_env_ids = env_ids[~assigned_to_pits[env_ids]]  # 获取未分配给坑洞的环境ID（使用位非操作）

    # Reset pit environments to default state (no random perturbations)
    # 将坑洞环境重置为默认状态（无随机扰动）
    if len(pit_env_ids) > 0:  # 如果有坑洞环境
        root_states = asset.data.default_root_state[pit_env_ids].clone()  # 克隆默认根状态
        positions = root_states[:, 0:3] + env.scene.env_origins[pit_env_ids]  # 计算位置（默认位置加上环境原点）
        orientations = root_states[:, 3:7]  # 提取方向（四元数）
        velocities = torch.zeros_like(root_states[:, 7:13])  # 创建零速度（与根状态的速度部分形状相同）
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=pit_env_ids)  # 将位置和方向写入仿真（沿最后一个维度连接）
        asset.write_root_velocity_to_sim(velocities, env_ids=pit_env_ids)  # 将速度写入仿真

    # Reset non-pit environments with random perturbations
    # 使用随机扰动重置非坑洞环境
    if len(non_pit_env_ids) > 0:  # 如果有非坑洞环境
        root_states = asset.data.default_root_state[non_pit_env_ids].clone()  # 克隆默认根状态

        # poses
        # 姿态
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]  # 从字典中获取每个轴的姿态范围，如果不存在则使用(0.0, 0.0)
        ranges = torch.tensor(range_list, device=asset.device)  # 将范围列表转换为张量
        rand_samples = math_utils.sample_uniform(  # 从均匀分布中采样随机值
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device  # 采样形状为(环境数, 6)的随机值（6个维度：x, y, z, roll, pitch, yaw）
        )

        positions = root_states[:, 0:3] + env.scene.env_origins[non_pit_env_ids] + rand_samples[:, 0:3]  # 计算位置（默认位置+环境原点+随机偏移）
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])  # 从欧拉角（roll, pitch, yaw）创建方向增量四元数
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)  # 将默认方向与方向增量相乘（组合旋转）
        # velocities
        # 速度
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]  # 从字典中获取每个轴的速度范围，如果不存在则使用(0.0, 0.0)
        ranges = torch.tensor(range_list, device=asset.device)  # 将范围列表转换为张量
        rand_samples = math_utils.sample_uniform(  # 从均匀分布中采样随机值
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device  # 采样形状为(环境数, 6)的随机值（6个维度：x, y, z, roll, pitch, yaw）
        )

        velocities = root_states[:, 7:13] + rand_samples  # 计算速度（默认速度+随机偏移）

        # set into the physics simulation
        # 设置到物理仿真中
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=non_pit_env_ids)  # 将位置和方向写入仿真（沿最后一个维度连接）
        asset.write_root_velocity_to_sim(velocities, env_ids=non_pit_env_ids)  # 将速度写入仿真
