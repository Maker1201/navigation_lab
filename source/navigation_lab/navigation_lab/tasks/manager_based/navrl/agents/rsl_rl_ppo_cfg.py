# 从 isaaclab.utils 模块导入 configclass 装饰器，用于创建配置类
from isaaclab.utils import configclass

# 从 isaaclab_rl.rsl_rl 模块导入PPO算法相关的配置类
# RslRlOnPolicyRunnerCfg: 在线策略运行器配置基类
# RslRlPpoActorCriticCfg: PPO Actor-Critic网络配置类
# RslRlPpoAlgorithmCfg: PPO算法超参数配置类
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


# 使用 configclass 装饰器标记配置类，使其支持配置文件的序列化和反序列化
@configclass
# 定义导航环境的PPO运行器配置类，继承自 RslRlOnPolicyRunnerCfg
class NavigationEnvPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境在每个迭代中收集的步数，用于经验回放缓冲区
    num_steps_per_env = 8
    # 训练的最大迭代次数，达到此值后训练停止
    max_iterations = 4000
    # 模型保存的间隔（每多少次迭代保存一次）
    save_interval = 100
    # 实验名称，用于标识和区分不同的训练实验
    experiment_name = "navrl_navigation"
    # 策略网络（Actor-Critic）的配置
    policy = RslRlPpoActorCriticCfg(
        # 策略网络初始化的噪声标准差，用于探索
        init_noise_std=0.5,
        # Actor网络是否对观察值进行归一化处理
        actor_obs_normalization=False,
        # Critic网络是否对观察值进行归一化处理
        critic_obs_normalization=False,
        # Actor网络隐藏层的维度，[128, 128] 表示两个隐藏层，每层128个神经元
        actor_hidden_dims=[128, 128],
        # Critic网络隐藏层的维度，[128, 128] 表示两个隐藏层，每层128个神经元
        critic_hidden_dims=[128, 128],
        # 激活函数类型，"elu" 表示使用指数线性单元激活函数
        activation="elu",
    )
    # PPO算法的超参数配置
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值损失（value loss）的权重系数，用于平衡策略损失和价值损失
        value_loss_coef=1.0,
        # 是否使用裁剪后的价值损失，有助于稳定训练
        use_clipped_value_loss=True,
        # PPO的裁剪参数（epsilon），限制策略更新的幅度，防止策略变化过大
        clip_param=0.2,
        # 熵系数，鼓励探索，值越大探索性越强
        entropy_coef=0.005, #0.005
        # 每次更新时进行学习的轮数（epochs），即对同一批数据重复训练的次数
        num_learning_epochs=5,
        # 每次更新时的小批量（mini-batch）数量，用于将数据分批处理
        num_mini_batches=4,
        # 学习率，控制参数更新的步长
        learning_rate=1.0e-3,
        # 学习率调度策略，"adaptive" 表示自适应调整学习率
        schedule="adaptive",
        # 折扣因子（gamma），用于计算未来奖励的现值，范围通常在0.9-0.99之间
        gamma=0.99,
        # GAE（Generalized Advantage Estimation）的lambda参数，用于平衡偏差和方差
        lam=0.95,
        # 期望的KL散度值，用于自适应调整学习率，防止策略更新过大
        desired_kl=0.01,
        # 梯度裁剪的最大范数，防止梯度爆炸问题
        max_grad_norm=1.0,
    )
