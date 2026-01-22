import argparse
import torch
from isaaclab.app import AppLauncher

# =========================
# 1. Launch Isaac Sim
# =========================
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# =========================
# 2. Create env
# =========================
from navigation_lab.tasks.manager_based.navigation_go2.navigation_env_cfg_eval import (
    NavigationEnvCfg_EVAL,
)

env_cfg = NavigationEnvCfg_EVAL()
env = NavigationEnv(cfg=env_cfg)

device = env.device

# =========================
# 3. Load navigation policy
# =========================
from rsl_rl.modules import ActorCritic

policy = ActorCritic(
    num_obs=env.num_obs,
    num_privileged_obs=env.num_privileged_obs,
    num_actions=env.num_actions,
).to(device)

policy.load_state_dict(
    torch.load("logs/navigation/checkpoints/model_50000.pt", map_location=device)
)
policy.eval()

# =========================
# 4. Reset
# =========================
obs, _ = env.reset()
obs = obs.to(device)

# =========================
# 5. Inference loop
# =========================
while simulation_app.is_running():

    with torch.no_grad():
        actions = policy.act(obs)

    obs, rewards, dones, infos = env.step(actions)

    if dones.any():
        obs, _ = env.reset()

# =========================
# 6. Close
# =========================
env.close()
simulation_app.close()
