def goal_progress(env):
    prev = env.extras["prev_goal_dist"]
    curr = torch.norm(
        env.scene["goal"].data.root_pos_w[:, :2]
        - env.scene["robot"].data.root_pos_w[:, :2],
        dim=-1,
    )
    env.extras["prev_goal_dist"] = curr
    return prev - curr

def goal_reached(env, threshold=0.3):
    dist = torch.norm(
        env.scene["goal"].data.root_pos_w[:, :2]
        - env.scene["robot"].data.root_pos_w[:, :2],
        dim=-1,
    )
    return (dist < threshold).float()

def collision_penalty(env):
    return env.scene["robot"].data.has_contact.float()

def action_smoothness(env):
    prev = env.extras["prev_action"]
    curr = env.actions
    env.extras["prev_action"] = curr
    return torch.sum((curr - prev) ** 2, dim=-1)
