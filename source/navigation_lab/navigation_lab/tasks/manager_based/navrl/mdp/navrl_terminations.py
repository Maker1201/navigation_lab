def collision_termination(env):
    return env.scene["robot"].data.has_contact

def goal_reached(env):
    dist = ...
    return dist < 0.3

def timeout(env):
    return env.episode_length_buf >= env.max_episode_length
