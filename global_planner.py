import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from env2 import DynamicObstacleEnv

env = DynamicObstacleEnv()
init_obs = env.reset()

done = False

dp_map = {}

# print(init_obs.tobytes())

def func(obs):

    if dp_map.get(obs.tobytes()) is not None:
        return dp_map[obs.tobytes()]

    env.set_observation(obs)
    feasible_vels, feasible_actions, toward_goal_vel, toward_goal_action = env.get_reachable_velocities()

    if len(feasible_vels) == 0 or len(feasible_actions) == 0:
        return -10000000, None

    num_sample = 10

    # uniform sample from feasible_actions array
    sample_idx = np.random.choice(feasible_actions.shape[0], min(num_sample, feasible_actions.shape[0]), replace=False)
    sampled_actions = feasible_actions[sample_idx]

    max_reward = -10000000
    max_action = None

    for action in sampled_actions:
        env.set_observation(obs)
        obs_new, reward, done, info = env.step(action)
        if env.reached_goal():
            return reward, [action]
        elif done:
            return -10000000, None
        
        acc_reward, actions = func(obs_new)
        tmp_rew = reward + acc_reward
        if actions == None:
            continue

        if tmp_rew > max_reward:
            max_reward = tmp_rew
            max_action = actions.append(action)


    dp_map[obs.tobytes()] = (max_reward, max_action)
    return max_reward, max_action


max_reward, max_action = func(init_obs)
print("Max Reward:", max_reward)
print("Max Action:", max_action)