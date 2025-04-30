import heapq
from env2 import DynamicObstacleEnv
import numpy as np
import pickle as pkl
import time



env = DynamicObstacleEnv()
init_obs = env.reset()
done = False

start_state = {
    'curr_obs': init_obs.copy(),
    'time': 0.0,
    # 'path': [env.robot_pos.copy()],
    'obs': [init_obs.copy()],
    'actions': [],
    'cost': 0.0,
    # 'obs': init_obs.copy(),
}

time_limit = 10.0
render = True

open_set = []
heapq.heappush(open_set, (start_state['cost'], start_state))

while open_set:
    _, current = heapq.heappop(open_set)

    env.set_observation(current['curr_obs'])
    if env.reached_goal():
        print("Goal reached!")
        # print("Path:", current['path'])
        print("Actions:", current['actions'])
        with open(f'obs_{int(time.time())}.pkl') as f:
            pkl.dump({'obs': current['obs'], 'actions': current['actions']}, f)
        
        if render:
            env.render_obs(current['obs'])

        break

    if current['time'] > time_limit:
        break

    feasible_vels, feasible_actions, toward_goal_vel, toward_goal_action = env.get_reachable_velocities()
    if len(feasible_vels) == 0 or len(feasible_actions) == 0:
        continue

    num_sample = 50
    sample_idx = np.random.choice(feasible_actions.shape[0], min(num_sample, feasible_actions.shape[0]), replace=False)
    sampled_actions = feasible_actions[sample_idx]
    for action in sampled_actions:
        env.set_observation(current['curr_obs'])
        obs_new, reward, done, info = env.step(action)

        cost = current['cost'] + (
            np.linalg.norm(env.robot_pos - env.goal_pos) * 0.1 +
            env.dt
        )
        
        if not env.reached_goal() and done:
            continue

        if done:
            print("Reached Goal!")

        new_state = {
            'curr_obs': obs_new.copy(),
            'time': current['time'] + env.dt,
            # 'path': current['path'] + [env.robot_pos.copy()],
            'obs': current['obs'] + [obs_new.copy()],
            'actions': current['actions'] + [action],
            'cost': cost,
            # 'obs': obs_new.copy(),
        }

        heapq.heappush(open_set, (new_state['cost'], new_state))