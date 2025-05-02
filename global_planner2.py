import heapq
from env2 import DynamicObstacleEnv
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt
import copy



env = DynamicObstacleEnv(dt=0.3)
init_obs = env.reset()
done = False

start_state = {
    'curr_obs': copy.deepcopy(init_obs),
    'time': 0.0,
    'path': [copy.deepcopy(env.robot_pos)],
    'obs': [copy.deepcopy(init_obs)],
    'actions': [],
    'cost': 0.0,
    # 'obs': init_obs,
}

time_limit = 100
render = True

open_set = []
heapq.heappush(open_set, (start_state['cost'], start_state))

# env.set_observation(*start_state['curr_obs'])
# env.render()
# with open('obs_0.pkl', 'wb') as f:
#     pkl.dump({'obs': start_state['obs'], 'actions': start_state['actions']}, f)
# print("Initial Agent Position:", env.robot_pos, env.robot_vel)
# print("Initial Goal Position:", env.goal_pos, "\n")
# print("Initial State:", start_state['curr_obs'])

# exit(0)

while open_set:
    print(".", end='')
    _, current = heapq.heappop(open_set)
    current_copy = copy.deepcopy(current)

    env.set_observation(*current['curr_obs'])
    if env.reached_goal():
        print("Final Goal reached!")
        print("Path:", current['path'])
        print("Actions:", current['actions'])
        with open(f'obs_{int(time.time())}.pkl', 'wb') as f:
            pkl.dump({'obs': current['obs'], 'actions': current['actions'], 'path': current['path']}, f)
        
        if render:
            env.render_obs(current['obs'])

        break

    if current['time'] > time_limit:
        print("Time limit reached!")
        break

    feasible_vels, feasible_actions, toward_goal_vel, toward_goal_action = env.get_reachable_velocities()
    if len(feasible_vels) == 0 or len(feasible_actions) == 0:
        continue

    num_sample = 10
    sample_idx = np.random.choice(feasible_actions.shape[0], min(num_sample, feasible_actions.shape[0]), replace=False)
    sampled_actions = feasible_actions[sample_idx]
    # breakpoint()
    
    for action in sampled_actions:
        current = copy.deepcopy(current_copy)
        # breakpoint()
        env.set_observation(*current['curr_obs'])
        obs_new, reward, done, info = env.step(action)

        # print(obs_new[0].shape)
        # break

        cost = current['cost'] + (
            np.linalg.norm(env.robot_pos - env.goal_pos) 
        )
        
        if not env.reached_goal() and done:
            continue

        # if done:
        #     print("Reached Goal!")
        # breakpoint()

        new_state = {
            'curr_obs': copy.deepcopy(obs_new),
            'time': current_copy['time'] + env.dt,
            'path': current_copy['path'] + [copy.deepcopy(env.robot_pos)],
            'obs': current_copy['obs'] + [copy.deepcopy(obs_new)],
            'actions': current_copy['actions'] + [action],
            'cost': cost,
            # 'obs': obs_new,
        }

#         start_state = {
#     'curr_obs': copy.deepcopy(init_obs),
#     'time': 0.0,
#     'path': [copy.deepcopy(env.robot_pos)],
#     'obs': [copy.deepcopy(init_obs)],
#     'actions': [],
#     'cost': 0.0,
#     # 'obs': init_obs,
# }


        heapq.heappush(open_set, (new_state['cost'], new_state))