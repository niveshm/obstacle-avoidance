import pickle as pkl
from env2 import DynamicObstacleEnv

env = DynamicObstacleEnv()
env.reset()

with open('obs_1746078611.pkl', 'rb') as f:
    data = pkl.load(f)
    obs = data['obs']
    env.set_observation(*obs[-1])
    print(env.reached_goal())
    # # env.render()
    # # breakpoint()
    # for action in data['actions']:
    #     obs, reward, done, info = env.step(action)
    #     # print("Action:", action)
    #     # print("Reward:", reward)
    #     # print("Done:", done)
    #     env.render()
    #     if done:
    #         break
    # env.render_obs(obs)
