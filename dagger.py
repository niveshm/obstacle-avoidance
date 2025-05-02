from env2 import DynamicObstacleEnv
import numpy as np
import torch
import copy
from tqdm import tqdm
from model import DynamicEnvAction, DynamicEnvDataset, create_dataloader_with_dataset
from time import sleep

def collect_trajectory(model, render=False):
    env = DynamicObstacleEnv(dt=0.3)
    init_obs = env.reset()
    # init_obs = np.concatenate([init_obs[0], np.array(init_obs[1])])
    curr_obs = init_obs

    done = False
    observations = [copy.deepcopy(curr_obs)]
    # actions = []

    while not done:
        curr_obs = np.concatenate([curr_obs[0], np.array(curr_obs[1])])
        action = model(torch.tensor(curr_obs, dtype=torch.float32)).detach().numpy()
        obs, reward, done, info = env.step(action)
        curr_obs = obs
        # actions.append(copy.deepcopy(action))
        observations.append(copy.deepcopy(curr_obs))
        if render:
            env.render()
        # env.render()
        if done:
            break
    
    return observations

def collect_actions_for_trajectory(obs):
    env = DynamicObstacleEnv(dt=0.3)
    env.reset()

    observations = []
    actions = []

    for o in tqdm(obs):
        tmp_obs = copy.deepcopy(o)
        env.set_observation(*o)
        if env.is_done():
            continue

        feasible_vels, feasible_actions, toward_goal_vel, toward_goal_action = env.get_reachable_velocities()

        if len(feasible_vels) == 0 or len(feasible_actions) == 0:
            continue
        
        p = np.random.random()
        if p < 0.1:
            action_idx = np.random.choice(feasible_actions.shape[0], 1, replace=False)
            action = feasible_actions[action_idx[0]]
        else:
            action = toward_goal_action
        observations.append(tmp_obs)
        actions.append(action)
    
    return observations, actions



def train_model(model, dataloader, optimizer, criterion):
    for epoch in range(10):
        for i, (obs, action) in enumerate(dataloader):
            optimizer.zero_grad()
            pred_action = model(obs)
            loss = criterion(pred_action, action)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/10], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    return model

    # torch.save(model.state_dict(), "dynamic_env_model.pth")
    # print("Model saved to dynamic_env_model.pth")
    
def dagger():
    model = DynamicEnvAction(input_dim=31, output_dim=2)
    obs_dir = "obs"
    dataset = DynamicEnvDataset(obs_dir)
    
    for i in range(10):
        print(f"*** Iteration {i+1} ***")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        dataloader = create_dataloader_with_dataset(dataset, train=True, batch_size=32)
        if i == 0:
            model = train_model(model, dataloader, optimizer, criterion)
        else:
            for j in range(10):
                obs = collect_trajectory(model)
                observations, actions = collect_actions_for_trajectory(obs)
                dataset.add_obs_actions(observations, actions)
            
            dataloader = create_dataloader_with_dataset(dataset, train=True, batch_size=32)
            # for d in dataloader:
            #     print(d[0].shape, d[1].shape)

            
            model = train_model(model, dataloader, optimizer, criterion)
            sleep(10)
        
        torch.save(model.state_dict(), f"dagger_models/dynamic_env_model_{i}.pth")
        print("*** END OF ITERATION ***")


            


    


if __name__ == "__main__":
    model_path = "dagger_models/dynamic_env_model_9.pth"
    model = DynamicEnvAction(input_dim=31, output_dim=2)
    model.load_state_dict(torch.load(model_path))
    collect_trajectory(model, render=True)
    # dagger()