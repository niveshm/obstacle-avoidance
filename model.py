import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle as pkl
import numpy as np
import os


class DynamicEnvAction(nn.Module):
    ## Predicts the action for a given state
    def __init__(self, input_dim, output_dim):
        super(DynamicEnvAction, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


## Dataloader
class DynamicEnvDataset(Dataset):
    def __init__(self, obs_dir):
        files = os.listdir(obs_dir)
        self.data = []
        for file in files:
            if file.endswith('.pkl'):
                with open(os.path.join(obs_dir, file), 'rb') as f:
                    data = pkl.load(f)
                    for i, obs in enumerate(data['obs']):
                        tmp = np.concatenate([obs[0], np.array(obs[1])])
                        if i == len(data['obs']) - 1:
                            continue
                        self.data.append((tmp, data['actions'][i]))

                    

                    
                    # self.data = [(obs, action) for obs, action in zip(data['obs'], data['actions'])]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs, action = self.data[idx]
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)
    
    def add_obs_actions(self, obs, action):
        # breakpoint()
        cnt = 0
        for i, obs in enumerate(obs):
            tmp = np.concatenate([obs[0], np.array(obs[1])])
            self.data.append((tmp, action[i]))
            cnt += 1
        print(f"Added {cnt} observations and actions to the dataset.")



def collate_fn(batch):
    obs, actions = zip(*batch)
    obs = torch.stack(obs)
    actions = torch.stack(actions)
    return obs, actions

def create_dataloader_with_dataset(dataset, train=False, batch_size=32):
    shuffle = True if train else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def create_dataloader(obs_dir, train=False, batch_size=32):
    dataset = DynamicEnvDataset(obs_dir)
    shuffle = True if train else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    # breakpoint()
    return dataloader

## Training Loop
def train_model(input_dim, output_dim, obs_dir, num_epochs=10):
    model = DynamicEnvAction(input_dim=input_dim, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataloader = create_dataloader(obs_dir, train=True, batch_size=32)

    for epoch in range(num_epochs):
        for i, (obs, action) in enumerate(dataloader):
            optimizer.zero_grad()
            pred_action = model(obs)
            loss = criterion(pred_action, action)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] completed.")
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    

if __name__ == "__main__":
    obs_dir = 'obs'
    model_path = 'dynamic_env_model.pth'
    input_dim = 31
    output_dim = 2
    num_epochs = 10
    model = train_model(input_dim, output_dim, obs_dir, num_epochs)
    save_model(model, model_path)
    print("Model training completed.")



    
