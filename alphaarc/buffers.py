import numpy as np
import torch
from torch.utils.data import Dataset

from alphaarc.utils import pad_and_convert





# will need to add a capacity here . 
class TrajectoryBuffer(Dataset): 
    def __init__(self, capacity=100_000, n_actions=5,  max_n_tokens=1024):
        self.capacity = capacity
        self.idx = 0
        self.n_actions = n_actions
        self.max_n_tokens = max_n_tokens

        self.tasks = np.empty((self.capacity, self.max_n_tokens))
        self.states = np.empty((self.capacity, self.max_n_tokens))
        self.actions = np.empty((self.capacity, self.n_actions, self.max_n_tokens))
        self.action_probs = np.empty((self.capacity, self.n_actions))
        self.rewards = np.empty((self.capacity, 1))
    
 
    def __len__(self):
        return self.idx 

    def __getitem__(self, idx):
        task = torch.tensor(self.tasks[idx])
        state = torch.tensor(self.states[idx])
        actions = torch.tensor ( self.actions[idx])
        action_probs = torch.tensor( self.action_probs[idx])
        rewards = torch.tensor(self.rewards[idx])
        return task, state, actions, action_probs, rewards
    

    def _add_sample(self, task, state, action, action_prob, reward): 
        self.idx +=1

        if self.idx == self.capacity: 
            self.idx = 0

        self.tasks[self.idx] = task
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.action_probs[self.idx] = action_prob
        self.rewards[self.idx] = reward

    def add_trajectory(self, trajectory): 
        for sample in trajectory: 
            task, state, actions, action_probs, rewards = sample
            task, state, actions = pad_and_convert(task, state, actions)
            self._add_sample(task, state, actions, action_probs, rewards)
    


class ReplayBuffer(): 
    pass