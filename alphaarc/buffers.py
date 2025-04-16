import numpy as np
import torch
from torch.utils.data import Dataset

 



"""
Stores the trajectories of a task.
"""
class TrajectoryBuffer(Dataset): 
    def __init__(self, capacity=100_000, n_actions=5, max_task_size=1024,  max_state_size=512, max_action_size=20):
        self.capacity = capacity
        self.idx = 0
        self.n_actions = n_actions
        self.max_state_size = max_state_size
        self.max_action_size = max_action_size
        self.max_task_size = max_task_size
        self.tasks = np.empty((self.capacity, self.max_task_size), dtype=np.int64)
        self.states = np.empty((self.capacity, self.max_state_size), dtype=np.int64)
        self.actions = np.empty((self.capacity, self.n_actions, self.max_action_size), dtype=np.int64)
        self.action_probs = np.empty((self.capacity, self.n_actions), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
    
    def _pad(self, task, state, actions, pad_value=0.0):
    
        padded_task = np.pad(task, pad_width=(0, self.max_task_size - task.shape[-1]), mode='constant', constant_values=pad_value)
        padded_state = np.pad(state, pad_width=(0, self.max_state_size - state.shape[-1]), mode='constant', constant_values=pad_value)
        padded_actions = []
        for action in actions:
            pad_len = self.max_action_size - action.shape[-1]
            padded_action = np.pad(action, pad_width=((0, pad_len)), mode='constant', constant_values=pad_value)
            padded_actions.append(padded_action)
        
        padded_actions = np.stack(padded_actions, axis=0)
        return padded_task, padded_state, padded_actions

    def __len__(self):
        return self.idx 


    # shrink the tensors to the largest non-padded within the batch.
    def _unpad(self, task, state, actions, pad_value): 
        # TODO: write this.
        pass
    
    def __getitem__(self, idx):
        task = torch.tensor(self.tasks[idx])
        state = torch.tensor(self.states[idx])
        actions = torch.tensor ( self.actions[idx])
        action_probs = torch.tensor( self.action_probs[idx])
        rewards = torch.tensor(self.rewards[idx])
        return task, state, actions, action_probs, rewards
    


    def _idx_accounting(self): 
        self.idx +=1

        if self.idx == self.capacity: 
            self.idx = 0

    def _add_sample(self, task, state, action, action_prob, reward): 
        
        self.tasks[self.idx] = task
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.action_probs[self.idx] = action_prob
        self.rewards[self.idx] = reward

        self._idx_accounting()

    def add_trajectory(self, trajectory): 
        for sample in trajectory: 
            task, state, actions, action_probs, rewards = sample
            task, state, actions = self._pad(task, state, actions)
            self._add_sample(task, state, actions, action_probs, rewards)
    


class ReplayBuffer(Dataset): 
    def __init__(self, capacity=100_000, max_state_len=512, max_task_len=1024): 
        self.idx = 0
        
        self.max_task_len = max_task_len
        self.max_state_len = max_state_len
        self.capacity = capacity

        self.tasks = np.empty((self.capacity, self.max_task_len), dtype=np.int64)
        self.states = np.empty((self.capacity, self.max_state_len), dtype=np.int64)

    def __len__(self):
        return self.idx 

    def __getitem__(self, idx):
        return torch.tensor(self.tasks[idx]), torch.tensor(self.states[idx])
    
    def _pad(self, task, program):
        padded_program = np.pad(program, pad_width=(0, self.max_state_len - program.shape[-1]), mode='constant', constant_values=0)
        padded_task = np.pad(task, pad_width=(0, self.max_task_len - task.shape[-1 ]), mode='constant', constant_values=0)
        return padded_task, padded_program


    def _idx_accounting(self): 
        self.idx +=1

        if self.idx == self.capacity: 
            self.idx = 0


    def add_program_and_task(self, task, program): 
        task, program = self._pad(task, program)   

        self.tasks[self.idx] = task
        self.states [self.idx]= program
        

        self._idx_accounting()
        
    
