import numpy as np
import torch
from torch.utils.data import Dataset
from alphaarc.task import Task
from alphaarc.policy.tokenize import tokenize_task




class TrajectoryBuffer(Dataset): 
    def __init__(self, capacity=100_000, n_actions=5, max_task_len=1024,  max_state_len=512, max_action_len=20, n_examples=10):
        self.capacity = capacity
        self.idx = 0
        self.n_actions = n_actions
        self.max_state_len = max_state_len * n_examples
        self.max_action_len = max_action_len * n_examples 
        self.max_task_len = max_task_len * n_examples
        
        self.tasks = np.empty((self.capacity, self.max_task_len), dtype=np.int64)
        self.states = np.empty((self.capacity, self.max_state_len), dtype=np.int64)
        self.actions = np.empty((self.capacity, self.n_actions, self.max_action_len), dtype=np.int64)
        self.action_probs = np.empty((self.capacity, self.n_actions), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)

        self.tasks_lens = np.empty((self.capacity), dtype= np.int16)
        self.state_lens = np.empty((self.capacity), dtype=np.int16)


    def _pad(self, task, state, actions, pad_value=0.0):
        
        print(task.shape[-1])
        padded_task = np.pad(task, pad_width=(0, self.max_task_len - task.shape[-1]), mode='constant', constant_values=pad_value)
        padded_state = np.pad(state, pad_width=(0, self.max_state_len - state.shape[-1]), mode='constant', constant_values=pad_value)
        padded_actions = []
        for action in actions:
            pad_len = self.max_action_len - action.shape[-1]
            padded_action = np.pad(action, pad_width=((0, pad_len)), mode='constant', constant_values=pad_value)
            padded_actions.append(padded_action)
        
        padded_actions = np.stack(padded_actions, axis=0)
        return padded_task, padded_state, padded_actions

    def __len__(self):
        return self.idx 


    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        state = self.states[idx]
        actions = self.actions[idx]
        action_probs = self.action_probs[idx]
        rewards = self.rewards[idx]
        
        task_len = self.tasks_lens[idx]
        state_len = self.state_lens[idx]

        return task[:task_len], state[:state_len], actions, action_probs, rewards
    
    @staticmethod
    def collate_fn(batch, pad_token=0):
        tasks, states, actions, action_probs, rewards = zip(*batch)
        
        longest_task = max(max([len(x) for x in tasks]), 1)
        longest_program = max(max([len(x) for x in states]), 1)
        padded_tasks = [np.pad(x, pad_width=(0, longest_task - x.shape[-1]), mode='constant', constant_values=pad_token) for x in tasks ]
        padded_states = [np.pad(x, pad_width=(0, longest_program - x.shape[-1]), mode='constant', constant_values=pad_token) for x in states]
        
        return (torch.stack([torch.tensor(x) for x in padded_tasks]), 
                  torch.stack([torch.tensor(x) for x in padded_states]),
                  torch.stack([torch.tensor( x) for x in actions]),
                  torch.stack([torch.tensor( x) for x in action_probs]),
                  torch.stack([torch.tensor( x) for x in rewards]))


    def _idx_accounting(self): 
        self.idx +=1
        if self.idx == self.capacity: 
            self.idx = 0

    def _add_sample(self, task, state, action, action_prob, reward, task_len, state_len): 
        self.tasks[self.idx] = task
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.action_probs[self.idx] = action_prob
        self.rewards[self.idx] = reward
        self.tasks_lens[self.idx]= task_len
        self.state_lens[self.idx] = state_len
        self._idx_accounting()

    def add_trajectory(self, trajectory): 
        for sample in trajectory: 
            task, state, actions, action_probs, rewards = sample
            task_len, state_len = task.shape[-1], state.shape[-1]
            task, state, actions = self._pad(task, state, actions)
            self._add_sample(task, state, actions, action_probs, rewards, task_len=task_len, state_len=state_len)
    


class ReplayBuffer(Dataset): 
    def __init__(self, capacity=100_000, max_state_len=2048, max_task_len=2048, n_examples=10): 
        self.idx = 0
        self.max_task_len = max_task_len * n_examples
        self.max_state_len = max_state_len * n_examples
        self.capacity = capacity

        self.tasks = np.empty((self.capacity, self.max_task_len), dtype=np.int64)
        self.states = np.empty((self.capacity, self.max_state_len), dtype=np.int64)
        self.task_lengths = np.empty((self.capacity), dtype=np.int16)
        self.program_lengths = np.empty((self.capacity), dtype=np.int16)

        self.pad_token = 0

    def __len__(self):
        return self.idx 

    def __getitem__(self, idx):
        task_length = self.task_lengths[idx]
        program_length = self.program_lengths[idx]
        return self.tasks[idx][:task_length], self.states[idx][:program_length]
    

    @staticmethod
    def collate_fn(batch, pad_token=0):
        tasks, states = zip(*batch)
        longest_task = max([len(x) for x in tasks])
        longest_program = max([len(x) for x in states])
        padded_tasks = [np.pad(x, pad_width=(0, longest_task - x.shape[-1]), mode='constant', constant_values=pad_token) for x in tasks ]
        padded_states = [np.pad(x, pad_width=(0, longest_program - x.shape[-1]), mode='constant', constant_values=pad_token) for x in states]
        return torch.stack([torch.tensor(x) for x in padded_tasks]), torch.stack([torch.tensor(x) for x in padded_states])

    def _pad(self, task, program):
        padded_program = np.pad(program, pad_width=(0, self.max_state_len - program.shape[-1]), mode='constant', constant_values=self.pad_token)
        padded_task = np.pad(task, pad_width=(0, self.max_task_len - task.shape[-1 ]), mode='constant', constant_values=self.pad_token)
        return padded_task, padded_program


    def _idx_accounting(self): 
        self.idx +=1
        if self.idx == self.capacity: 
            self.idx = 0


    def preload_tasks(self, tasks: list[Task], tokenizer,n_examples, max_state_len, max_task_len ): 
        for task in tasks:
            program_lines = task.program_lines
            encoded_program = tokenizer( program_lines, return_tensors='np')['input_ids'].squeeze()
            encoded_task = np.array(tokenize_task(task, tokenizer, n_examples, max_task_len, max_state_len)['input_ids'])
            self.add_program_and_task(encoded_task, encoded_program)

    def add_program_and_task(self, task, program): 
        task_len, program_len = task.shape[-1], program.shape[-1]        
        task, program = self._pad(task, program)   
        self.tasks[self.idx] = task
        self.states [self.idx]= program
        self.task_lengths[self.idx] = task_len
        self.program_lengths[self.idx] = program_len
        self._idx_accounting()
        
    
