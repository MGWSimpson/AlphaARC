import numpy as np
import torch


class ReplayBuffer(): 
    def __init__(self, sample_batch_size=8):
        self.history = []
        self.sample_batch_size = sample_batch_size

    
    def add(self, new_data): 
        self.history.extend(new_data)

    def sample(self):
        sample_ids = np.random.randint(len(self.history), size=self.sample_batch_size)
        states, action_probs, values = list(zip(*[self.history[i] for i in sample_ids]))

   
        return states, action_probs, values