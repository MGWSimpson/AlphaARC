import numpy as np

class ReplayBuffer():
    def __init__(self, capacity, input_shape, n_actions):

        self.capacity = capacity
        self.counter = 0

        self.state = np.zeros((self.capacity, input_shape))
        self.new_state = np.zeros((self.capacity, input_shape))
        self.action = np.zeros((self.capacity, n_actions)) 
        self.reward = np.zeros(self.capacity)
        self.terminal = np.zeros(capacity, dtype=np.bool)  
        
    def store_transition(self, state, action, reward, state_, done):
        
        index = self.counter % self.capacity
        self.state[index] = state
        self.new_state[index] = state_
        self.action[index] = action 
        self.reward[index] = reward
        self.terminal[index] = done

        self.counter +=1

    def sample(self, batch_size):
        max_mem = min(self.counter, self.capacity)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state[batch]
        states_ = self.new_state[batch]
        actions = self.action[batch]
        rewards = self.reward[batch]
        dones = self.terminal[batch]

        return states, states_, actions, rewards, dones