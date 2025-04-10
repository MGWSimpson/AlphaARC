import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer

from alphaarc.policy.environment import execute_candidate_program
from alphaarc.task import Task
from alphaarc.env import LineLevelArcEnv
from alphaarc.mcts import MCTS

import os
import torch.optim as optim
import torch

import torch.nn.functional as F
from tqdm import tqdm


def pad_and_convert(states, actions, pad_value=0.0, device='cuda'):
    # Determine the maximum sequence length for states
    max_state_seq_length = max(state.shape[-1] for state in states)
    
    # Pad each state along the sequence dimension
    padded_states = []
    for state in states:
        pad_len = max_state_seq_length - state.shape[-1]
        # If state is 1D, pad width is (before, after)
        padded_state = np.pad(state, pad_width=(0, pad_len), mode='constant', constant_values=pad_value)
        padded_states.append(padded_state)
    # Convert list to a numpy array (shape: [batch_size, max_state_seq_length])
    padded_states = np.stack(padded_states, axis=0)
    
    # Determine the maximum sequence length for actions
    # We assume all actions have the same number of actions, only the sequence length varies.
    max_action_seq_length = max(action.shape[-1] for action in actions)
    
    # Pad each action along the sequence dimension (last dimension)
    padded_actions = []
    for action in actions:
        pad_len = max_action_seq_length - action.shape[-1]
        # For a 2D array of shape (n_actions, sequence_length), pad only the sequence axis.
        padded_action = np.pad(action, pad_width=((0, 0), (0, pad_len)), mode='constant', constant_values=pad_value)
        padded_actions.append(padded_action)
    # Convert list to a numpy array (shape: [batch_size, n_actions, max_action_seq_length])
    padded_actions = np.stack(padded_actions, axis=0)
    
    # Convert to PyTorch tensors and move to specified device
    states_tensor = torch.LongTensor(padded_states).to(device)
    actions_tensor = torch.LongTensor(padded_actions).to(device)
    
    return states_tensor, actions_tensor


# save.
class Agent(): 
    def __init__(self, replay_buffer, model, n_episodes, n_simulations, n_training_iterations, action_temperature):
        self.n_episodes = n_episodes
        self.n_simulations  = n_simulations
        self.n_training_iterations = n_training_iterations
        self.action_temperature = action_temperature

        self.replay_buffer = replay_buffer
        self.model = model
        
    def execute_episode(self, env): 
        
        state = env.reset()
        train_examples = []
        terminated = False

        while not terminated:
            self.mcts = MCTS(env , n_simulations=self.n_simulations)
            root = self.mcts.run(self.model, state)
            

            actions = root.child_actions
            action_probs = [v.visit_count for v in root.children]
            action_probs = action_probs / np.sum(action_probs)
            
            train_examples.append((state, actions, action_probs))

            action = root.select_action(temperature=0.5)
            state, reward, terminated = env.step(action=action, state=state)

            if terminated:
                ret = []
                solved = (reward == len(env.initial_states))
                for hist_state, hist_actions,  hist_action_probs in train_examples:
                    # [state, actions,  actionProbabilities, Reward]
                    # NOTE: It may be theoretically better to store each transition seperately.  
                    ret.append((np.concatenate((env.reset(), hist_state)), hist_actions, hist_action_probs, reward))
                return ret, solved



    def learn(self, env): 
        task_solved = False

        for eps in range(self.n_episodes):
            episode_history, solved = self.execute_episode(env)
            if solved:
                task_solved
            self.replay_buffer.add(episode_history)

        self.train() # TODO: where to train?
        return int(task_solved)
    
    
    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        self.model.train()

        for i in tqdm(range(self.n_training_iterations)):
            states, actions, action_probs, values = self.replay_buffer.sample()
            target_vs = torch.FloatTensor(np.array(values).astype(np.float64)).to('cuda')
            target_pis = torch.FloatTensor(np.array(action_probs).astype(np.float64)).to('cuda')
            states, actions = pad_and_convert(states, actions, pad_value=self.model.tokenizer.pad_token_type_id)

            predicted_pi = self.model.forward(state=states, actions=actions).to('cuda')
            predicted_vs = self.model.value_forward(state=states).to('cuda')

            policy_loss = F.cross_entropy(predicted_pi, target_pis)
            value_loss = F.mse_loss( predicted_vs, target_vs)

            loss = policy_loss + value_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
    task = Task.from_json('data/training/67385a82.json')
    env = LineLevelArcEnv(task, tokenizer=tokenizer)
    agent = Agent()
    
    agent.learn(env)