import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer

from alphaarc.policy.environment import execute_candidate_program
from alphaarc.task import Task
from alphaarc.env import LineLevelArcEnv
from alphaarc.mcts import MCTS
import os
import torch.optim as optim
import torch
from alphaarc.env import LineLevelArcEnv
from alphaarc.curriculum import Curriculum
from alphaarc.buffers import ReplayBuffer
from alphaarc.networks import PolicyValueNetwork
import torch.nn.functional as F
from tqdm import tqdm

from dataclasses import dataclass

@dataclass
class AlphaARCConfig:
    batch_size: int = 2 
    model_path: str = 'alphaarc/pretrained/last.ckpt.dir'
    tokenizer_path: str = 'Salesforce/codet5-small'
    model_temperature: float = 0.95
    model_samples: int = 5
    
    n_episodes_per_task: int = 5
    n_simulations: int = 10
    n_training_iterations: int = 100
    action_temperature: float = 0.95

def pad_and_convert(states, actions, pad_value=0.0, device='cuda'):
    max_state_seq_length = max(state.shape[-1] for state in states)
    padded_states = []
    for state in states:
        pad_len = max_state_seq_length - state.shape[-1]
        padded_state = np.pad(state, pad_width=(0, pad_len), mode='constant', constant_values=pad_value)
        padded_states.append(padded_state)
    padded_states = np.stack(padded_states, axis=0)
    max_action_seq_length = max(action.shape[-1] for action in actions)
    padded_actions = []
    for action in actions:
        pad_len = max_action_seq_length - action.shape[-1]
        padded_action = np.pad(action, pad_width=((0, 0), (0, pad_len)), mode='constant', constant_values=pad_value)
        padded_actions.append(padded_action)
    padded_actions = np.stack(padded_actions, axis=0)
    
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
        
    def execute_episode(self, env, temperature): 
        
        state = env.reset()
        train_examples = []
        terminated = False
        
        while not terminated:
            self.mcts = MCTS(env , n_simulations=self.n_simulations)
            root = self.mcts.run(self.model, state)
            print(env._decode(root.state))
            actions = root.child_actions
            action_probs = [v.visit_count for v in root.children]
            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((state, actions, action_probs))
            action = root.select_action(temperature=temperature)
            state, reward, terminated = env.step(action=action, state=state)
            if terminated:
                ret = []
                solved = (reward == 1.0)
                for hist_state, hist_actions,  hist_action_probs in train_examples:
                    # [state, actions,  actionProbabilities, Reward]
                    # NOTE: It may be theoretically better to store each transition seperately.  
                    ret.append((np.concatenate((env.reset(), hist_state)), hist_actions, hist_action_probs, reward))
                return ret, solved


    def evaluate(self, env):
        episode_history, solved = self.execute_episode(env, 0)
        return int(solved)

    def learn(self, env): 
        task_solved = False
        for eps in range(self.n_episodes):
            episode_history, solved = self.execute_episode(env, self.action_temperature)
            self.replay_buffer.add(episode_history)
            if solved:
                task_solved = True
                break 
        
        print(task_solved)
        # self.train() # TODO: where to train?
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    task = Task.from_json('data/training/42a50994.json')
    print(task.program)
    config = AlphaARCConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    replay_buffer =  ReplayBuffer(config.batch_size)
    model = PolicyValueNetwork( config. model_path, config.tokenizer_path, config.model_temperature, num_samples=config.model_samples)
    model.to('cuda')
    agent = Agent(replay_buffer, model, config.n_episodes_per_task, config.n_simulations, config.n_training_iterations, config.action_temperature)
    env = LineLevelArcEnv(task, tokenizer)
    print(agent.learn(env))