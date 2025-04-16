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
from alphaarc.buffers import ReplayBuffer, TrajectoryBuffer
from alphaarc.networks import PolicyValueNetwork
import torch.nn.functional as F
from tqdm import tqdm
import time
from dataclasses import dataclass
from torch.utils.data import DataLoader 

import lightning.pytorch as pl

@dataclass
class AlphaARCConfig:
    batch_size: int = 2 
    model_path: str = 'alphaarc/pretrained/last.ckpt.dir'
    tokenizer_path: str = 'Salesforce/codet5-small'
    model_temperature: float = 0.95
    model_samples: int = 5
    
    n_episodes_per_task: int = 1
    n_simulations: int = 10
    n_training_iterations: int = 100
    action_temperature: float = 0.95



# save.
class Agent(): 
    def __init__(self, trajectory_buffer, model, n_episodes, n_simulations, n_training_iterations, action_temperature):
        self.n_episodes = n_episodes
        self.n_simulations  = n_simulations
        self.n_training_iterations = n_training_iterations
        self.action_temperature = action_temperature

        self.trajectory_buffer = trajectory_buffer
        self.model = model
        
    def execute_episode(self, env, temperature): 
        
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
            action = root.select_action(temperature=temperature)
            state, reward, terminated = env.step(action=action, state=state)
            if terminated:
                ret = []
                solved = (reward == 1.0)
                for hist_state, hist_actions,  hist_action_probs in train_examples:
                    ret.append(( env.tokenized_task, hist_state, hist_actions, hist_action_probs, reward))
                
                return ret, solved


    def evaluate(self, env):
        episode_history, solved = self.execute_episode(env, 0)
        return int(solved)

    def learn(self, env): 
        task_solved = False
        for eps in range(self.n_episodes):
            episode_history, solved = self.execute_episode(env, self.action_temperature)
            self.trajectory_buffer.add_trajectory(episode_history)
            if solved:
                task_solved = True
                break 
        
        
        self.train() 
        return int(task_solved)
    
    
    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        self.model.train()
        trajectory_dataloader = DataLoader(self.trajectory_buffer, 
                                           batch_size=2, 
                                           collate_fn=self.trajectory_buffer._collate_fn)


        for sample in trajectory_dataloader:
            print(sample)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    task = Task.from_json('data/training/42a50994.json')
    pl.seed_everything(0)
    print(task.program)
    config = AlphaARCConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    trajectory_buffer =  TrajectoryBuffer()

    model = PolicyValueNetwork( config. model_path, config.tokenizer_path, config.model_temperature, num_samples=config.model_samples)
    model.to('cuda')
    agent = Agent(trajectory_buffer, model, config.n_episodes_per_task, config.n_simulations, config.n_training_iterations, config.action_temperature)
    env = LineLevelArcEnv(task, tokenizer)
    print(agent.learn(env))