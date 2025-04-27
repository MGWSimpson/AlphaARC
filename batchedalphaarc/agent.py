import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer

from batchedalphaarc.policy.environment import execute_candidate_program
from batchedalphaarc.task import Task
from batchedalphaarc.env import LineLevelArcEnv
from batchedalphaarc.mcts import MCTS
import os
import torch.optim as optim
import torch
from batchedalphaarc.env import LineLevelArcEnv
from batchedalphaarc.curriculum import Curriculum
from batchedalphaarc.buffers import ReplayBuffer, TrajectoryBuffer
import torch.nn.functional as F
from tqdm import tqdm
import time
from dataclasses import dataclass
from torch.utils.data import DataLoader 
from torch.amp.grad_scaler import GradScaler
from torch import autocast
import lightning.pytorch as pl
from tqdm import tqdm

from batchedalphaarc.logger import make_episode_log



@dataclass
class RLTrainingConfig:
    rl_batch_size: int =2

@dataclass
class SupervisedTrainingConfig:
    supervised_batch_size: int = 2

@dataclass
class ModelConfig:
    model_path: str = 'finetune/2025-04-18_12-38-42/model'
    tokenizer_path: str = 'Salesforce/codet5p-220m'
    model_temperature: float = 0.95
    device: str = 'cuda'

@dataclass
class batchedalphaarcConfig:
    rl_training_config: RLTrainingConfig = RLTrainingConfig()
    supervised_training_config: SupervisedTrainingConfig = SupervisedTrainingConfig()
    model_config: ModelConfig = ModelConfig()
    n_actions: int = 5
    n_examples: int = 10
    n_episodes_per_task: int = 1
    n_simulations: int = 10
    action_temperature: float = 1
    seed: int = 0
    max_state_len: int = 1024
    max_task_len: int = 512
    max_action_len: int = 20
    trajectory_buffer_capacity = 100_000
    replay_buffer_capacity: int = 100_000
    train_every: int = 100




# TODO: decide where to move all the train stuff.
class Agent(): 
    def __init__(self, trajectory_q,
                 replay_q,
                 #trajectory_buffer, 
                        #replay_buffer, 
                    model, 
                    n_episodes, n_simulations, action_temperature, #logger
                    evaluation_action_temperature,
                    ):
        self.n_episodes = n_episodes
        self.n_simulations  = n_simulations
        self.action_temperature = action_temperature

        
        self.trajectory_q = trajectory_q
        self.replay_q = replay_q
        self.model = model
        self.learning_count = 0
        self.evaluation_action_temperature = evaluation_action_temperature


    def execute_episode(self, env, temperature): 
        
        state = env.reset()
        train_examples = []
        terminated = False
        
        while not terminated:
            self.mcts = MCTS(env , encoder_output=self.encoder_output,  n_simulations=self.n_simulations)
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
                full_task_and_program = (env.tokenized_task, state)
                for hist_state, hist_actions,  hist_action_probs in train_examples:
                    ret.append(( env.tokenized_task, hist_state, hist_actions, hist_action_probs, reward))

                return ret, solved, full_task_and_program


    """def evaluate(self, env): 
        task_solved = False
        self.model.set_task(env.tokenized_task)
        
        for eps in range(self.n_episodes):
            episode_history, solved, full_task_and_program = self.execute_episode(env, 0) # set temp to zero.
            if solved:
                task_solved = True
                break 
        
        return int(task_solved)"""
    
    def evaluate(self, env):
        task_solved = False
        episode_log = make_episode_log(env.task.task_key)
        self.encoder_output = self.model.encode(env.tokenized_task).squeeze()
        for eps in range(self.n_episodes): 
            episode_history, solved, full_task_and_program = self.execute_episode(env, self.evaluation_action_temperature) # set temp to zero.

            if solved:
                task_solved = True
                break
        

        episode_log['solved'] = float(task_solved)
        return episode_log


    def learn(self, env): 
        task_solved = False
        self.encoder_output = self.model.encode(env.tokenized_task).squeeze()
        episode_log = make_episode_log(env.task.task_key)

        for eps in range(self.n_episodes):
            episode_history, solved, full_task_and_program = self.execute_episode(env, self.action_temperature)
            # self.trajectory_buffer.add_trajectory(episode_history)
            self.trajectory_q.put(episode_history)
            if solved:
                # self.replay_buffer.add_program_and_task(full_task_and_program[0], full_task_and_program[1])
                self.replay_q.put((full_task_and_program[0], full_task_and_program[1]))
                task_solved = True
                break 
        
        episode_log['solved'] = float(task_solved)
        return episode_log

    

       
 