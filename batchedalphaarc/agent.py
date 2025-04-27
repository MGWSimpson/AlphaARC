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



 
def compute_n_of_syntax_correct_actions(state, actions, env):
    
    n_correct_actions = 0
    for action in actions:
        is_valid_syntax = env.is_valid_syntax( action, state)
        n_correct_actions += float(is_valid_syntax)

    return n_correct_actions



class Agent(): 
    def __init__(self, trajectory_q,
                 replay_q,
                    model, 
                    n_episodes, n_simulations, action_temperature, #logger
                    evaluation_action_temperature, is_debugging=True
                    ):
        self.n_episodes = n_episodes
        self.n_simulations  = n_simulations
        self.action_temperature = action_temperature

        self.trajectory_q = trajectory_q
        self.replay_q = replay_q
        self.model = model
        self.learning_count = 0
        self.evaluation_action_temperature = evaluation_action_temperature
        self.is_debugging = is_debugging

    def execute_episode(self, env, temperature, episode_log): 
        
        state = env.reset()
        train_examples = []
        terminated = False
        total_actions, n_correct_syntax = 0,0.0
        while not terminated:
            self.mcts = MCTS(env , encoder_output=self.encoder_output,  n_simulations=self.n_simulations)
            root = self.mcts.run(self.model, state)
            actions = root.child_actions


            action_probs = [v.visit_count for v in root.children]
            action_probs = action_probs / np.sum(action_probs)

            n_correct_syntax += compute_n_of_syntax_correct_actions(state, actions, env )
            total_actions += len(action_probs)
          
            train_examples.append((state, actions, action_probs))
            action = root.select_action(temperature=temperature)
            state, reward, terminated = env.step(action=action, state=state)
            
            if terminated:
                ret = []
                solved = (reward == 1.0)
                full_task_and_program = (env.tokenized_task, state)
                for hist_state, hist_actions,  hist_action_probs in train_examples:
                    ret.append(( env.tokenized_task, hist_state, hist_actions, hist_action_probs, reward))

                episode_log['correct_syntax_ratio'] = n_correct_syntax / total_actions
                return ret, solved, full_task_and_program


 
    
    def evaluate(self, env):
        task_solved = False
        episode_log = make_episode_log(env.task.task_key)
        self.encoder_output = self.model.encode(env.tokenized_task).squeeze()
        for eps in range(self.n_episodes): 
            episode_history, solved, full_task_and_program = self.execute_episode(env, self.evaluation_action_temperature, episode_log) # set temp to zero.

            if solved:
                task_solved = True
                break
        

        episode_log['solved'] = float(task_solved)
        return episode_log


    def learn(self, env): 
        task_solved = False
        self.encoder_output = self.model.encode(env.tokenized_task).squeeze()
        episode_log = make_episode_log(env.task.task_key)

        # TODO: may need to make changes to log to account for fact there can be multipole episodes.
        for eps in range(self.n_episodes):
            episode_history, solved, full_task_and_program = self.execute_episode(env, self.action_temperature, episode_log=episode_log)
            # self.trajectory_buffer.add_trajectory(episode_history)
            self.trajectory_q.put(episode_history)
            if solved:
                # self.replay_buffer.add_program_and_task(full_task_and_program[0], full_task_and_program[1])
                self.replay_q.put((full_task_and_program[0], full_task_and_program[1]))
                task_solved = True
                break 
        
        episode_log['solved'] = float(task_solved)
        return episode_log

    

       
 