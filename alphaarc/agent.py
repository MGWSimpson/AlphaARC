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
from torch.amp.grad_scaler import GradScaler
from torch import autocast

from alphaarc.logger import Logger
from collections import defaultdict
import lightning.pytorch as pl

from tqdm import tqdm

@dataclass
class AlphaARCConfig:
    batch_size: int = 2 
    model_path: str = 'alphaarc/pretrained/last.ckpt.dir'
    tokenizer_path: str = 'Salesforce/codet5-small'
    model_temperature: float = 0.95
    model_samples: int = 5
    
    n_episodes_per_task: int = 3
    n_simulations: int = 20
    n_training_iterations: int = 3
    action_temperature: float = 0.95



# TODO: decide where to move all the train stuff.
class Agent(): 
    def __init__(self, trajectory_buffer, replay_buffer, model, n_episodes, n_simulations, action_temperature, logger):
        self.n_episodes = n_episodes
        self.n_simulations  = n_simulations
        self.action_temperature = action_temperature

        self.trajectory_buffer = trajectory_buffer
        self.replay_buffer = replay_buffer
        self.model = model
        self.logger = logger
        self.optimizer = optim.AdamW(self.model.parameters())
        self.learning_count = 0
        

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
                full_task_and_program = (env.tokenized_task, state)
                for hist_state, hist_actions,  hist_action_probs in train_examples:
                    ret.append(( env.tokenized_task, hist_state, hist_actions, hist_action_probs, reward))

                return ret, solved, full_task_and_program


    def evaluate(self, env): 
        task_solved = False
        self.model.set_task(env.tokenized_task)
        
        for eps in range(self.n_episodes):
            episode_history, solved, full_task_and_program = self.execute_episode(env, 0) # set temp to zero.
            if solved:
                task_solved = True
                break 
        
        return int(task_solved)
    
    def learn(self, env): 
        task_solved = False

        self.model.set_task(env.tokenized_task)
        
        for eps in range(self.n_episodes):
            episode_history, solved, full_task_and_program = self.execute_episode(env, self.action_temperature)
            self.trajectory_buffer.add_trajectory(episode_history)

            if solved:
                self.replay_buffer.add_program_and_task(full_task_and_program[0], full_task_and_program[1])
                task_solved = True
                break 
        return int(task_solved)
    

    def _train_rl(self, batch_logs): 
        trajectory_dataloader = DataLoader(self.trajectory_buffer, 
                                           batch_size=1,
                                           collate_fn=TrajectoryBuffer.collate_fn)
        
        scaler = GradScaler()

        for batch in tqdm(trajectory_dataloader, desc="rl training"):
            task, state, actions, target_pis, target_vs = batch
            self.optimizer.zero_grad()
            

            with autocast(device_type='cuda', dtype=torch.float16):
                target_pis = target_pis.to(self.model.device)
                target_vs = target_vs.to(self.model.device)
                predicted_pis, predicted_vs = self.model.forward( task.to(self.model.device), state.to(self.model.device), actions.to(self.model.device))

                policy_loss = F.cross_entropy(predicted_pis, target_pis)
                value_loss = F.mse_loss( predicted_vs, target_vs)
                
                loss = policy_loss + value_loss 
                
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()


            batch_logs["policy"].append(policy_loss.detach().item())
            batch_logs["value"].append(value_loss.detach().item())
            batch_logs["trajectory_total"].append(loss.detach().item())



    def _train_supervised(self, batch_logs):
        scaler = GradScaler()
        replay_dataloader = DataLoader(self.replay_buffer, batch_size=1, collate_fn=ReplayBuffer.collate_fn)

        print(f"starting supervised training")
        for batch in tqdm(replay_dataloader, desc="supervised training"):
            task, state = batch
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = self.model.model(   input_ids=task.to(self.model.device),
                                    labels=state.to(self.model.device)).loss
                
            scaler.scale(loss).backward()
            scaler.step(self. optimizer)
            scaler.update()
            batch_logs["replay"].append(loss.detach().item())


        # TODO: add a check in here!
        batch_logs['replay'].append(0)
        




    def train(self):
        batch_logs = defaultdict(list)
        self.model.train()

        self._train_rl(batch_logs)
        self._train_supervised(batch_logs)

        epoch_means = {k: sum(v)/len(v) for k, v in batch_logs.items()}
        self.logger.log_training_data(epoch_means["policy"], 
                                      epoch_means["value"], 
                                      epoch_means["replay"],
                                      self.learning_count,
                                      len(self.trajectory_buffer),
                                      len(self.replay_buffer))        
        self.learning_count +=1


       

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    task = Task.from_json('data/training/42a50994.json')
    pl.seed_everything(0)
    logger = Logger()
    config = AlphaARCConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    trajectory_buffer =  TrajectoryBuffer()
    replay_buffer = ReplayBuffer()
    replay_buffer.preload_tasks([task], tokenizer)
    model = PolicyValueNetwork( config. model_path, config.tokenizer_path, config.model_temperature, num_samples=config.model_samples)
    model.to('cuda')
    agent = Agent(trajectory_buffer, replay_buffer, model, config.n_episodes_per_task, config.n_simulations, config.n_training_iterations, config.action_temperature, logger)
    env = LineLevelArcEnv(task, tokenizer)
    print(agent.learn(env))