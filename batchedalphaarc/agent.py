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

from batchedalphaarc.logger import Logger
from collections import defaultdict
import lightning.pytorch as pl

from tqdm import tqdm


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
                    ):
        self.n_episodes = n_episodes
        self.n_simulations  = n_simulations
        self.action_temperature = action_temperature

        # self.trajectory_buffer = trajectory_buffer
        # self.replay_buffer = replay_buffer
        
        self.trajectory_q = trajectory_q
        self.replay_q = replay_q
        self.model = model
        # self.logger = logger
        # self.optimizer = optim.AdamW(self.model.parameters())
        self.learning_count = 0
        

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
    
    def learn(self, env): 
        task_solved = False
        self.encoder_output = self.model.encode(env.tokenized_task).squeeze()
        for eps in range(self.n_episodes):
            episode_history, solved, full_task_and_program = self.execute_episode(env, self.action_temperature)
            # self.trajectory_buffer.add_trajectory(episode_history)
            self.trajectory_q.put(episode_history)
            if solved:
                # self.replay_buffer.add_program_and_task(full_task_and_program[0], full_task_and_program[1])
                self.replay_q.put((full_task_and_program[0], full_task_and_program[1]))
                task_solved = True
                break 
        return int(task_solved)
    

    """def _train_rl(self, batch_logs): 
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

    """
       

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    task = Task.from_json('data/training/6ecd11f4.json')
    print(task.program)
    pl.seed_everything(0)
    logger = Logger()
    config = batchedalphaarcConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_config.tokenizer_path)
    trajectory_buffer =  TrajectoryBuffer(capacity=config.trajectory_buffer_capacity,
                                          n_actions=config.n_actions,
                                          max_action_len=config.max_action_len,
                                          max_state_len=config.max_state_len,
                                          max_task_len=config.max_task_len)
    
    replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity,
                                 max_state_len=config.max_state_len,
                                 max_task_len=config.max_task_len)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_config.tokenizer_path)
    
    model = PolicyValueNetwork(model_path=config.model_config.model_path,
                               tokenizer=tokenizer,
                               temperature=config.model_config.model_temperature,
                               num_samples=config.n_actions,
                               device= config.model_config.device)
    
    model = model.to('cuda')
    agent = Agent(trajectory_buffer=trajectory_buffer,
                  replay_buffer=replay_buffer, 
                  model=model,
                  n_episodes=config.n_episodes_per_task,
                  n_simulations=config.n_simulations,
                  action_temperature=config.action_temperature,
                  logger=logger)

    env = LineLevelArcEnv(task, 
                                tokenizer=tokenizer, 
                                max_task_len=config.max_task_len, 
                                max_state_len=config.max_state_len, 
                                n_actions=config.n_actions,
                                n_examples=config.n_examples)
    print(agent.learn(env))