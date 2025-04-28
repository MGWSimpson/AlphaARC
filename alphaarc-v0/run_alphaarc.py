# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import cProfile
import csv
import gc
import json
import os
import time
from typing import Any

from tqdm import tqdm
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from dataclasses import dataclass
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from transformers import T5ForConditionalGeneration, AutoTokenizer



from alphaarc.env import LineLevelArcEnv
from alphaarc.curriculum import Curriculum
from alphaarc.agent import Agent
from alphaarc.buffers import ReplayBuffer, TrajectoryBuffer
from alphaarc.networks import PolicyValueNetwork
from alphaarc.logger import Logger


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


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
class AlphaARCConfig:
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


def evaluate(agent, evaluation_set, tokenizer, config):
    solved_tasks = 0
    for task in tqdm(evaluation_set.tasks):
        env =  LineLevelArcEnv(task, 
                              tokenizer=tokenizer, 
                              max_task_len=config.max_task_len, 
                              max_state_len=config.max_state_len, 
                              n_actions=config.n_actions,
                              n_examples=config.n_examples)
        solved_tasks += agent.evaluate(env)
        print(f"solve rate on the evaluation set: {solved_tasks} / {len(evaluation_set.tasks)} ")

def main() -> None:
    print("\n" + "=" * 10, "Configuration", "=" * 10)
    config = AlphaARCConfig()
    logger = Logger()

    pl.seed_everything(config.seed)
    
    curriculum = Curriculum(dir_paths=['data/training'],file_paths=['data/mutated_tasks_train_19200.json'])
    evaluation = Curriculum(dir_paths=['data/evaluation'])
    
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
    
    model.to(config.model_config.device)
    
    agent = Agent(trajectory_buffer=trajectory_buffer,
                  replay_buffer=replay_buffer, 
                  model=model,
                  n_episodes=config.n_episodes_per_task,
                  n_simulations=config.n_simulations,
                  action_temperature=config.action_temperature,
                  logger=logger)

    evaluate(agent, evaluation_set=evaluation, tokenizer=tokenizer, config=config)

    terminated = False # TODO: decide on termination condition
    while not terminated:
        for task_iteration, task in tqdm(enumerate(curriculum.generate_curriculum()), total=len(curriculum)):
            print(f"starting on task {task.task_key}")
            env = LineLevelArcEnv(task, 
                                tokenizer=tokenizer, 
                                max_task_len=config.max_task_len, 
                                max_state_len=config.max_state_len, 
                                n_actions=config.n_actions,
                                n_examples=config.n_examples)
            
            tasks_solved = agent.learn(env)
            
            if tasks_solved:
                curriculum.handle_solved_tasks(task)
                print(f"number of tasks solved: {curriculum.get_n_solved()} / {curriculum.get_total_n()}")

        if task_iteration % config.train_every:
            agent.train()
            print("starting eval!")
            

if __name__ == "__main__":
    main()
