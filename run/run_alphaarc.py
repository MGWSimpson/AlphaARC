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
from alphaarc.buffers import ReplayBuffer
from alphaarc.networks import PolicyValueNetwork

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# will port to hydra later but quick class to store hyper-parameters.
@dataclass
class AlphaARCConfig:
    batch_size: int = 2 
    model_path: str = 'alphaarc/pretrained/last.ckpt.dir'
    tokenizer_path: str = 'Salesforce/codet5-small'
    model_temperature: float = 0.95
    model_samples: int = 5
    
    n_episodes_per_task: int = 10
    n_simulations: int = 10
    n_training_iterations: int = 100
    action_temperature: float = 1





@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: Any) -> None:
    print("\n" + "=" * 10, "Configuration", "=" * 10)
    pl.seed_everything(config.seed)
    curriculum = Curriculum(dir_paths=['data/training', 'data/evaluation'], 
                   file_paths=['data/mutated_tasks_train_9600.json'])

    terminated = False # TODO: decide on termination condition
    
    config = AlphaARCConfig()

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    replay_buffer =  ReplayBuffer(config.batch_size)
    model = PolicyValueNetwork( config. model_path, config.tokenizer_path, config.model_temperature, num_samples=config.model_samples)
    model.to('cuda')
    agent = Agent(replay_buffer, model, config.n_episodes_per_task, config.n_simulations, config.n_training_iterations, config.action_temperature)


    while not terminated:
        task = curriculum.select_task()
        env = LineLevelArcEnv(task, tokenizer=tokenizer)
        agent.learn(env)
        

    

    
    
    


   
        


if __name__ == "__main__":
    main()
