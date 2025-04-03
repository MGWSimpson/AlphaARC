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
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from alphaarc.env import ARCEnv
from alphaarc.agent import Agent
from alphaarc.replay_buffer import ReplayBuffer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"





@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: Any) -> None:
    print("\n" + "=" * 10, "Configuration", "=" * 10)
    pl.seed_everything(config.seed)
    
    n_attempts = 10
    buffer = ReplayBuffer()    
    agent = Agent()
    
    for task in tqdm():
        env = ARCEnv(task)
        for i in range(n_attempts): 
            
            observation = env.reset()

            done = False
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done = env.step(action)

                agent.store_transition(
                    


                )
                agent.learn()
                observation = observation_
    
    


   
        


if __name__ == "__main__":
    main()
