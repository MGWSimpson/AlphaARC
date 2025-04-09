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
from dataclasses import dataclass

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# will port to hydra later but quick class to store hyper-parameters.
@dataclass
class AlphaARCConfig:
    pass








@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: Any) -> None:
    print("\n" + "=" * 10, "Configuration", "=" * 10)
    pl.seed_everything(config.seed)
    

    
    
    


   
        


if __name__ == "__main__":
    main()
