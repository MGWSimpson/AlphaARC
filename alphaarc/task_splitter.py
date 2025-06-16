from datasets import Dataset

import torch
import torch.optim as optim
import pytorch_lightning as pl

from torch.amp.grad_scaler import GradScaler
from torch import autocast
import random

from tqdm import tqdm
from dataclasses import dataclass
from alphaarc.buffers import ReplayBuffer
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer, PretrainedConfig
from alphaarc.task import Task, from_dict
import json
import os
import wandb
import datetime as dt
from pathlib import Path

import logging

import random
from collections import defaultdict
from typing import List, Tuple, Dict
import os
import json

def split_tasks_based_on_key( tasks, split_keys_path= 'data/split_keys.json',): 
    split_keys = load_key_split(split_keys_path)


    train_list = []
    eval_list = []

    for task in tasks:
        if task.parent_key is None: # check the main key
            if task.task_key in split_keys['train']:
                train_list.append(task)
            else:
                eval_list.append(task)
        else:
            if task.parent_key in split_keys['train']:
                train_list.append(task)
         

    return train_list, eval_list

def load_tasks_from_folders(dir_path):
        tasks = []
        file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        new_tasks = [Task.from_json(path, False) for path in file_paths]
        tasks.extend(new_tasks)

        return tasks

def load_tasks_from_files(file_path):
    tasks = [ ]
    with open(file_path) as fp:
        json_object = json.load(fp)
    task_keys = json_object.keys()
    new_tasks = [from_dict(json_object[key], False) for key in task_keys]
    tasks.extend(new_tasks)

    return tasks


def load_key_split(split_keys_path): 

    with open(split_keys_path) as fp:
        json_object = json.load(fp)

    return json_object

def prune_tasks(tasks, train_set_list ): 
    
    prune_list = []
    for task in tasks:
        if task.parent_key is None: # check the main key
            if task.task_key not in train_set_list:
                prune_list.append(task)
        else:
            if task.parent_key not in train_set_list:
                prune_list.append(task)
    


    for task in prune_list:
        tasks.remove(task)
    
    print(f"removed {len(prune_list)} tasks")
    return tasks
    

def load_train_tasks(dirs, files, split_keys_path= 'data/split_keys.json', dev_mode=True):
    

    
    tasks = []
    for folder_path in dirs:
         tasks.extend(load_tasks_from_folders(folder_path))
    
    for file_path in files:
        tasks.extend(load_tasks_from_files(file_path))


    if dev_mode:
        split_keys = load_key_split(split_keys_path)
        tasks = prune_tasks(tasks, split_keys['train'])

    random.shuffle(tasks)
    return tasks



def split_tasks(tasks): 
    

    eval_tasks = {}

    l = []
    for task in tasks:
        # not core task and no eval task yet 
        if task.parent_key is None and task.parent_key not in eval_tasks:
            eval_tasks[task.parent_key] = task


    eval_list = list(eval_tasks.values())
    


    for task in eval_list:
        tasks.remove(task)

    return tasks, eval_list




def infer_buckets(train: List[Task],
                  dev:   List[Task]) -> Dict[int, int]:
 
    train_count = defaultdict(int)
    dev_count   = defaultdict(int)

    for _, L in train:
        train_count[L] += 1
    for _, L in dev:
        dev_count[L]   += 1

    bucket_dict = {L: train_count[L] for L in train_count}
    return bucket_dict


def stratified_sample_k(train: List[Task],
                        bucket_dict: Dict[int, int],
                        k: int,
                        rng_seed: int = 42
                       ) -> Tuple[List[Task], List[Task]]:
  
    rng = random.Random(rng_seed)

    # group task indices by length
    idx_by_len = defaultdict(list)
    for idx, (_, L) in enumerate(train):
        idx_by_len[L].append(idx)

    total_train = len(train)
    chosen_idx  = set()
    remaining_k = k

    for L, idx_list in sorted(idx_by_len.items()):
        prop = len(idx_list) / total_train
        alloc = min(int(round(prop * k)), len(idx_list))
        if alloc > 0:
            picks = rng.sample(idx_list, alloc)
            chosen_idx.update(picks)
            remaining_k -= alloc

    if remaining_k > 0:
        available = [i for i in range(total_train) if i not in chosen_idx]
        chosen_idx.update(rng.sample(available, remaining_k))

    # Build splits
    val_split  = [train[i] for i in chosen_idx]
    new_train  = [task     for i, task in enumerate(train) if i not in chosen_idx]

    return val_split, new_train


# ---------------- example usage ----------------
if __name__ == "__main__":
    # Example data
    tasks = load_train_tasks(dirs=[ 'data/training'], files=[], dev_mode=False)
    train_tasks, eval_tasks = split_tasks_based_on_key(tasks)

    train_tasks = [(t.task_key, len(t.program_lines)) for t in train_tasks]
    eval_tasks = [(t.task_key, len(t.program_lines)) for t in eval_tasks]



    bucket_dict = infer_buckets(train_tasks, eval_tasks)
    val_split, new_train = stratified_sample_k(train_tasks, bucket_dict, k=11)

    print("Held-out validation tasks (10):", [tid for tid, _ in val_split])
    print("Remaining training set size  :", len(new_train))
