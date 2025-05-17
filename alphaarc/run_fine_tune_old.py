import torch
import torch.optim as optim
import pytorch_lightning as pl

from torch.amp.grad_scaler import GradScaler
from torch import autocast


from tqdm import tqdm
from dataclasses import dataclass
from alphaarc.buffers import ReplayBuffer
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer, PretrainedConfig
from alphaarc.task import Task, from_dict
import json
import os

import datetime as dt
from pathlib import Path

import logging

logger = logging.getLogger(__name__)
timestamp_fmt = "%Y-%m-%d_%H-%M-%S"


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


@dataclass
class FineTuneConfig: 
    model_path: str = 'Salesforce/codet5p-220m'
    batch_size: int = 8
    device: str = 'cuda'
    lr: float = 5e-5
    n_epochs: int = 10
    fine_tune_dir: str = "./finetune/"
    n_examples =   10
    max_task_len= 256
    max_state_len = 256

"""
This function is where I would make the change
"""
def load_tasks(file_path)->list[Task]: 
    with open(file_path) as fp:
        json_object = json.load(fp)
        task_keys = json_object.keys()
        new_tasks = [from_dict(json_object[key]) for key in task_keys]

    return new_tasks



def setup_output_dir(config):
    base = Path(config.fine_tune_dir)
    log_dir = (
            base / dt.datetime.now().strftime(timestamp_fmt)
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename= log_dir / 'training.log', encoding='utf-8', level=logging.DEBUG)
    return log_dir




def run(config: FineTuneConfig):
    dataset = ReplayBuffer()
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tasks =     load_train_tasks(dirs=[ 'data/training'], files=['data/mutated_tasks_train_9600.json', 'data/mutated_tasks_train_19200.json'])
    dataset.preload_tasks(tasks, tokenizer, n_examples=config.n_examples, max_state_len=config.max_state_len, max_task_len=config.max_task_len)
    pl.seed_everything(0)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=ReplayBuffer.collate_fn)
    model = T5ForConditionalGeneration.from_pretrained(config.model_path)        
    model = model.to(config.device)    
    model = torch.compile(model)

    log_dir = setup_output_dir(config)
    

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in tqdm(range(config.n_epochs)):
        losses = []
        for input, target in tqdm( dataloader):
            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float32):
                loss = model(input_ids=input.to(config.device),labels=target.to(config.device)).loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.detach().item())
        
        logger.info(f"epoch {epoch}: avg loss: {sum(losses) / len(losses)}")
        model.save_pretrained(log_dir / "model", from_pt=True)




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
    

def load_train_tasks(dirs, files, split_keys_path= 'data/split_keys.json'):
    

    
    tasks = []
    for folder_path in dirs:
         tasks.extend(load_tasks_from_folders(folder_path))
    
    for file_path in files:
        tasks.extend(load_tasks_from_files(file_path))

    split_keys = load_key_split(split_keys_path)
    tasks = prune_tasks(tasks, split_keys['train'])
    return tasks






if __name__ == "__main__":
    config = FineTuneConfig()
    run(config)
