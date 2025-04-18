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

@dataclass
class FineTuneConfig: 
    model_path: str = 'Salesforce/codet5p-220m'
    batch_size: int = 8
    device: str = 'cuda'
    lr: float = 5e-5
    n_epochs: int = 10
    fine_tune_dir: str = "./finetune/"
    

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
    tasks = load_tasks('data/mutated_tasks_train_9600.json')
    dataset.preload_tasks(tasks, tokenizer)
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
        for input, target in dataloader:
            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                loss = model(input_ids=input.to(config.device),labels=target.to(config.device)).loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        
        logger.info(f"epoch {epoch}: avg loss: {sum(loss) / len(loss)}")
        model.save_pretrained(log_dir / "model", from_pt=True)



if __name__ == "__main__": 
    config = FineTuneConfig()
    run(config=config)