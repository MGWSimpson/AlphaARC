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
import time

import logging

logger = logging.getLogger(__name__)
timestamp_fmt = "%Y-%m-%d_%H-%M-%S"


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from alphaarc.task import Task
from alphaarc.policy.tokenize import tokenize_task

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    AutoTokenizer
)

from transformers import DataCollatorForSeq2Seq
import wandb


@dataclass
class FineTuneConfig: 
    model_path: str = 'Salesforce/codet5p-220m'
    device: str = 'cuda'
    train_batch_size: int = 8
    eval_batch_size: int = 2 
    lr: float =1e-5
    output_dir: str = './finetune/'
    num_epochs: int = 15




def fine_tune(  model, 
                tokenizer,
                num_epochs,
                train_batch_size,
                eval_batch_size,
                train_ds,
                dev_ds,
                eval_ds,
                lr,
                output_dir,
                ): 
    
    wandb.init(
    project="my-finetune-project",
    config={
        "epochs": num_epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": lr,
    })

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=lr,
        lr_scheduler_type='constant',
        logging_steps=3,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=10,
        gradient_accumulation_steps=64,
        bf16=True, 
        report_to=["wandb"],  
    )

   

    class MultiEvalTrainer(Trainer):
        def __init__(self, *args, eval_extra_dataset=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_extra_dataset = eval_extra_dataset
        
        def _safe_save_metrics(self, split_name, metrics):
            output_dir = os.path.join(self.args.output_dir, split_name)
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "results.json")
            with open(path, "w") as f:
                json.dump(metrics, f, indent=4)

        
        def evaluate(self, eval_dataset=None, **kwargs):
            # Determine logging step

            # Regular dev set evaluation
            dev_metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)
            self.log_metrics("eval/dev", dev_metrics)
            self._safe_save_metrics("eval/dev", dev_metrics)

            # Always log dev metrics
            wandb.log({f"dev_{k}": v for k, v in dev_metrics.items()})

            # Extra test set evaluation
            if self.eval_extra_dataset is not None:
                test_metrics = super().evaluate(eval_dataset=self.eval_extra_dataset, **kwargs)
                self.log_metrics("eval/test", test_metrics)
                self._safe_save_metrics("eval/test", test_metrics)

                wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

            return dev_metrics

    trainer = MultiEvalTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        eval_extra_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)




def convert_task_to_jsonl(task, tokenizer , max_encoder_length = 512, max_decoder_length=512, n_examples= 10): 
    json_object = {}
    x = tokenize_task(task, tokenizer, n_examples, int(max_encoder_length/2),  int(max_encoder_length/2) ) # divide by 2 so that its between input and output, -1 as it adds a token
    json_object['input_ids'] = x['input_ids']
    json_object['attention_mask'] = x['attention_mask']
    json_object['labels'] = tokenizer(task.program_lines)['input_ids'][:max_decoder_length]
    return json_object




def construct_ds(tasks, tokenizer):
    json_objects = []
    for task in tasks:
        json_objects.append(convert_task_to_jsonl(task, tokenizer ))
    return Dataset.from_list(json_objects)


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





def setup_output_dir(config): 
    log_dir = Path(config.output_dir) / dt.datetime.now().strftime(timestamp_fmt)
    log_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(log_dir)

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



def split_dev_tasks(train_tasks, dev_set_keys): 

    new_train_tasks = []
    dev_tasks = []

    for t in train_tasks:
        if t.parent_key is None: # check the main key
            if t.task_key in dev_set_keys:
                dev_tasks.append(t)
            else:
                new_train_tasks.append(t)
        else:
            if t.parent_key not in dev_set_keys:
                new_train_tasks.append(t)


    return new_train_tasks, dev_tasks


# handles all the orchestrating.
def main(config): 

    setup_output_dir(config)

    tasks = load_train_tasks(dirs=[ 'data/training'], files=['data/mutated_tasks_train_9600.json', 'data/mutated_tasks_train_19200.json'])
    
    # computed from task splitter
    dev_set_keys = ['ddf7fa4f', '0962bcdd', '444801d8', 'c1d99e64', 'b1948b0a', 'e26a3af2', '8e1813be', 'd9f24cd1', 'a2fd1cf0', 'ce22a75a', '4290ef0e']
                    
    train_tasks, eval_tasks = split_tasks_based_on_key(tasks)

    train_tasks, dev_tasks = split_dev_tasks(train_tasks, dev_set_keys)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = T5ForConditionalGeneration.from_pretrained(config.model_path)        
    model.to(config.device)


    train_ds, eval_ds, dev_ds = construct_ds(train_tasks, tokenizer), construct_ds(eval_tasks, tokenizer), construct_ds(dev_tasks, tokenizer)

    fine_tune(  model, 
                tokenizer, 
                config.num_epochs,
                config.train_batch_size,
                config.eval_batch_size,
                train_ds,
                dev_ds,
                eval_ds, 
                config.lr,
                config.output_dir)


              
    




if __name__ == "__main__":
    config = FineTuneConfig()
    main(config)