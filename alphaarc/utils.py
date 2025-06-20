from alphaarc.task import Task

from importlib import import_module
from typing import Any, Dict, Iterable, List, Union
from matplotlib import pyplot as plt

import shutil

import os

import numpy as np
import json

import copy


# -- start new -- 



def save_answer(answer_dict):
    with open("data.jsonl", "w") as f:
        json.dump(answer_dict, f)


def prepare_output_dir(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  
    os.makedirs(output_dir)

def save_stats_to_file(stats, output_dir):
    stats_path = os.path.join(output_dir, "epoch_stats.jsonl")
    with open(stats_path, "w") as f:
        for entry in stats:
            json.dump(entry, f)
            f.write("\n")


def save_model(model, output_dir, epoch): 
    model.save_pretrained(os.path.join(output_dir, f"model_epoch_{epoch}"))


# -- end new -- 

def load_key_split(split_keys_path): 
    with open(split_keys_path) as fp:
        json_object = json.load(fp)
    return json_object


def transform_to_function(input_str: str, function_name: str) -> str:
    header = f"def {function_name}(I):\n"
    indented_content = "    " + input_str.strip().replace("\n", "\n    ")
    footer = "\n    return O"
    return header + indented_content + footer


def get_tokenizer(config):
    tokenizer_class = get_class(config.data.dataloader.tokenizer.cls)
    tokenizer = tokenizer_class.from_pretrained(
        config.data.dataloader.tokenizer.name, cache_dir=config.model.cache_dir
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = config.data.dataloader.tokenizer.pad_token_id
    if not tokenizer.eos_token_id:
        tokenizer.eos_token_id = config.data.dataloader.tokenizer.eos_token_id
    if (
        tokenizer.pad_token_id != config.data.dataloader.tokenizer.pad_token_id
        or tokenizer.eos_token_id != config.data.dataloader.tokenizer.eos_token_id
    ):
        raise Exception("Mismatch in tokenizer")
    return tokenizer


def get_class(class_str: str) -> Any:
    """
    A little util to import a class from any package, withot the need for an explicit import
    statement in the header. Useful when your classes can come from many different modules (like HF models).
    Can actually work also with modules which are not classes, like functions.
    :param class_str: The name of the class, inclusive of namespace from the root of the
     module is comes from (e.g. torch.nn.Linear, NOT just Linear
    :return: The class itself.
    """
    package = import_module(".".join(class_str.split(".")[:-1]))
    return getattr(package, class_str.split(".")[-1])


def get_grid_size(grid):
    num_rows = len(grid)
    num_columns = len(grid[0]) if num_rows > 0 else 0
    return (num_rows, num_columns)


def get_num_pixels(grid):
    num_rows = len(grid)
    num_columns = len(grid[0]) if num_rows > 0 else 0
    return num_rows if num_columns == 0 else num_rows * num_columns


def pad_and_convert(task, state, actions, pad_value=0.0, max_state_size=1024, max_action_size=20):
    
    padded_task = np.pad(task, pad_width=(0, max_state_size- task.shape[-1]), mode='constant', constant_values=pad_value)
    padded_state = np.pad(state, pad_width=(0, max_state_size - state.shape[-1]), mode='constant', constant_values=pad_value)
    padded_actions = []
    for action in actions:
        pad_len = max_action_size - action.shape[-1]
        padded_action = np.pad(action, pad_width=((0, pad_len)), mode='constant', constant_values=pad_value)
        padded_actions.append(padded_action)
    
    padded_actions = np.stack(padded_actions, axis=0)
    return padded_task, padded_state, padded_actions


# somehow need to get 
def relabel_task(task, env, program, program_string): 
    env.set_task(task)
    outputs = env.get_outputs(program)


    new_training_examples = []
    new_test_examples = []

    output_counter = 0
    for i, data in enumerate(task.training_examples):
        x = copy.deepcopy(data)
        x["output"] = outputs[output_counter]
        new_training_examples.append(x)

        output_counter +=1
    
    for i, data in enumerate(task.test_examples):
        x = copy.deepcopy(data)

        x["output"] = outputs[output_counter]
        new_test_examples.append(x)
        output_counter +=1

    ret_task = Task(program_string, new_training_examples, new_test_examples, task.task_key, task.task_key, extra_info=task.extra_info)
    
    return ret_task
