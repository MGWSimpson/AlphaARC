import os
import argparse
from alphaarc.configs import load_config, build_curriculum, build_env
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import numpy as np 
from alphaarc.policy.tokenize import tokenize_task
import torch 
from alphaarc.env import BaseEnv, ExceededTokenBudget
from alphaarc.buffers import ReplayBuffer
from alphaarc.utils import relabel_task
from alphaarc.utils import load_key_split
from alphaarc.train import BaseTrainer, SampleFilterTrainer
from alphaarc.env import LineLevelArcEnv
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import heapq, math, time
from typing import List
from dataclasses import dataclass, field
from alphaarc.task import Task
import numpy as np 
from alphaarc.policy.tokenize import tokenize_task
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn.functional as F
from alphaarc.dsl.primitives import PRIMITIVE_FUNCTIONS
from alphaarc.program_completer import ProgramCompleter, ProgramSampler
import re
import traceback


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


import itertools
from typing import List, Optional
from transformers import AutoTokenizer
from typing import List, Optional
from transformers import AutoTokenizer

from typing import List, Optional
from transformers import AutoTokenizer
from alphaarc.entropy_guided_search import entropy_fanout_search_encdec

from tqdm import tqdm


def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task


def run_experiment( model,
                    tok,
                    env, 
                    completer,
                    time_limit,
                    curriculum,):
    
    solved_task_ids = []
    full_curriculum = curriculum.generate_curriculum()

    full_curriculum = full_curriculum[20:]
    for i in tqdm(range(0, len (full_curriculum))):

        task = full_curriculum[i]
        env.set_task(task)
        input_ids = torch.tensor(encode_task(task, tok, model)). to('cuda')

        solved = entropy_fanout_search_encdec(  model,
                                                tok,
                                                input_ids,
                                                env,
                                                completer,
                                                task,
                                                time_limit,
                                                )
        
        if solved:
            solved_task_ids.append(task.task_key)
            print(f"solved: {len(set(solved_task_ids))}")
            print(set(solved_task_ids))



def main():
    # TODO: add in like config stuff.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='alphaarc/configs/egs_config.yaml')
    args = parser.parse_args()

    model = T5ForConditionalGeneration.from_pretrained('./finetune-checkpoint/dev-checkpoint')
    tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    model.to('cuda')
    config = load_config(args.config_path)

    env = LineLevelArcEnv('Salesforce/codet5p-220m',  10, 512, 512, 10, 50000)
    
    curriculum = build_curriculum(config['training_curriculum_config'])
    task_key_split = load_key_split('data/split_keys.json')
    curriculum.prune_tasks_not_in_list(tasks_to_keep=task_key_split['val'])

    sampler   = ProgramSampler(data_path="./data/")
    completer = ProgramCompleter(sampler)

    run_experiment( model, 
                   tok,
                   env, 
                   completer,
                   (60 * 2),
                   curriculum)


if __name__ == "__main__": 
    main()