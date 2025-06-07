"""
File contains all the code for running the tree search based experiments.
"""

import copy
import time
import math
import copy
import numpy as np
import torch
import os 
from alphaarc.env import LineLevelArcEnv
from transformers import AutoTokenizer, T5ForConditionalGeneration
from alphaarc.program_completer import ProgramCompleter, ProgramSampler
from alphaarc.policy.tokenize import tokenize_task
from torch.nn.utils.rnn import pad_sequence
from alphaarc.task import Task
import torch.nn.functional as F
import traceback
import random
import torch.nn as nn

import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# -- helpers --

def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task


class BaseMethod:
    def __init__(self, uses_model):
        self.uses_model = uses_model

# generates all actions + no priors 
class MCTSMethod(BaseMethod): 
    pass

# generates all actions + priors
class TGMCTSMethod(BaseMethod): 
    pass

# generates actions based on entropy + priors 
class SplintMCTSMethod(BaseMethod):
    pass


class Node: 
    def __init__(self, prior= 0):
        self.prior = prior

    def expand(self, state, actions, action_probs):
        state = state.to('cpu')
        actions = [x.to('cpu') for x in actions]
        self.state = state.clone()
        self.child_actions = copy.deepcopy(actions)
        self.children = [Node(prior=prob) for prob in action_probs]
    

def rollout( state,
            actions,
            enc_out,
            model: BasMethod,
            env: LineLevelArcEnv): 


    # need to make a random choice of actions    
    action = random.choice(actions)
    action= torch.cat((state, action))
    program = model.rollout(enc_out, action.unsqueeze(0).to('cuda'))
    reward, terminated = env.evaluate_program(program.squeeze(), should_token_account=False)
    return reward

def backpropagate(path, value):
    for node in reversed(path):    
        node.visit_count += 1
        node.value_sum  += value

def run_search(env: LineLevelArcEnv,
               task, 
               prompt_ids,
               model, 
               time_limit=60):
    

    if model.uses_model:
        model.eval()
        with torch.no_grad():
            enc_out = model.encode(prompt_ids)
    else:
        enc_out = None
    

    start_time = time.time()
    root = Node(0)

    init_state = torch.tensor([0,1], device='cuda')
    actions, action_probs = model.predict(enc_out, init_state, task, prompt_ids) # perform the predictions, but with 
    root.expand(init_state, actions, action_probs)

    while (time.time() - start_time) < time_limit:
        node = root
        search_path = [node]
        # SELECT
        while node.expanded():
            action, node = node.select_child()
            search_path.append(node)

        parent = search_path[-2]
        state = parent.state

        # expansion 
        next_state, value, terminated = env.step(action=action, state=state, should_do_token_accounting=False)
            
        # for some reason this is a numpy array
        next_state = torch.tensor(next_state)

        if not terminated: 
            actions, action_probs = model.predict(enc_out, next_state.to('cuda'), task, prompt_ids)
                
            value = rollout(state, actions, enc_out, model, env) # rollout
            node.expand(next_state, actions, action_probs)

            # backprop
        backpropagate(search_path, value)
    return root

"""
set up small rig for running the experiment
"""
def run_experiment( method: BaseMethod,
                    tasks: list,
                    time_limit: int,
                    tok: AutoTokenizer,
                    ):
    
    # TODO: wrap this in loop around all tasks.
    task = Task.from_json('./data/training/c8f0f002.json')
    input_ids = torch.tensor(encode_task(task, tok, None)).to('cuda')
    env = LineLevelArcEnv('Salesforce/codet5p-220m',  10, 512, 512, 10, 50000)
    env.set_task(task)
    run_search(env, task, input_ids,  method, time_limit)



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='alphaarc/configs/base_config.yaml')


    args = parser.parse_args()

    # TODO: add proper config files
    model = T5ForConditionalGeneration.from_pretrained('./finetune/2025-05-27_17-42-37/checkpoint-1650')
    model.to('cuda')
    tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    sampler   = ProgramSampler(data_path="./data/")
    completer = ProgramCompleter(sampler)
    
    
    if args.method == "MCTS": 
        pass
    elif args.method == "TGMCTS":
        pass
    elif args.method == "SPLINTMCTS":
        pass
    else:
        raise ValueError("Method does not exist!")

    run_experiment(method=None,
                   tasks=None,
                   time_limit=(60 * 3),
                   tok=tok)


if __name__ == "__main__": 
    main()