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

def puct_score(parent, child):
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0

    return value_score + prior_score

def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task


def format_as_dummy_program(program_lines):
    return f"""def solve_28bf18c6(I):
    {program_lines}"""


def return_empty_nodes(): 
    return [], []

def merge_with_overlap(s1, s2):
    max_overlap = 0
    overlap_start = 0

    for i in range(1, min(len(s1), len(s2)) + 1):
        if s1[-i:] == s2[:i]:
            max_overlap = i

    return s1 + s2[max_overlap:]

def compute_log_probs_batched(model, input_batch, ids_batch):
    
    with torch.no_grad():
        labels = ids_batch.clone()

        labels[labels == 0] = -100
        mask = (labels != -100)
        labels[:, 0] = 0

        logits = model(input_ids=input_batch.repeat(ids_batch.shape[0], 1), labels=labels).logits  # (B, L, V)
        log_probs = torch.log_softmax(logits, dim=-1)

        token_logp = log_probs.gather(dim=-1, index=ids_batch.unsqueeze(-1)).squeeze(-1)  # (B, L)
        token_logp = token_logp * mask
        return token_logp.sum(dim=-1)  # (B,)

def entropy_bits(logits):
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    return (-(p * logp).sum(dim=-1) / math.log(2))




# -- end helpers --

class BaseMethod:
    def __init__(self, uses_model):
        self.uses_model = uses_model


    def rollout(self, enc_out, action, task): 
        raise NotImplementedError
    
    def predict(enc_out, init_state, task, prompt_ids): 
        raise NotImplementedError 
    
    
# generates all actions + no priors 
class MCTSMethod(BaseMethod): 
    
    def __init__(self, uses_model, tok,  completer):
        super().__init__(uses_model) 

        self.tok = tok
        self.completer = completer
    

    # TODO: come back and fix this...
    def rollout(self, enc_out, state, task):

        program = self.tok.decode(state.squeeze(), skip_special_tokens=True)

        # while no return ()
        # keep sampling and appending basically
        completions = [None]
        while len(completions) > 0: # if we can still generate completions, keep going
            try:
                completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
                random_action = random.choice(completions)
                program = program + random_action
            except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
                traceback.print_exc()
                break
                
        program = self.tok.encode(program, add_special_tokens=False, return_tensors='pt')
        return program
        
        
    def predict(self, _, state, task, __):
        program = self.tok.decode(state, skip_special_tokens=True)

        prev_program = program.split("\n")[:-1]
        partial_line = program.split("\n")[-1]



        try:
            completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            traceback.print_exc()
            return return_empty_nodes()
    
        if len(completions) ==0:
            return return_empty_nodes()

        prev_program_str = "\n".join(prev_program)
        prev_program_str = prev_program_str + "\n"

        
        completions = [merge_with_overlap(partial_line, x) for x in completions]         
        
        if len(prev_program) != 0:
            completions = [ prev_program_str + x for x in completions]


        completions = [torch.cat((torch.tensor([0, 1]), 
                                  self.tok(x, add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1))) for x in completions]


        priors = [0 for i in range(len(completions))]

        return completions, priors

# generates all actions + priors
class TGMCTSMethod(BaseMethod): 
    
    def __init__(self, uses_model, model, tok, completer):
        super().__init__(uses_model)

        self.model = model
        self.tok = tok 
        self.completer = completer

    
    def rollout(self, enc_out, next_state):
        with torch.no_grad(): 
            output = self.model.generate(encoder_outputs=enc_out, decoder_input_ids=next_state)
        
        return output
    
    def predict(self, enc_out, state, task, prompt_ids):
        program = self.tok.decode(state, skip_special_tokens=True)

        prev_program = program.split("\n")[:-1]
        partial_line = program.split("\n")[-1]



        try:
            completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            traceback.print_exc()
            return return_empty_nodes()
    
        if len(completions) ==0:
            return return_empty_nodes()

        prev_program_str = "\n".join(prev_program)
        prev_program_str = prev_program_str + "\n"

        
        completions = [merge_with_overlap(partial_line, x) for x in completions]         
        
        if len(prev_program) != 0:
            completions = [ prev_program_str + x for x in completions]


        completions = [torch.cat((torch.tensor([0, 1]), 
                                  self.tok(x, add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1))) for x in completions]

        completions_batched = pad_sequence(completions, batch_first=True, padding_value =0, padding_side='right')

        # then compute priors over all.
        priors = compute_log_probs_batched(self.model, prompt_ids .to('cuda'), completions_batched.to ('cuda'))

        return completions, priors

# generates actions based on entropy + priors 
class SplintMCTSMethod(BaseMethod):
    def __init__(self, uses_model, model, tokenizer, completer, tau, k):
        super().__init__(uses_model) 

        self.model = model
        self.completer = completer
        self.tok = tokenizer

        # tau = the threshold for an entropy spike
        self.tau = tau

        # k = the default fan out when not handling entropy spikes.
        self.k = k


    def _fwd_step_encdec(self, enc_out, dec_ids): 
        out = self.model(   encoder_outputs=enc_out,
                            decoder_input_ids=dec_ids)  
        
        return out.logits[:, -1, :]   

    
    
    def _handle_entropy_spike(self, state, enc_out, input_ids): 
        
        program = self.tok.decode(state, skip_special_tokens=True)

        prev_program = program.split("\n")[:-1]
        partial_line = program.split("\n")[-1]



        try:
            completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            traceback.print_exc()
            return return_empty_nodes()
    
        if len(completions) ==0:
            return return_empty_nodes()

        prev_program_str = "\n".join(prev_program)
        prev_program_str = prev_program_str + "\n"

        
        completions = [merge_with_overlap(partial_line, x) for x in completions]         
        
        if len(prev_program) != 0:
            completions = [ prev_program_str + x for x in completions]


        completions = [torch.cat((torch.tensor([0, 1]), 
                                  self.tok(x, add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1))) for x in completions]
        completions_batched = pad_sequence(completions, batch_first=True, padding_value =0, padding_side='right')

        # TODO, shouldnt actually be log probs, need to change it to softmaxs.
        log_ps = compute_log_probs_batched(self.model, input_ids .to('cuda'), completions_batched.to ('cuda'))
        return completions, log_ps


    def _handle_non_entropy_spike(self, state, task, next_tokens):    

        program = self.tok.decode(state. squeeze(), skip_special_tokens=True)
        partial_line = program.split("\n")[-1]


        try:
            completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            traceback.print_exc()
            return return_empty_nodes()
            
        


        
        comps = []
        probs = []

        for k in next_tokens: 
            tok_id    = k.item()
            
            # check to make sure that the completion is within the top; 
            top_k_str = self.tok.decode(tok_id, skip_special_tokens=True)
            line_check = partial_line + top_k_str
            is_valid = any(valid.startswith(line_check) for valid in completions)
            if not is_valid:
                continue
            
            
            new_state = torch.cat([state,
                                   torch.tensor([tok_id], device='cuda')])
            
            comps.append(new_state)
            probs.append(0)

        return comps, probs

    """
    Contains the fanout logic. 
    TODO: remove the input ids.
    """
    def predict(self, enc_out, state, task, input_ids): # so this is where I would return it.
        logits = self._fwd_step_encdec(enc_out, state.unsqueeze(0))
        entropy = entropy_bits(logits).item()
        topk_values, topk_indices = torch.topk(logits, k=self.k, dim=-1)  # Get top-k log-probabilities and their indices

        if entropy > self.tau:
            comps, probs = self._handle_entropy_spike(state, enc_out, input_ids)
        else: 
            comps, probs = self._handle_non_entropy_spike(state, task, topk_indices)

        # format the returns.
        return comps, probs
        
        

    def encode(self, prompt_ids): 
        with torch.no_grad(): # first encode the input once.
            enc_out = self.model.get_encoder()(prompt_ids.unsqueeze(0))
        return enc_out

    def eval(self):
        self.model.eval()

    # perform a rollout from the current state.
    def rollout(self, enc_out, next_state):
        with torch.no_grad(): 
            output = self.model.generate(encoder_outputs=enc_out, decoder_input_ids=next_state)
        
        return output



class Node: 
    def __init__(self, prior= 0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.child_actions = None
        self.children = []
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, state, actions, action_probs):
        state = state.to('cpu')
        actions = [x.to('cpu') for x in actions]
        self.state = state.clone()
        self.child_actions = copy.deepcopy(actions)
        self.children = [Node(prior=prob) for prob in action_probs]

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children])
        actions = self.child_actions
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action_index = np.random.choice(len(actions), p=visit_count_distribution)
            action = actions[action_index]
        return action

    def select_child(self): 
        best_score = -np.inf
        best_action = -1
        best_child = None

        for i in range(len(self.children)):
            score = puct_score(self, self.children[i])
            if score > best_score:
                best_score = score
                best_action = self.child_actions[i]
                best_child = self.children[i]


        return best_action, best_child    

def rollout( state,
            actions,
            enc_out,
            model: BaseMethod,
            env: LineLevelArcEnv,
            task): 


    # need to make a random choice of actions    
    action = random.choice(actions)
    action= torch.cat((state, action))
    program = model.rollout(enc_out, action.unsqueeze(0).to('cuda'), task)
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
                
            value = rollout(state, actions, enc_out, model, env, task) # rollout
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


    for task_id in tasks:
        task = Task.from_json(f'./data/training/{task_id}.json')
        input_ids = torch.tensor(encode_task(task, tok, None)).to('cuda')
        env = LineLevelArcEnv('Salesforce/codet5p-220m',  10, 512, 512, 10, 50000)
        env.set_task(task)
        run_search(env, task, input_ids,  method, time_limit)



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='TGMCTS')


    args = parser.parse_args()

    # TODO: add proper config files
    model = T5ForConditionalGeneration.from_pretrained('./finetune/2025-05-27_17-42-37/checkpoint-1650')
    model.to('cuda')
    tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    sampler   = ProgramSampler(data_path="./data/")
    completer = ProgramCompleter(sampler)
    
    
    if args.method == "MCTS": 
        method = MCTSMethod(uses_model=False, tok=tok, completer=completer)
    elif args.method == "TGMCTS":
        method = TGMCTSMethod(uses_model=True, model=model, tok=tok, completer=completer)
    elif args.method == "SPLINTMCTS":
        method = SplintMCTSMethod(True, model, tok, completer, 0.5, 1)
    else:
        raise ValueError("Method does not exist!")

    run_experiment(method=method,
                   tasks=None,
                   time_limit=(60 * 3),
                   tok=tok)


if __name__ == "__main__": 
    main()